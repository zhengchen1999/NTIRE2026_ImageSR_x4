import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import trunc_normal_
import math

def exists(val):
    return val is not None

def is_empty(t):
    return t.nelement() == 0

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)

def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))
    
    
def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def center_iter(x, means, buckets = None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means
    
class IASA(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_size, zero_init_proj=False):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        if zero_init_proj:
            nn.init.zeros_(self.proj.weight)
        self.group_size = group_size
        
    
    def forward(self, normed_x, idx_last, k_global, v_global):
        x = normed_x
        B, N, _ = x.shape
       
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))
   
        gs = min(N, self.group_size)  # group size
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N
        
        paded_q = torch.cat((q, torch.flip(q[:,N-pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d",ng=ng,h=self.heads)
        paded_k = torch.cat((k, torch.flip(k[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2,2*gs,gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        paded_v = torch.cat((v, torch.flip(v[:,N-pad_n-gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2,2*gs,gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d",h=self.heads)
        out1 = F.scaled_dot_product_attention(paded_q,paded_k,paded_v)
        
        
        k_global = k_global.reshape(1,1,*k_global.shape).expand(B,ng,-1,-1,-1)
        v_global = v_global.reshape(1,1,*v_global.shape).expand(B,ng,-1,-1,-1)
       
        out2 = F.scaled_dot_product_attention(paded_q,k_global,v_global)
        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]
 
        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)
    
        return out
    
class IRCA(nn.Module):
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
      
    def forward(self, normed_x, x_means):
        x = normed_x
        if self.training:
            x_global = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))
        else:
            x_global = x_means

        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)

        return k,v, x_global.detach()
    
    
class TAB(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128,
                 ema_decay = 0.999, attn_type='se', layer_scale_init=1e-4, zero_init=False, use_layer_scale=True):
        super().__init__()

        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens
        
        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim,mlp_dim, layer_scale_init=layer_scale_init, zero_init=zero_init))
        self.irca_attn = IRCA(dim,qk_dim,heads)
        self.iasa_attn = IASA(dim,qk_dim,heads,group_size, zero_init_proj=zero_init)
        self.use_layer_scale = use_layer_scale
        # layer scale for fast convergence and stable residuals
        self.layer_scale_iasa = nn.Parameter(torch.ones(dim) * layer_scale_init)
        self.layer_scale_mlp = nn.Parameter(torch.ones(dim) * layer_scale_init)
        # pyramid multi-scale fusion inside TAB
        self.pyramid = PyramidModule(dim)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim,dim,1, bias=False)

    
    def forward(self, x):
        _,_,h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape
        
        idx_last = torch.arange(N, device=x.device).reshape(1,N).expand(B,-1)
        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat((x, torch.flip(x[:,N-pad_n:N, :], dims=[-2])), dim=-2)
            x_means=torch.mean(rearrange(paded_x, 'b (cnt n) c->cnt (b n) c',cnt=self.num_tokens),dim=-2).detach()   
        else:  
            x_means = self.means.detach()

        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter-1):
                    x_means = center_iter(F.normalize(x,dim=-1), F.normalize(x_means,dim=-1))
                        
                
        k_global, v_global, x_means = self.irca_attn(x, x_means)
        
        with torch.no_grad():
            x_scores = torch.einsum('b i c,j c->b i j', 
                                        F.normalize(x, dim=-1), 
                                        F.normalize(x_means, dim=-1))
            x_belong_idx = torch.argmax(x_scores, dim=-1)
    
            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)
        
        y = self.iasa_attn(x, idx_last,k_global,v_global)
        # optionally apply layer-scale to attention output for stable residuals
        if self.use_layer_scale:
            y = y * self.layer_scale_iasa.view(1, 1, -1)
        y = rearrange(y,'b (h w) c->b c h w',h=h).contiguous()
        y = self.conv1x1(y)
        # apply pyramid multi-scale fusion and residual
        pyr = self.pyramid(y)
        y = y + pyr
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        # apply layer-scale to mlp output
        mlp_out = self.mlp(x, x_size=(h, w))
        if self.use_layer_scale:
            mlp_out = mlp_out * self.layer_scale_mlp.view(1, 1, -1)
        x = mlp_out + x
        
 
        if self.training:
            with torch.no_grad():
                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    self.initted.data.copy_(torch.tensor(True))
                else: 
                    ema_inplace(self.means, new_means, self.ema_decay)
            
    
        return rearrange(x, 'b (h w) c->b c h w',h=h)
        
        
        

def patch_divide(x, step, ps):
    """Crop image into patches.
    Args:
        x (Tensor): Input feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        crop_x (Tensor): Cropped patches.
        nh (int): Number of patches along the horizontal direction.
        nw (int): Number of patches along the vertical direction.
    """
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    """Reverse patches into image.
    Args:
        crop_x (Tensor): Cropped patches.
        x (Tensor): Feature map of shape(b, c, h, w).
        step (int): Divide step.
        ps (int): Patch size.
    Returns:
        output (Tensor): Reversed image.
    """
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU, layer_scale_init=1e-4, zero_init=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)
        # keep SE after FFN for channel recalibration
        self.se = SEBlock(out_features, reduction=4)
        # zero init for last fc to stabilize residual learning (fast converge)
        if zero_init:
            try:
                nn.init.zeros_(self.fc2.weight)
                if self.fc2.bias is not None:
                    nn.init.zeros_(self.fc2.bias)
            except Exception:
                pass

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        x = self.se(x)
        return x


# Note: ECA for sequence removed from ConvFFN; sequence-level attention is provided
# by IRCA/IASA modules. ECA is available as ChannelECA2d for 2D feature maps and
# can be applied at network framework level (see CATANet ctor options).


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for sequence tensors.

    Operates on input of shape (b, n, c). Squeezes across the sequence
    dimension and produces channel-wise gating.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (b, n, c)
        s = x.mean(dim=1)  # (b, c)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.sig(s).unsqueeze(1)  # (b, 1, c)
        return x * s


class ChannelSE2d(nn.Module):
    """Squeeze-and-Excitation block for 2D feature maps.

    Input: (b, c, h, w). Outputs gated feature map same shape.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (b, c, h, w)
        s = self.pool(x)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.sig(s)
        return x * s


class ChannelECA2d(nn.Module):
    """ECA block adapted for 2D features."""
    def __init__(self, channels, reduction=4, gamma=2, b=1):
        super().__init__()
        # compute kernel size heuristically
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 == 1 else t + 1
        if k < 1:
            k = 3
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (b, c, h, w)
        s = self.pool(x).squeeze(-1).transpose(1, 2)  # (b, 1, c)
        y = self.conv(s)  # (b,1,c)
        y = self.sig(y).transpose(1, 2).unsqueeze(-1)  # (b,c,1,1)
        return x * y


class DualPoolAttention2d(nn.Module):
    """Dual Pooling Attention (DPA) for 2D feature maps.

    Computes channel attention using both adaptive average and max pooling
    followed by a small MLP (1x1 convs). Outputs channel-wise gating.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        # branch for avg pool
        self.avg_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True)
        )
        # branch for max pool
        self.max_fc = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (b, c, h, w)
        a = self.avg_fc(x)
        m = self.max_fc(x)
        attn = self.sig(a + m)
        return x * attn


class PyramidModule(nn.Module):
    """Multi-scale depthwise pyramid fusion for TAB.

    Takes input (b, c, h, w) and produces fused output (b, c, h, w).
    """
    def __init__(self, channels, dw_kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in dw_kernel_sizes:
            pad = (k - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=k, padding=pad, groups=channels, bias=False),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            ))
        self.fuse = nn.Conv2d(len(self.branches) * channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        cat = torch.cat(outs, dim=1)
        return self.fuse(cat)


class Attention(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
    """

    def __init__(self, dim, heads, qk_dim, zero_init_proj=False):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        if zero_init_proj:
            nn.init.zeros_(self.proj.weight)
        
        

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
       
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
      
        out = F.scaled_dot_product_attention(q,k,v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class LRSA(nn.Module):
    """Attention module.
    Args:
        dim (int): Base channels.
        num (int): Number of blocks.
        qk_dim (int): Channels of query and key in Attention.
        mlp_dim (int): Channels of hidden mlp in Mlp.
        heads (int): Head numbers of Attention.
    """

    def __init__(self, dim, qk_dim, mlp_dim,heads=1, eca=False):
        super().__init__()
     

        self.layer = nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, qk_dim)),
                PreNorm(dim, ConvFFN(dim, mlp_dim))])
        # optional Channel ECA applied on 2D feature map after patch fusion
        self.eca = ChannelECA2d(dim) if eca else None

    def forward(self, x, ps):
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)
        
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        # apply Channel ECA on 2D feature map if configured
        if self.eca is not None:
            x = self.eca(x)
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c->b c h w', h=h)
        
        return x


    
@ARCH_REGISTRY.register()
class ProCata(nn.Module):
    setting = dict(dim=40, block_num=8, qk_dim=36, mlp_dim=96, heads=4,
                     patch_size=[16, 20, 24, 28, 16, 20, 24, 28],
                     use_channel_se=True, se_reduction=4,
                     use_eca=False, eca_position='mid',
                     use_dpa=False,
                     use_layer_scale=True, layer_scale_init=1e-4, zero_init=True)

    def __init__(self,in_chans=3,n_iters=[5,5,5,5,5,5,5,5],
                 num_tokens=[16,32,64,128,16,32,64,128],
                 group_size=[256,128,64,32,256,128,64,32],
                 upscale: int = 4, **kwargs):
        super().__init__()
        
    
        self.dim = self.setting['dim']
        self.block_num = self.setting['block_num']
        self.patch_size = self.setting['patch_size']
        self.qk_dim = self.setting['qk_dim']
        self.mlp_dim = self.setting['mlp_dim']
        self.upscale = upscale
        self.heads = self.setting['heads']
        
        


        self.n_iters = n_iters
        self.num_tokens = num_tokens
        self.group_size = group_size
        # configurable options can be passed via network opt (kwargs) when building
        self.use_channel_se = kwargs.get('use_channel_se', self.setting.get('use_channel_se', True))
        self.se_reduction = kwargs.get('se_reduction', self.setting.get('se_reduction', 4))
        self.use_eca = kwargs.get('use_eca', self.setting.get('use_eca', False))
        self.eca_position = kwargs.get('eca_position', self.setting.get('eca_position', 'mid'))
        self.use_dpa = kwargs.get('use_dpa', self.setting.get('use_dpa', False))
        self.use_layer_scale = kwargs.get('use_layer_scale', self.setting.get('use_layer_scale', True))
        self.layer_scale_init = kwargs.get('layer_scale_init', self.setting.get('layer_scale_init', 1e-4))
        self.zero_init = kwargs.get('zero_init', self.setting.get('zero_init', False))
    
        #-----------1 shallow--------------
        self.first_conv = nn.Conv2d(in_chans, self.dim, 3, 1, 1)

        #----------2 deep--------------
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
   
        for i in range(self.block_num):
          
            # TAB uses 'eca' in FFN if use_eca and position=='ffn'
            tab_attn_type = 'eca' if (self.use_eca and self.eca_position == 'ffn') else 'se'
            layer_scale_init = self.setting.get('layer_scale_init', 1e-4)
            zero_init = self.setting.get('zero_init', True)
            # decide whether to put ECA inside LRSA
            lrsa_eca = True if (self.use_eca and self.eca_position == 'lrsa') else False
            self.blocks.append(nn.ModuleList([TAB(self.dim, self.qk_dim, self.mlp_dim,
                                                                 self.heads, self.n_iters[i], 
                                                                 self.num_tokens[i],self.group_size[i], attn_type=tab_attn_type, layer_scale_init=layer_scale_init, zero_init=zero_init, use_layer_scale=self.use_layer_scale), 
                                              LRSA(self.dim, self.qk_dim, 
                                                             self.mlp_dim,self.heads, eca=lrsa_eca)]))
            self.mid_convs.append(nn.Conv2d(self.dim, self.dim,3,1,1))
            if self.use_channel_se:
                # channel attention for 2D features: choose ECA or SE according to config
                if self.use_dpa:
                    self.mid_convs.append(DualPoolAttention2d(self.dim, reduction=self.se_reduction))
                elif self.use_eca and self.eca_position == 'mid':
                    self.mid_convs.append(ChannelECA2d(self.dim, reduction=self.se_reduction))
                else:
                    self.mid_convs.append(ChannelSE2d(self.dim, reduction=self.se_reduction))
            else:
                # placeholder to keep indexing consistent
                self.mid_convs.append(nn.Identity())
            
        #----------3 reconstruction---------
        
      
     
        if upscale == 4:
            self.upconv1 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(self.dim, self.dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv2d(self.dim, self.dim * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
    
        self.last_conv = nn.Conv2d(self.dim, in_chans, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(self.block_num):
            residual = x
      
            global_attn,local_attn = self.blocks[i]
            
            x = global_attn(x)
            
            x = local_attn(x, self.patch_size[i])
            # apply mid conv then optional channel SE
            mid_conv = self.mid_convs[2 * i] if self.use_channel_se else self.mid_convs[i]
            mid_se = self.mid_convs[2 * i + 1] if self.use_channel_se else None
            if self.use_channel_se and mid_se is not None:
                x = mid_conv(x)
                x = mid_se(x)
            else:
                x = mid_conv(x)

            x = residual + x
        return x
        
    def forward(self, x):
        
        if self.upscale != 1: 
            base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        else: 
            base = x
        x = self.first_conv(x)
        
   
        x = self.forward_features(x) + x
    
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))
        out = self.last_conv(out) + base
       
    
        return out
    
    
    def __repr__(self):
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        return '#Params of {}: {:<.4f} [K]'.format(self._get_name(),
                                                      num_parameters / 10 ** 3) 
  
  


if __name__ == '__main__':


    model = ProCata(upscale=3).cuda()
    x = torch.randn(2, 3, 128, 128).cuda()
    print(model)
 
  
