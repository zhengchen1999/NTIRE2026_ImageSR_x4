import torchvision.transforms.functional as F
import sys
sys.modules['torchvision.transforms.functional_tensor'] = F
import torch
import numpy as np
from .hat.archs.hat_arch import HAT

SCALE = 4
WINDOW_SIZE = 16
TILE = 256
OVERLAP = 32
USE_TTA = True


class Model:
    def __init__(self, model_path, device):
        self.device = device

        self.model = HAT(
            upscale=4, in_chans=3, img_size=64, window_size=16,
            compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
            overlap_ratio=0.5, img_range=1.,
            depths=[6,6,6,6,6,6], embed_dim=180,
            num_heads=[6,6,6,6,6,6], mlp_ratio=2,
            upsampler='pixelshuffle', resi_connection='1conv'
        ).to(device)

        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get('params_ema') or ckpt.get('params') or ckpt
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    # ---------------- PAD ----------------
    def pad(self, x):
        _, _, h, w = x.shape
        ph = (WINDOW_SIZE - h % WINDOW_SIZE) % WINDOW_SIZE
        pw = (WINDOW_SIZE - w % WINDOW_SIZE) % WINDOW_SIZE
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode='reflect')
        return x, ph, pw

    def unpad(self, x, ph, pw):
        if ph:
            x = x[:, :, :-ph*SCALE, :]
        if pw:
            x = x[:, :, :, :-pw*SCALE]
        return x

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)

    # ---------------- TTA ----------------
    def tta(self, x):
        outs = []
        for fn, inv in [
            (lambda x: x, lambda x: x),
            (lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-1])),
            (lambda x: torch.flip(x, [-2]), lambda x: torch.flip(x, [-2])),
            (lambda x: torch.flip(x, [-1,-2]), lambda x: torch.flip(x, [-1,-2]))
        ]:
            outs.append(inv(self.forward(fn(x))))
        return torch.stack(outs).mean(0)

    # ---------------- GAUSSIAN ----------------
    def get_weight(self, h, w):
        y = torch.linspace(-1, 1, h).view(h, 1)
        x = torch.linspace(-1, 1, w).view(1, w)
        weight = torch.exp(-(x**2 + y**2) / 0.5)
        return weight.to(self.device)

    # ---------------- TILE ----------------
    def tile(self, x):
        b, c, h, w = x.shape

        out = torch.zeros(b, c, h*SCALE, w*SCALE).to(self.device)
        weight_map = torch.zeros_like(out)

        stride = TILE - OVERLAP

        for i in range(0, h, stride):
            for j in range(0, w, stride):
                patch = x[:, :, i:min(i+TILE, h), j:min(j+TILE, w)]

                patch, ph, pw = self.pad(patch)

                if USE_TTA:
                    pred = self.tta(patch)
                else:
                    pred = self.forward(patch)

                pred = self.unpad(pred, ph, pw)

                ph2, pw2 = pred.shape[2], pred.shape[3]

                w_patch = self.get_weight(ph2, pw2).unsqueeze(0).unsqueeze(0)

                out[:, :, i*SCALE:i*SCALE+ph2,
                           j*SCALE:j*SCALE+pw2] += pred * w_patch

                weight_map[:, :, i*SCALE:i*SCALE+ph2,
                                 j*SCALE:j*SCALE+pw2] += w_patch

        out = out / torch.clamp(weight_map, min=1e-6)
        return out

    # ---------------- MAIN ----------------
    @torch.no_grad()
    def process(self, img):
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(self.device)

        img, ph, pw = self.pad(img)

        out = self.tile(img)

        out = self.unpad(out, ph, pw)

        out = out.squeeze().permute(1,2,0).cpu().numpy()
        out = np.clip(out, 0, 1)
        out = np.nan_to_num(out)

        out = (out * 255.0).round().astype(np.uint8)

        return out
