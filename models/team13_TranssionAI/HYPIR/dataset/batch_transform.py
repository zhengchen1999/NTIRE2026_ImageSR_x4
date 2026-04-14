from typing import Any, overload, Dict, List
import torch
from torch.nn import functional as F

class BatchTransform:
    @overload
    def __call__(self, batch: Any) -> Any: ...

class IdentityBatchTransform(BatchTransform):
    def __call__(self, batch: Any) -> Any:
        return batch

class RealESRGANBatchTransform(BatchTransform):
    def __init__(
        self,
        hq_key,
        extra_keys,

        # ✅ 新增：支持 YAML 里传入这些参数（你当前配置里就有）
        scale: int = 4,
        interpolation: str = "bicubic",
        antialias: bool = True,
        align_corners: bool = False,

        # 下面所有参数全部保留，为了兼容旧 Config 文件，但不再实际使用
        use_sharpener=False,
        queue_size=0,
        resize_prob=None,
        resize_range=None,
        gray_noise_prob=None,
        gaussian_noise_prob=None,
        noise_range=None,
        poisson_scale_range=None,
        jpeg_range=None,
        second_blur_prob=None,
        stage2_scale=None,
        resize_prob2=None,
        resize_range2=None,
        gray_noise_prob2=None,
        gaussian_noise_prob2=None,
        noise_range2=None,
        poisson_scale_range2=None,
        jpeg_range2=None,

        resize_back: bool = False,  # 这个参数很重要，决定 LQ 是小图还是插值回原本大小的模糊图
        **kwargs,  # ✅ 兜底：避免你未来 config 多写字段再次炸
    ):
        super().__init__()
        self.hq_key = hq_key
        self.extra_keys = extra_keys
        self.resize_back = resize_back

        # ✅ scale 优先使用显式 scale；如果你想兼容旧的 stage2_scale 也可以启用下面两行
        # if stage2_scale is not None:
        #     scale = stage2_scale

        self.scale = int(scale)
        assert self.scale in [2, 3, 4, 8], f"Unexpected scale={self.scale}, please use common scales like 2/3/4/8."

        self.interpolation = str(interpolation).lower()
        # torch 支持的常用 mode：nearest / bilinear / bicubic / area
        assert self.interpolation in ["nearest", "bilinear", "bicubic", "area"], \
            f"Unsupported interpolation mode: {self.interpolation}"

        self.antialias = bool(antialias)
        self.align_corners = bool(align_corners)

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor | List[str]]) -> Dict[str, torch.Tensor | List[str]]:
        """
        执行单纯的 scale 倍 Bicubic（或其它插值模式）降采样
        """
        # 1) 获取 HQ 图像
        hq = batch[self.hq_key]  # [B, C, H, W]

        # 2) 降采样（x1/scale）
        # 注意：align_corners 只对 linear/bilinear/bicubic 有意义，nearest/area 会被忽略
        # antialias 在某些老 torch 版本不支持，这里做 try/except 兼容
        sf = 1.0 / self.scale
        try:
            lq = F.interpolate(
                hq, scale_factor=sf, mode=self.interpolation,
                align_corners=self.align_corners if self.interpolation in ["bilinear", "bicubic"] else None,
                antialias=self.antialias if self.interpolation in ["bilinear", "bicubic"] else False,
            )
        except TypeError:
            # 兼容老版本 torch：没有 antialias 参数
            lq = F.interpolate(
                hq, scale_factor=sf, mode=self.interpolation,
                align_corners=self.align_corners if self.interpolation in ["bilinear", "bicubic"] else None,
            )

        # 3) (可选) 插值回原本大小（down×scale -> up×scale），保持 LQ 与 HQ 同尺寸
        if self.resize_back:
            h, w = hq.shape[2], hq.shape[3]
            try:
                lq = F.interpolate(
                    lq, size=(h, w), mode=self.interpolation,
                    align_corners=self.align_corners if self.interpolation in ["bilinear", "bicubic"] else None,
                    antialias=self.antialias if self.interpolation in ["bilinear", "bicubic"] else False,
                )
            except TypeError:
                lq = F.interpolate(
                    lq, size=(h, w), mode=self.interpolation,
                    align_corners=self.align_corners if self.interpolation in ["bilinear", "bicubic"] else None,
                )

        # 4) clamp 到 [0, 1]
        lq = torch.clamp(lq, 0, 1)

        # 5) 组装返回
        return_batch = {"GT": hq, "LQ": lq}
        for k in self.extra_keys:
            if k in batch:
                return_batch[k] = batch[k]
        return return_batch