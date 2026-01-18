# easr.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_act(name: str = "silu") -> nn.Module:
    name = name.lower()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name in ("relu",):
        return nn.ReLU(inplace=True)
    if name in ("lrelu", "leakyrelu"):
        return nn.LeakyReLU(0.1, inplace=True)
    if name in ("gelu",):
        return nn.GELU()
    raise ValueError(f"Unsupported act: {name}")


class SpatialSoftmax2d(nn.Module):
    """Softmax over spatial dims (H*W) per-sample per-channel."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        x = torch.softmax(x, dim=-1)
        return x.view(b, c, h, w)


class Sobel(nn.Module):
    """Sobel operator with fixed kernels. Input can be 1 or C channels."""
    def __init__(self, in_channels: int, reduce_to_gray: bool = True):
        super().__init__()
        self.reduce = None
        if reduce_to_gray and in_channels != 1:
            # 学习到的 1x1 降维，稳定于不同模态
            self.reduce = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
            in_channels = 1

        # 深度可分离方式对每个通道做相同的 Sobel
        self.sobel_x = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                 groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                 groups=in_channels, bias=False)

        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]])
        with torch.no_grad():
            for m, k in ((self.sobel_x, kx), (self.sobel_y, ky)):
                w = torch.zeros_like(m.weight)
                # 每个通道共享同一核
                for c in range(w.shape[0]):
                    w[c, 0, :, :] = k
                m.weight.copy_(w)

        # 冻结 Sobel 权重
        for p in self.sobel_x.parameters():
            p.requires_grad = False
        for p in self.sobel_y.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduce is not None:
            x = self.reduce(x)  # Bx1xHxW
        gx = self.sobel_x(x)
        gy = self.sobel_y(x)
        # 边缘幅值
        g = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return g


class EASR(nn.Module):
    """
    Edge-Aware Super-Resolution (EASR)
    输入:  F ∈ R^{B×C×H×W}
    输出:  F_out ∈ R^{B×C_out×(rH)×(rW)}

    分支:
      1) Edge 分支: Sobel -> Conv (C_edge*r^2) -> PixelShuffle(r)
      2) Attention 分支: 1x1 Conv -> 上采样到 rH×rW -> Spatial Softmax/Sigmoid -> 与 Edge 分支逐点相乘
      3) Context 分支: 空洞卷积 (dilation=d) -> 上采样到 rH×rW -> 1x1 调整到 C_ctx
      4) 融合: concat([edge_refined, context], dim=1) -> 1x1 conv -> C_out
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 upscale: int = 2,
                 edge_channels: int = None,
                 ctx_channels: int = None,
                 dilation: int = 2,
                 act: str = "silu",
                 attention: str = "softmax"):  # or "sigmoid"
        super().__init__()
        assert upscale in (2, 3, 4), "upscale r must be 2/3/4"
        self.r = upscale
        C = in_channels
        C_edge = edge_channels or C // 2 or 16
        C_ctx  = ctx_channels  or C // 2 or 16
        C_out  = out_channels  or C

        self.sobel = Sobel(C, reduce_to_gray=True)

        # edge: conv -> pixelshuffle -> (B,C_edge, rH, rW)
        self.edge_proj = nn.Sequential(
            nn.Conv2d(1, C_edge * self.r * self.r, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C_edge * self.r * self.r),
            _make_act(act)
        )
        self.ps = nn.PixelShuffle(self.r)

        # attention: 1x1 -> 上采样到 rH×rW -> 空间归一化
        self.att_logits = nn.Conv2d(C, 1, kernel_size=1, bias=True)
        self.attention = attention.lower()
        if self.attention == "softmax":
            self.att_norm = SpatialSoftmax2d()
        elif self.attention == "sigmoid":
            self.att_norm = nn.Sigmoid()
        else:
            raise ValueError("attention must be 'softmax' or 'sigmoid'")

        # context: 空洞卷积 + 上采样 + 1x1 到 C_ctx
        self.ctx = nn.Sequential(
            nn.Conv2d(C, C_ctx, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_ctx),
            _make_act(act),
        )

        # fuse: concat(edge_refined, context_up) -> 1x1 -> out
        self.fuse = nn.Sequential(
            nn.Conv2d(C_edge + C_ctx, C_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_out),
            _make_act(act)
        )

    def forward(self, F_in: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F_in.shape
        rH, rW = H * self.r, W * self.r

        # Edge branch
        E0 = self.sobel(F_in)                              # Bx1xHxW
        E1 = self.edge_proj(E0)                            # Bx(Ce*r^2)xHxW
        E  = self.ps(E1)                                   # BxCe x rH x rW

        # Attention branch -> 1x rH x rW
        Alog = self.att_logits(F_in)                       # Bx1xHxW
        A    = F.interpolate(Alog, size=(rH, rW), mode="bilinear", align_corners=False)
        if self.attention == "softmax":
            A = self.att_norm(A)                           # softmax over spatial
        else:
            A = self.att_norm(A)                           # sigmoid in [0,1]

        E_ref = E * A                                      # BxCe x rH x rW

        # Context branch
        U = self.ctx(F_in)                                 # BxCctx x H x W
        U = F.interpolate(U, size=(rH, rW), mode="bilinear", align_corners=False)

        # Fuse
        F_cat = torch.cat([E_ref, U], dim=1)               # Bx(Ce+Cctx) x rH x rW
        F_out = self.fuse(F_cat)
