
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class BridgeConvV2(nn.Module):
#     """
#     Bridge‐Oriented Dynamic Steerable Gabor Convolution
#     只需 in_ch, out_ch，内部固定 scales=[(4,2),(8,4)], n_dirs=4, ksz=15
#     """
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.in_ch, self.out_ch = in_ch, out_ch
#         scales  = [(4, 2), (8, 4)]
#         n_dirs  = 4
#         self.ksz = 15  # Gabor 核尺寸

#         # 1）Gabor kernels
#         def gabor_kernel(lmbd, sigma, theta, size):
#             cos_t, sin_t = math.cos(theta), math.sin(theta)
#             grid = torch.arange(size, dtype=torch.float32) - size//2
#             y, x = torch.meshgrid(grid, grid, indexing='ij')
#             xr = x * cos_t + y * sin_t
#             yr = -x * sin_t + y * cos_t
#             return torch.exp(-(xr**2 + yr**2)/(2*sigma**2)) * torch.cos(2*math.pi * xr / lmbd)

#         kernels = []
#         thetas = [i*math.pi/n_dirs for i in range(n_dirs)]
#         for lmbd, sigma in scales:
#             for theta in thetas:
#                 kernels.append(gabor_kernel(lmbd, sigma, theta, self.ksz))
#         self.register_buffer('gabor_kernels', torch.stack(kernels).unsqueeze(1))  # [M,1,ksz,ksz]
#         self.M = len(kernels)

#         # 2）融合与激活
#         self.fuse = nn.Conv2d(in_ch * self.M, in_ch, 1, bias=False)
#         self.bn   = nn.BatchNorm2d(in_ch)
#         self.act  = nn.SiLU()

#         # 3）方向预测
#         mid = max(in_ch//4, 1)
#         self.angle_pred = nn.Sequential(
#             nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(mid),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid, n_dirs, 1)
#         )

#         # 4）主支残差下采（in_ch→out_ch）
#         self.res_proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

#         # 5）主支输出投影
#         self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # 1）下采到 H/2×W/2
#         x_ds = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

#         # 2）方向预测
#         angles = F.softmax(self.angle_pred(x), dim=1)  # [B, n_dirs, H2, W2]

#         # 3）depthwise Gabor + padding
#         M, ks = self.M, self.ksz
#         w = self.gabor_kernels.repeat(1, C, 1, 1).view(M*C, 1, ks, ks)
#         x_rep = x_ds.repeat(1, M, 1, 1)                         # [B,C*M,H2,W2]
#         y = F.conv2d(x_rep, w, groups=C*M, padding=ks//2)       # [B,C*M,H2,W2]
#         B2, CM, H2, W2 = y.shape
#         y = y.view(B2, M, C, H2, W2)                            # [B,M,C,H2,W2]

#         # 4）动态融合
#         ang = angles.unsqueeze(2).repeat(1, 1, M//angles.size(1), 1, 1)
#         ang = ang.view(B2, M, 1, H2, W2)                        # [B,M,1,H2,W2]
#         fused = (y * ang).view(B2, M*C, H2, W2)                 # [B,C*M,H2,W2]
#         fused = self.act(self.bn(self.fuse(fused)))            # [B,C,H2,W2]

#         # 5）投影到 out_ch
#         out_main = self.proj(fused)                            # [B,out_ch,H2,W2]

#         # 6）残差分支
#         res = self.res_proj(x_ds)                              # [B,out_ch,H2,W2]

#         return out_main + res


#-----------------------------------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class BridgeConvV2(nn.Module):
#     """
#     轻量化版 Bridge‐Oriented Dynamic Steerable Gabor Convolution
#     + Softmax 温度调节
#     + 最大权重门控
#     + 背景抑制
#     + 自动对齐输入/权重 dtype，兼容混合精度
#     """
#     def __init__(self, in_ch, out_ch, tau=0.5, thresh=0.3):
#         super().__init__()
#         self.in_ch, self.out_ch = in_ch, out_ch
#         self.tau = tau
#         self.thresh = thresh

#         # Gabor 参数：one scale，4 方向，核尺寸 7
#         scale = (8, 4)
#         n_dirs = 4
#         ksz = 7
#         self.ksz = ksz

#         # 1) 预定义 Gabor 核
#         def gabor_kernel(lmbd, sigma, theta, size):
#             grid = torch.arange(size, dtype=torch.float32) - size//2
#             y, x = torch.meshgrid(grid, grid, indexing='ij')
#             xr = x*math.cos(theta) + y*math.sin(theta)
#             return torch.exp(-(xr**2)/(2*sigma**2)) * torch.cos(2*math.pi*xr/lmbd)

#         thetas = [i*math.pi/n_dirs for i in range(n_dirs)]
#         kernels = [gabor_kernel(scale[0], scale[1], th, ksz) for th in thetas]
#         self.register_buffer('gabor_kernels', torch.stack(kernels).unsqueeze(1))  # [M,1,ksz,ksz]
#         self.M = len(kernels)

#         # 2) 融合：depthwise + pointwise
#         self.dw_fuse = nn.Conv2d(self.M, self.M, 1, groups=self.M, bias=False)
#         self.pw_fuse = nn.Conv2d(self.M, in_ch, 1, bias=False)
#         self.bn      = nn.BatchNorm2d(in_ch)
#         self.act     = nn.SiLU()

#         # 3) 方向预测
#         mid = max(in_ch//8, 1)
#         self.angle_pred = nn.Sequential(
#             nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(mid),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid, n_dirs, 1)
#         )

#         # 4) 背景抑制模块
#         self.bg_conv = nn.Sequential(
#             nn.Conv2d(in_ch, in_ch//2, 3, padding=1, bias=False),
#             nn.BatchNorm2d(in_ch//2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_ch//2, 1, 1),
#             nn.Sigmoid()
#         )

#         # 5) 主干 & 残差下采
#         self.proj     = nn.Conv2d(in_ch, out_ch, 1, bias=False)
#         self.res_proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

#     def forward(self, x):
#         # 1) 下采到 H/2×W/2
#         x_ds = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
#         B, C, H2, W2 = x_ds.shape

#         # 2) 方向预测 + 温度调节 softmax
#         logits = self.angle_pred(x)                        # [B,n_dirs,H2,W2]
#         angles = F.softmax(logits / self.tau, dim=1)       # [B,n_dirs,H2,W2]

#         # 3) 最大权重门控 mask
#         w_max, _ = angles.max(dim=1, keepdim=True)         # [B,1,H2,W2]
#         mask = (w_max > self.thresh).float()

#         # 4) collapse 通道 → 灰度图
#         x_gray = x_ds.mean(dim=1, keepdim=True)            # [B,1,H2,W2]

#         # 5) 解决 dtype 不匹配
#         # dtype_k = self.gabor_kernels.dtype
#         # dev_k   = self.gabor_kernels.device
#         # x_gray  = x_gray.to(device=dev_k, dtype=dtype_k)

#         # 6) Gabor 卷积
#         y = F.conv2d(x_gray, self.gabor_kernels, padding=self.ksz//2)  # [B,M,H2,W2]

#         # 7) 动态融合
#         angles = angles.to(dtype=y.dtype, device=y.device)
#         fused = y * angles                                        # [B,M,H2,W2]

#         # 8) 融合卷积 & BN
#         target_dtype = self.dw_fuse.weight.dtype
#         fused = fused.to(dtype=target_dtype, device=y.device)
#         fused = self.dw_fuse(fused)                               # [B,M,H2,W2]
#         fused = self.pw_fuse(fused)                               # [B,C,H2,W2]
#         fused = self.bn(fused)
#         fused = self.act(fused)
#         fused = fused * mask.to(dtype=fused.dtype, device=fused.device)

#         # 9) 背景抑制：预测背景强度并抑制
#         bg_map = self.bg_conv(fused)      # [B,1,H2,W2]
#         fused = fused * (1 - bg_map)

#         # 10) 主干 & 残差输出
#         out_main = self.proj(fused)
#         out_res  = self.res_proj(x_ds.to(dtype=out_main.dtype, device=out_main.device))
#         return out_main + out_res





#-------------------------------------------------------------------------------------------------



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class BridgeConvV2(nn.Module):
#     """
#     轻量化版 Bridge‐Oriented Dynamic Steerable Gabor Convolution
#     - 仅 one scale=[(8,4)]、n_dirs=4、ksz=7
#     - collapse 通道后做 Gabor，显著减小计算量
#     """
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.in_ch, self.out_ch = in_ch, out_ch
#         scale = (8, 4)
#         n_dirs = 4
#         ksz = 7
#         self.ksz = ksz

#         # 1）预定义 Gabor 核
#         def gabor_kernel(lmbd, sigma, theta, size):
#             grid = torch.arange(size, dtype=torch.float32) - size//2
#             y, x = torch.meshgrid(grid, grid, indexing='ij')
#             xr = x*math.cos(theta) + y*math.sin(theta)
#             return torch.exp(-(xr**2)/(2*sigma**2)) * torch.cos(2*math.pi*xr/lmbd)

#         thetas = [i*math.pi/n_dirs for i in range(n_dirs)]
#         kernels = [gabor_kernel(scale[0], scale[1], th, ksz) for th in thetas]
#         self.register_buffer('gabor_kernels', torch.stack(kernels).unsqueeze(1))  # [M,1,ksz,ksz]
#         self.M = len(kernels)

#         # 2）融合：depthwise+pointwise
#         self.dw_fuse = nn.Conv2d(self.M, self.M, 1, groups=self.M, bias=False)
#         self.pw_fuse = nn.Conv2d(self.M, in_ch, 1, bias=False)
#         self.bn = nn.BatchNorm2d(in_ch)
#         self.act = nn.SiLU()

#         # 3）方向预测：in_ch → mid → n_dirs
#         mid = max(in_ch//8, 1)
#         self.angle_pred = nn.Sequential(
#             nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(mid),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid, n_dirs, 1)
#         )

#         # 4）主干和残差下采
#         self.proj     = nn.Conv2d(in_ch, out_ch, 1, bias=False)
#         self.res_proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

#     def forward(self, x):
#         # 下采到 H/2, W/2
#         x_ds = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

#         # 方向权重
#         angles = F.softmax(self.angle_pred(x), dim=1)  # [B, n_dirs, H2, W2]
#         B, C, H2, W2 = x_ds.shape

#         # collapse 通道 → 灰度图
#         x_gray = x_ds.mean(dim=1, keepdim=True)        # [B,1,H2,W2]

#         # Gabor 卷积
#         y = F.conv2d(x_gray, self.gabor_kernels, padding=self.ksz//2)  # [B,M,H2,W2]

#         # 动态融合
#         ang = angles  # [B,M,H2,W2]
#         fused = y * ang                                   # [B,M,H2,W2]
#         fused = self.dw_fuse(fused)                       # [B,M,H2,W2]
#         fused = self.act(self.bn(self.pw_fuse(fused)))    # [B,C,H2,W2]

#         # 主干与残差输出
#         out_main = self.proj(fused)                       # [B,out_ch,H2,W2]
#         out_res  = self.res_proj(x_ds)                    # [B,out_ch,H2,W2]
#         return out_main + out_res



import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BridgeConvV2(nn.Module):
    """
    创新版 BridgeConvV2：
    - 多尺度 Gabor steerable filters + 动态方向权重
    - Vesselness 分支 (LoG) 用于桥梁线状结构增强
    - 自适应残差背景抑制
    - 自适应主/残差融合
    - 轻量化分支并行设计，插拔式替换 YOLOv8 Conv
    """
    def __init__(self, in_ch, out_ch,
                 gabor_scales=[(8,4),(16,8)], n_dirs=4, ksz=7,
                 log_sigma=2.0, alpha_v=0.8):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.alpha_v = alpha_v
        # --- Gabor 基函数 ---
        kernels = []
        for lmbd, sigma in gabor_scales:
            for i in range(n_dirs):
                theta = i * math.pi / n_dirs
                grid = torch.arange(ksz, dtype=torch.float32) - ksz//2
                y, x = torch.meshgrid(grid, grid, indexing='ij')
                xr = x*math.cos(theta) + y*math.sin(theta)
                kern = torch.exp(-xr**2/(2*sigma**2)) * torch.cos(2*math.pi*xr/lmbd)
                kernels.append(kern)
        self.M = len(kernels)
        self.register_buffer('gabor_kernels', torch.stack(kernels).unsqueeze(1))  # [M,1,ksz,ksz]

        # --- LoG (Laplacian of Gaussian) Vesselness 分支 ---
        # 生成 LoG 核
        r = torch.arange(ksz, dtype=torch.float32) - ksz//2
        Y, X = torch.meshgrid(r, r, indexing='ij')
        rsq = X**2 + Y**2
        sigma = log_sigma
        # LoG 核公式：((r^2 - 2σ^2)/(σ^4)) * exp(-r^2/(2σ^2))
        log_kern = ((rsq - 2*sigma**2) / (sigma**4)) * torch.exp(-rsq/(2*sigma**2))
        self.register_buffer('log_kernel', log_kern.unsqueeze(0).unsqueeze(0))  # [1,1,ksz,ksz]

        # --- 动态方向预测 ---
        mid = max(in_ch//8,1)
        self.angle_pred = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, n_dirs, 1)
        )

        # --- Gabor 输出融合 ---
        self.dw = nn.Conv2d(self.M, self.M, 1, groups=self.M, bias=False)
        self.pw = nn.Conv2d(self.M, in_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.SiLU()

        # --- 主/残分支 ---
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.res_proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

        # --- 背景抑制和自适应融合 ---
        self.bg_gate = nn.Conv2d(out_ch, 1, 1)
        self.fuse_gate = nn.Conv2d(out_ch*2, 1, 1)

    def forward(self, x):
        # 下采样至 H/2
        x_ds = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        # 灰度化
        x_gray = x_ds.mean(dim=1, keepdim=True)

        # --- Gabor 分支 ---
        y = F.conv2d(x_gray, self.gabor_kernels, padding=self.gabor_kernels.shape[-1]//2)
        att = F.softmax(self.angle_pred(x)/1.0, dim=1)  # temperature=1
        att_rep = att.repeat_interleave(len(self.gabor_kernels)//att.shape[1], dim=1)
        gabor_feat = y * att_rep
        gabor_feat = self.dw(gabor_feat)
        gabor_feat = self.act(self.bn(self.pw(gabor_feat)))  # [B,in_ch,H2,W2]

        # --- Vesselness 分支 ---
        v_map = F.conv2d(x_gray, self.log_kernel, padding=self.log_kernel.shape[-1]//2)
        v_map = torch.abs(v_map)
        v_map = v_map / (v_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6)
        # 融合到特征中
        fused = gabor_feat * (1 + self.alpha_v * v_map)

        # --- 主/残分支 ---
        out_main = self.proj(fused)
        out_res = self.res_proj(x_ds)

        # 背景抑制
        bg_gate = torch.sigmoid(self.bg_gate(out_res))
        out_res = out_res * bg_gate

        # 自适应融合
        cat = torch.cat([out_main, out_res], dim=1)
        gate = torch.sigmoid(self.fuse_gate(cat))
        y= gate * out_main + (1-gate) * out_res

        if not self.training:               # 只在推理 / eval 模式抓
            self._dbg = {
                'gabor': gabor_feat.detach(),   # (B, C_in, H/2, W/2)
                'vmap' : v_map.detach(),        # (B, 1,    H/2, W/2)
                'fused': fused.detach(),        # (B, C_in, H/2, W/2)
                'out'  : y.detach()             # (B, C_out,H/2, W/2)
        }
        return y


 

    
    
    

