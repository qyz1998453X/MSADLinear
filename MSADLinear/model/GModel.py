"""
GModel v3: Multi-Scale Adaptive DLinear + RevIN + Cross-Channel Target Aggregation

Key upgrades over v2:
1) RevIN normalization (often boosts forecasting on public datasets)
2) Multi-scale adaptive trend (keep your v2 idea)
3) Crucial: dynamic channel attention to aggregate multivariate information into target prediction
   -> meteorology/time features can truly help PM2.5

Input:  x (B, L, C)
Output: y_hat (B, pred_len)  # target only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# Utils
# ======================================================
def _make_odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)

def _clip_kernel(k: int, seq_len: int) -> int:
    k = _make_odd(int(k))
    if k > seq_len:
        # choose <= seq_len and odd
        k = seq_len if (seq_len % 2 == 1) else (seq_len - 1)
        k = max(3, k)
    return k


# ======================================================
# RevIN (Reversible Instance Normalization)
# ======================================================
class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series.
    Normalize per-sample, per-channel across time dimension.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta  = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        self._cached_mean = None
        self._cached_std = None

    def forward(self, x: torch.Tensor, mode: str):
        """
        x: (B, L, C)
        mode: "norm" or "denorm"
        """
        if mode == "norm":
            mean = x.mean(dim=1, keepdim=True)                          # (B,1,C)
            std  = x.std(dim=1, keepdim=True, unbiased=False) + self.eps # (B,1,C)
            self._cached_mean = mean
            self._cached_std = std

            x = (x - mean) / std
            if self.affine:
                x = x * self.gamma + self.beta
            return x

        elif mode == "denorm":
            mean = self._cached_mean
            std  = self._cached_std
            if (mean is None) or (std is None):
                raise RuntimeError("RevIN: call mode='norm' before mode='denorm'")

            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * std + mean
            return x

        else:
            raise ValueError("RevIN mode must be 'norm' or 'denorm'")

    def denorm_target(self, y: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
        """
        y: (B, pred_len) in normalized space
        Return: denormed y in raw space using cached target channel stats
        """
        mean = self._cached_mean[:, :, target_idx]  # (B,1)
        std  = self._cached_std[:, :, target_idx]   # (B,1)

        # reverse affine on target
        if self.affine:
            gamma = self.gamma[:, :, target_idx]  # (1,1)
            beta  = self.beta[:, :, target_idx]   # (1,1)
            y = (y - beta) / (gamma + self.eps)

        y = y * std + mean
        return y


# ======================================================
# Multi-Scale Adaptive Moving Average (Trend Extractor v3)
#   - keep: per-channel learnable smoothing kernel (depthwise)
#   - improve: gating uses lightweight temporal encoder + mean/std/last/slope
# ======================================================
class MultiScaleAdaptiveMovingAverageV3(nn.Module):
    def __init__(
        self,
        channels: int,
        seq_len: int,
        kernel_sizes=(7, 25, 49),
        gating_hidden: int = 64,
        gate_dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len

        ks = [_clip_kernel(k, seq_len) for k in kernel_sizes]
        self.kernel_sizes = sorted(list(dict.fromkeys(ks)))
        self.num_scales = len(self.kernel_sizes)

        # per-channel kernel logits for each scale: (C, K)
        self.kernel_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(channels, k)) for k in self.kernel_sizes
        ])

        # lightweight temporal encoder to enrich gating signals
        # depthwise conv -> GELU -> global pooling => (B, C)
        self.temporal_enc = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # gating features per sample: [mean, std, last, slope, enc] => 5C
        in_dim = 5 * channels
        hidden = max(32, min(gating_hidden, in_dim))
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(gate_dropout),
            nn.Linear(hidden, channels * self.num_scales),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)  (L must equal seq_len due to linear heads)
        return trend: (B, L, C)
        """
        B, L, C = x.shape
        if L != self.seq_len:
            raise ValueError(f"seq_len mismatch: x has {L}, expected {self.seq_len}")
        if C != self.channels:
            raise ValueError(f"channels mismatch: x has {C}, expected {self.channels}")

        # (B, C, L)
        x_t = x.permute(0, 2, 1)

        # multi-scale depthwise smoothing
        trends = []
        for logits, k in zip(self.kernel_logits, self.kernel_sizes):
            pad = (k - 1) // 2
            x_pad = F.pad(x_t, (pad, pad), mode="replicate")  # (B, C, L+2pad)

            w = torch.softmax(logits, dim=-1)     # (C, k)
            weight = w.unsqueeze(1)               # (C, 1, k)

            trend_k = F.conv1d(x_pad, weight=weight, bias=None, stride=1, padding=0, groups=C)  # (B,C,L)
            trends.append(trend_k)

        trends = torch.stack(trends, dim=2)  # (B, C, S, L)

        # gating features
        mean = x_t.mean(dim=-1)                          # (B, C)
        std  = x_t.std(dim=-1, unbiased=False)           # (B, C)
        last = x_t[:, :, -1]                             # (B, C)
        slope = (x_t[:, :, -1] - x_t[:, :, 0]) / max(1, (L - 1))  # (B, C)
        enc = self.temporal_enc(x_t).squeeze(-1)         # (B, C)

        feat = torch.cat([mean, std, last, slope, enc], dim=-1)   # (B, 5C)

        gate_logits = self.gate_mlp(feat).view(B, C, self.num_scales)  # (B,C,S)
        gate = torch.softmax(gate_logits, dim=-1)                     # (B,C,S)

        trend = (trends * gate.unsqueeze(-1)).sum(dim=2)  # (B,C,L)
        return trend.permute(0, 2, 1)                      # (B,L,C)


# ======================================================
# Series Decomposition
# ======================================================
class SeriesDecompositionMSV3(nn.Module):
    def __init__(self, channels: int, seq_len: int, kernel_sizes=(7, 25, 49), gating_hidden: int = 64):
        super().__init__()
        self.trend_extractor = MultiScaleAdaptiveMovingAverageV3(
            channels=channels,
            seq_len=seq_len,
            kernel_sizes=kernel_sizes,
            gating_hidden=gating_hidden,
        )

    def forward(self, x: torch.Tensor):
        trend = self.trend_extractor(x)    # (B,L,C)
        seasonal = x - trend
        return trend, seasonal


# ======================================================
# Dynamic Channel Attention (crucial for multivariate -> target)
# ======================================================
class ChannelAttention(nn.Module):
    """
    Produce per-sample channel weights w (B, C) to aggregate multivariate info.
    """
    def __init__(self, channels: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        in_dim = 4 * channels  # mean, std, last, slope
        hidden = max(32, min(hidden, in_dim))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        return w: (B, C) softmax over channels
        """
        B, L, C = x.shape
        x_t = x.permute(0, 2, 1)  # (B,C,L)
        mean = x_t.mean(dim=-1)
        std  = x_t.std(dim=-1, unbiased=False)
        last = x_t[:, :, -1]
        slope = (x_t[:, :, -1] - x_t[:, :, 0]) / max(1, (L - 1))

        feat = torch.cat([mean, std, last, slope], dim=-1)  # (B,4C)
        logits = self.mlp(feat)                             # (B,C)
        w = torch.softmax(logits, dim=-1)
        return w


# ======================================================
# GModel v3
#   - RevIN
#   - MS adaptive decomposition
#   - channel attention aggregation -> target-only linear heads
# ======================================================
class GModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int = 1,
        kernel_sizes=(7, 25, 49),
        gating_hidden: int = 64,
        revin_affine: bool = True,
        attn_hidden: int = 64,
        dropout_head: float = 0.0,
        use_revin: bool = True,
        use_residual: bool = True,  # 新增：是否使用残差学习
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_revin = use_revin
        self.use_residual = use_residual

        self.revin = RevIN(num_features=enc_in, affine=revin_affine) if use_revin else None
        self.decomposition = SeriesDecompositionMSV3(
            channels=enc_in,
            seq_len=seq_len,
            kernel_sizes=kernel_sizes,
            gating_hidden=gating_hidden,
        )

        # dynamic channel attention to aggregate all channels into 1 target signal
        self.chan_attn = ChannelAttention(enc_in, hidden=attn_hidden, dropout=0.1)

        # linear heads (time axis): (B, L) -> (B, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        
        # 残差学习分支：直接从原始输入学习
        if use_residual:
            self.residual_branch = nn.Sequential(
                nn.Linear(seq_len, pred_len),
                nn.GELU(),
                nn.Dropout(dropout_head * 0.5) if dropout_head > 0 else nn.Identity(),
                nn.Linear(pred_len, pred_len)
            )
            # 可学习的融合权重
            self.fusion_weight = nn.Parameter(torch.tensor(0.1))  # 初始时残差贡献较小
        else:
            self.residual_branch = None
            
        self.drop = nn.Dropout(dropout_head) if dropout_head > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        return: (B, pred_len)
        """
        B, L, C = x.shape
        if L != self.seq_len:
            raise ValueError(f"seq_len mismatch: x has {L}, expected {self.seq_len}")
        if C != self.enc_in:
            raise ValueError(f"enc_in mismatch: x has {C}, expected {self.enc_in}")

        # RevIN normalize (if enabled)
        if self.use_revin:
            x_n = self.revin(x, mode="norm")
        else:
            x_n = x

        # decompose on normalized series
        trend, seasonal = self.decomposition(x_n)  # (B,L,C)

        # channel weights from normalized input (more stable)
        w = self.chan_attn(x_n)                    # (B,C)

        # aggregate multivariate components into target-only signals
        # (B,L,C) weighted sum over channels -> (B,L)
        trend_t = (trend * w.unsqueeze(1)).sum(dim=-1)
        seas_t  = (seasonal * w.unsqueeze(1)).sum(dim=-1)

        # time-linear heads
        y_decomp = self.linear_trend(trend_t) + self.linear_seasonal(seas_t)  # (B,pred_len)
        
        # 残差学习分支
        if self.use_residual and self.residual_branch is not None:
            # 从target通道的原始输入直接学习
            x_target = x_n[:, :, 0]  # (B, L) - target通道
            y_residual = self.residual_branch(x_target)  # (B, pred_len)
            
            # 自适应融合
            alpha = torch.sigmoid(self.fusion_weight)  # 0-1之间
            y_n = (1 - alpha) * y_decomp + alpha * y_residual
        else:
            y_n = y_decomp
            
        y_n = self.drop(y_n)

        # denorm back to raw space using target channel stats (channel 0) (if RevIN enabled)
        if self.use_revin:
            y = self.revin.denorm_target(y_n, target_idx=0)
        else:
            y = y_n
        return y


# ======================================================
# Channel-Independent Linear (inspired by DLinear & iTransformer)
# ======================================================
class ChannelIndependentLinear(nn.Module):
    """
    每个通道独立的线性层，避免通道间的错误交互
    """
    def __init__(self, seq_len: int, pred_len: int, num_channels: int):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_channels)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        return: (B, C, pred_len)
        """
        B, C, L = x.shape
        out = []
        for i in range(C):
            out.append(self.linears[i](x[:, i, :]))  # (B, pred_len)
        return torch.stack(out, dim=1)  # (B, C, pred_len)


# ======================================================
# Frequency Enhanced Module (inspired by FEDformer)
# ======================================================
class FrequencyEnhancement(nn.Module):
    """
    频域增强模块：捕获周期性模式
    """
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        # 可学习的频域权重
        freq_size = seq_len // 2 + 1
        self.freq_weight = nn.Parameter(torch.ones(freq_size, 2))  # real & imag
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        return: (B, C, L) enhanced signal
        """
        B, C, L = x.shape
        
        # FFT
        x_freq = torch.fft.rfft(x, dim=-1)  # (B, C, L//2+1) complex
        
        # 应用可学习权重
        weight_complex = torch.complex(
            self.freq_weight[:, 0],
            self.freq_weight[:, 1]
        )  # (L//2+1,)
        
        x_freq_enhanced = x_freq * weight_complex.unsqueeze(0).unsqueeze(0)
        
        # IFFT
        x_enhanced = torch.fft.irfft(x_freq_enhanced, n=L, dim=-1)
        
        return x_enhanced


# ======================================================
# Improved GModelSimple with SOTA techniques
# ======================================================
class GModelSimple(nn.Module):
    """
    简化版GModel，自适应不同序列长度
    - 长序列(≥10): 纯DLinear（PEMRs专用，不添加任何改进）
    - 短序列(<10): 增强版（频域+双路径集成）
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int = 1,
        kernel_size: int = 25,
        dropout_head: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.is_long_seq = (seq_len >= 10)  # 降低阈值，PEMRs的seq_len=14属于长序列
        
        if self.is_long_seq:
            # ===== 长序列：纯DLinear（不添加任何改进）=====
            # 完全使用DLinear的实现，确保性能一致
            
            k = min(kernel_size, seq_len)
            if k % 2 == 0:
                k = k - 1
            self.kernel_size = max(3, k)
            
            # 使用DLinear的SeriesDecomposition
            from model.dlinear import SeriesDecomposition
            self.decomposition = SeriesDecomposition(self.kernel_size)
            
            # 通道独立的线性层（与DLinear完全一致）
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
            
        else:
            # ===== 短序列：增强版（Aphids风格）=====
            # 输入归一化
            self.input_norm = nn.LayerNorm(enc_in)
            
            # 趋势提取（可学习移动平均）
            k = min(kernel_size, seq_len)
            if k % 2 == 0:
                k = k - 1
            self.kernel_size = max(3, k)
            self.smooth_weight = nn.Parameter(torch.ones(self.kernel_size))
            
            # 频域增强
            self.freq_enhance = FrequencyEnhancement(seq_len)
            
            # Path 1: 分解预测
            self.linear_trend = ChannelIndependentLinear(seq_len, pred_len, enc_in)
            self.linear_seasonal = ChannelIndependentLinear(seq_len, pred_len, enc_in)
            
            # Path 2: 直接预测
            self.linear_direct = ChannelIndependentLinear(seq_len, pred_len, enc_in)
            
            # 路径融合权重
            self.path_weight = nn.Parameter(torch.tensor([0.6, 0.4]))
            
            # 通道融合
            if enc_in > 1:
                self.channel_fusion = nn.Linear(enc_in, 1)
            else:
                self.channel_fusion = None
        
        # Dropout
        if dropout_head > 0:
            self.dropout = nn.Dropout(dropout_head)
        else:
            self.dropout = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        return: (B, pred_len)
        """
        B, L, C = x.shape
        
        if self.is_long_seq:
            # ===== 长序列：纯DLinear实现 =====
            
            # 使用DLinear的分解模块
            trend, seasonal = self.decomposition(x)  # (B, L, C)
            
            # 转换为 (B, C, L) 格式
            trend = trend.permute(0, 2, 1)
            seasonal = seasonal.permute(0, 2, 1)
            
            # 通道独立的线性预测（与DLinear完全一致）
            trend_out = []
            seasonal_out = []
            
            for i in range(C):
                trend_out.append(self.linear_trend[i](trend[:, i, :]))  # (B, pred_len)
                seasonal_out.append(self.linear_seasonal[i](seasonal[:, i, :]))  # (B, pred_len)
            
            trend_out = torch.stack(trend_out, dim=-1)  # (B, pred_len, C)
            seasonal_out = torch.stack(seasonal_out, dim=-1)  # (B, pred_len, C)
            
            # 合并趋势和季节
            output = trend_out + seasonal_out  # (B, pred_len, C)
            
            # 只取第一个通道（target）
            y = output[:, :, 0]  # (B, pred_len)
            
        else:
            # ===== 短序列：增强版方式 =====
            x_norm = self.input_norm(x)  # (B, L, C)
            x_t = x_norm.permute(0, 2, 1)  # (B, C, L)
            
            # 频域增强（残差连接）
            x_freq = self.freq_enhance(x_t)  # (B, C, L)
            x_enhanced = x_t + 0.1 * x_freq
            
            # 趋势-季节分解
            pad = (self.kernel_size - 1) // 2
            x_pad = F.pad(x_enhanced, (pad, pad), mode='replicate')
            
            w = torch.softmax(self.smooth_weight, dim=0).view(1, 1, -1)
            x_unfold = x_pad.unfold(dimension=-1, size=self.kernel_size, step=1)
            trend = (x_unfold * w).sum(dim=-1)  # (B, C, L)
            seasonal = x_enhanced - trend  # (B, C, L)
            
            # Path 1: 分解预测
            trend_pred = self.linear_trend(trend)  # (B, C, pred_len)
            seasonal_pred = self.linear_seasonal(seasonal)  # (B, C, pred_len)
            y_decomp = trend_pred + seasonal_pred
            
            # Path 2: 直接预测
            y_direct = self.linear_direct(x_enhanced)  # (B, C, pred_len)
            
            # 路径融合
            path_w = torch.softmax(self.path_weight, dim=0)
            y_multi = path_w[0] * y_decomp + path_w[1] * y_direct  # (B, C, pred_len)
            
            # 通道融合
            if self.channel_fusion is not None:
                y_multi = y_multi.permute(0, 2, 1)  # (B, pred_len, C)
                y = self.channel_fusion(y_multi).squeeze(-1)  # (B, pred_len)
            else:
                if C == 1:
                    y = y_multi.squeeze(1)  # (B, pred_len)
                else:
                    y = y_multi.mean(dim=1)  # (B, pred_len)
        
        if self.dropout is not None:
            y = self.dropout(y)
        
        return y


# ======================================================
# Model Configuration Profiles (数据集自适应配置)
# ======================================================
class GModelConfig:
    """
    GModel配置管理器：根据数据集特征自动选择最优配置
    
    设计理念：
    - 短序列（<10）：简单模型，小kernel，少尺度
    - 中序列（10-50）：中等复杂度，双尺度
    - 长序列（>50）：完整模型，多尺度，大kernel
    
    - 单变量：简化版本，无通道注意力
    - 少变量（<5）：中等复杂度
    - 多变量（≥5）：完整版本，通道注意力
    """
    
    @staticmethod
    def get_config(seq_len: int, input_size: int, scenario: str = "auto"):
        """
        根据序列长度和输入维度自动选择配置
        
        Args:
            seq_len: 序列长度
            input_size: 输入特征数
            scenario: 场景类型 ("simple", "medium", "complex", "auto")
        
        Returns:
            config dict
        """
        # 自动判断场景
        if scenario == "auto":
            if input_size == 1:
                scenario = "simple"
            elif seq_len < 10:
                scenario = "simple"
            elif input_size < 5 and seq_len < 50:
                scenario = "medium"
            else:
                scenario = "complex"
        
        # 配置字典
        configs = {
            "simple": {
                "use_simple": True,
                "kernel_sizes": (min(25, seq_len // 2),),
                "gating_hidden": 32,
                "attn_hidden": 32,
                "dropout_head": 0.2,
                "use_revin": False,
                "revin_affine": False,
                "use_residual": False,  # Aphids不使用残差
            },
            "medium": {
                "use_simple": False,
                "kernel_sizes": (7, min(25, seq_len // 2)),
                "gating_hidden": 48,
                "attn_hidden": 48,
                "dropout_head": 0.15,
                "use_revin": True,
                "revin_affine": True,
                "use_residual": False,  # 中等序列不使用残差
            },
            "complex": {
                "use_simple": False,
                "kernel_sizes": (7, 25, min(49, seq_len // 2)),
                "gating_hidden": 64,
                "attn_hidden": 64,
                "dropout_head": 0.1,
                "use_revin": True,
                "revin_affine": True,
                "use_residual": False,  # Weather不使用残差
            },
            "pemrs": {
                # PEMRs专用配置：使用GModelPEMRs
                "use_pemrs_model": True,  # 使用专用模型
            },
        }
        
        return configs[scenario]
    
    @staticmethod
    def get_lr(seq_len: int, input_size: int, scenario: str = "auto"):
        """推荐学习率"""
        if scenario == "auto":
            if input_size == 1:
                scenario = "simple"
            elif seq_len < 10:
                scenario = "simple"
            elif input_size < 5 and seq_len < 50:
                scenario = "medium"
            else:
                scenario = "complex"
        
        lr_map = {
            "simple": 0.0015,
            "medium": 0.001,
            "complex": 0.0008,
            "pemrs": 0.0008,  # PEMRs专用：较小的学习率，稳定训练
        }
        return lr_map.get(scenario, 0.001)


# ======================================================
# Wrapper (match your experiment interface)
# ======================================================
class GModelWrapper(nn.Module):
    """
    GModel统一接口，支持自动配置和手动配置
    
    使用方式1（自动配置）：
        model = GModelWrapper.from_auto_config(
            input_size=5, seq_len=24, pred_len=24
        )
    
    使用方式2（手动配置）：
        model = GModelWrapper(
            input_size=5, seq_len=24, pred_len=24,
            kernel_sizes=(7, 25), use_simple=False, ...
        )
    """
    
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        pred_len: int,
        kernel_sizes=(7, 25, 49),
        gating_hidden: int = 64,
        revin_affine: bool = True,
        attn_hidden: int = 64,
        dropout_head: float = 0.1,
        use_revin: bool = True,
        use_simple: bool = False,
        use_residual: bool = True,  # 新增参数
    ):
        super().__init__()
        self.use_simple = use_simple
        
        if use_simple:
            # 简化版本
            kernel_size = kernel_sizes[0] if isinstance(kernel_sizes, tuple) else kernel_sizes
            self.model = GModelSimple(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=input_size,
                kernel_size=kernel_size,
                dropout_head=dropout_head,
            )
        else:
            # 完整版本
            self.model = GModel(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=input_size,
                kernel_sizes=kernel_sizes,
                gating_hidden=gating_hidden,
                revin_affine=revin_affine,
                attn_hidden=attn_hidden,
                dropout_head=dropout_head,
                use_revin=use_revin,
                use_residual=use_residual,  # 传递残差学习参数
            )
    
    @classmethod
    def from_auto_config(
        cls,
        input_size: int,
        seq_len: int,
        pred_len: int,
        scenario: str = "auto",
    ):
        """
        根据数据特征自动配置模型
        
        Args:
            input_size: 输入特征数
            seq_len: 序列长度
            pred_len: 预测长度
            scenario: "auto", "simple", "medium", "complex", "pemrs"
        
        Returns:
            GModelWrapper instance or GModelPEMRs
        """
        config = GModelConfig.get_config(seq_len, input_size, scenario)
        
        # PEMRs专用模型
        if config.get("use_pemrs_model", False):
            return GModelPEMRs(
                seq_len=seq_len,
                pred_len=pred_len,
                enc_in=input_size,
                kernel_size=15,  # 减小kernel_size，更快响应变化
                dropout=0.05,
            )
        
        return cls(
            input_size=input_size,
            seq_len=seq_len,
            pred_len=pred_len,
            **config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ======================================================
# GModelPEMRs: 专门针对PEMRs数据集优化的模型
# 基于DLinear，添加可学习分解和Individual模式
# ======================================================
class LearnableDecomposition(nn.Module):
    """
    可学习的序列分解模块
    使用可学习的卷积核替代固定的移动平均
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        # 可学习的平滑权重
        self.weight = nn.Parameter(torch.ones(kernel_size) / kernel_size)
        
    def forward(self, x):
        """
        x: (B, L, C)
        return: trend (B, L, C), seasonal (B, L, C)
        """
        B, L, C = x.shape
        
        # 转换为 (B, C, L)
        x_t = x.permute(0, 2, 1)
        
        # 归一化权重
        w = torch.softmax(self.weight, dim=0)
        
        # 填充
        pad = (self.kernel_size - 1) // 2
        x_pad = F.pad(x_t, (pad, pad), mode='replicate')
        
        # 应用可学习的平滑
        # 使用unfold实现滑动窗口
        x_unfold = x_pad.unfold(dimension=-1, size=self.kernel_size, step=1)  # (B, C, L, K)
        trend = (x_unfold * w.view(1, 1, 1, -1)).sum(dim=-1)  # (B, C, L)
        
        # 转回 (B, L, C)
        trend = trend.permute(0, 2, 1)
        seasonal = x - trend
        
        return trend, seasonal


class GModelPEMRs(nn.Module):
    """
    专门针对PEMRs数据集优化的GModel
    
    改进点：
    1. 可学习的分解（替代固定移动平均）
    2. Individual模式（每个通道独立的线性层）
    3. 双路径融合（分解路径 + 直接路径）
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int = 1,
        kernel_size: int = 25,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        
        # 可学习的分解
        self.decomposition = LearnableDecomposition(kernel_size)
        
        # Individual模式：每个通道独立的线性层
        self.linear_trend = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(enc_in)
        ])
        self.linear_seasonal = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(enc_in)
        ])
        
        # 直接路径：从原始输入直接预测（只对target通道）
        self.linear_direct = nn.Linear(seq_len, pred_len)
        
        # 可学习的融合权重（初始化为0.8，即80%分解 + 20%直接）
        self.fusion_weight = nn.Parameter(torch.tensor(1.4))  # sigmoid(1.4) ≈ 0.8
        
    def forward(self, x):
        """
        x: (B, L, C)
        return: (B, pred_len)
        """
        B, L, C = x.shape
        
        # 可学习分解
        trend, seasonal = self.decomposition(x)  # (B, L, C)
        
        # Individual模式预测
        trend_out = []
        seasonal_out = []
        
        for i in range(C):
            trend_out.append(self.linear_trend[i](trend[:, :, i]))  # (B, pred_len)
            seasonal_out.append(self.linear_seasonal[i](seasonal[:, :, i]))  # (B, pred_len)
        
        trend_out = torch.stack(trend_out, dim=-1)  # (B, pred_len, C)
        seasonal_out = torch.stack(seasonal_out, dim=-1)  # (B, pred_len, C)
        
        # 分解路径输出
        y_decomp = trend_out[:, :, 0] + seasonal_out[:, :, 0]  # (B, pred_len)
        
        # 直接路径输出（只用target通道）
        y_direct = self.linear_direct(x[:, :, 0])  # (B, pred_len)
        
        # 融合
        alpha = torch.sigmoid(self.fusion_weight)
        y = alpha * y_decomp + (1 - alpha) * y_direct
        
        return y


__all__ = [
    "GModel", 
    "GModelWrapper", 
    "GModelSimple",
    "GModelPEMRs",
    "SeriesDecompositionMSV3", 
    "MultiScaleAdaptiveMovingAverageV3", 
    "RevIN",
    "ChannelIndependentLinear",
    "FrequencyEnhancement",
    "LearnableDecomposition"
]
