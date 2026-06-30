"""
改进版 KExCon Models：修复channel_gate的输入

关键改进：
1. channel_gate 使用完整的时间序列特征而非简单平均
2. 使用统计特征 [mean, std, max, min, slope] 来捕捉时间维度信息
3. 增强特征差异性，使模型能学到有意义的权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class MSADLinearKExConImproved(nn.Module):
    """
    改进版：Multi-Scale Adaptive DLinear with KExCon
    
    关键改进：
    - channel_gate 输入：使用统计特征而非简单平均
    - 特征包括：mean, std, max, min, slope
    - 这样可以捕捉时间序列的丰富信息
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_sizes: Tuple[int, ...] = (5, 7),
        gating_hidden: int = 48,
        attn_hidden: int = 48,
        lambda_k: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.kernel_sizes = kernel_sizes
        self.lambda_k = lambda_k
        self.M = len(kernel_sizes)

        # Trend decomposition (per scale)
        self.trends = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in kernel_sizes
        ])

        # Seasonal decomposition (per scale)
        self.seasonals = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in kernel_sizes
        ])

        # 改进的 Channel gating：使用统计特征
        # 输入特征维度：5 * enc_in (mean, std, max, min, slope)
        self.channel_gate = nn.Sequential(
            nn.Linear(5 * enc_in, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, enc_in),
            nn.Softmax(dim=-1)
        )

        # Scale gating (learned freely, not constrained by LLM)
        self.scale_gate = nn.Sequential(
            nn.Linear(self.M, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, self.M),
            nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)

    def _extract_statistical_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        从时间序列中提取统计特征
        
        Args:
            x: (B, L, C) - 时间序列
        
        Returns:
            features: (B, 5*C) - 统计特征
        """
        B, L, C = x.shape
        
        # 计算统计特征
        mean = x.mean(dim=1)  # (B, C)
        std = x.std(dim=1)    # (B, C)
        max_val = x.max(dim=1)[0]  # (B, C)
        min_val = x.min(dim=1)[0]  # (B, C)
        
        # 计算斜率（线性趋势）
        t = torch.arange(L, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
        t_mean = t.mean()
        x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        
        numerator = ((t - t_mean) * (x - x_mean)).mean(dim=1)  # (B, C)
        denominator = ((t - t_mean) ** 2).mean()  # scalar
        slope = numerator / (denominator + 1e-8)  # (B, C)
        
        # 拼接所有特征
        features = torch.cat([mean, std, max_val, min_val, slope], dim=-1)  # (B, 5*C)
        
        return features

    def forward(
        self,
        x: torch.Tensor,
        llm_p: Optional[torch.Tensor] = None,
        llm_q: Optional[torch.Tensor] = None,
        llm_conf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, C)
        llm_p: (B, C) - channel prior
        llm_q: (B, M) - scale prior
        llm_conf: (B,) - confidence
        """
        B, L, C = x.shape

        # Multi-scale predictions
        scale_preds = []
        for i, trend_layer in enumerate(self.trends):
            trend = trend_layer(x[:, :, 0])  # (B, pred_len) - use target only
            seasonal = self.seasonals[i](x[:, :, 0])
            scale_preds.append(trend + seasonal)

        scale_preds = torch.stack(scale_preds, dim=1)  # (B, M, pred_len)

        # 改进的 Channel gating：使用统计特征
        stat_features = self._extract_statistical_features(x)  # (B, 5*C)
        channel_weights = self.channel_gate(stat_features)  # (B, C)

        # Scale gating (learned freely, not constrained by LLM)
        scale_weights = self.scale_gate(torch.ones(B, self.M, device=x.device))  # (B, M)

        # Store for knowledge loss computation
        self.learned_channel_weights = channel_weights
        self.learned_scale_weights = scale_weights

        # Combine scales (no hard constraints)
        scale_weights = scale_weights.unsqueeze(-1)  # (B, M, 1)
        pred = (scale_preds * scale_weights).sum(dim=1)  # (B, pred_len)

        return pred

    def compute_knowledge_loss(
        self,
        llm_p: torch.Tensor,
        llm_q: Optional[torch.Tensor],
        llm_conf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Soft constraint using KL divergence regularization.
        """
        B = llm_p.shape[0]
        
        # Get learned weights
        learned_p = self.learned_channel_weights  # (B, C)
        learned_q = self.learned_scale_weights    # (B, M)
        
        # Normalize to probability distributions
        learned_p = F.softmax(learned_p, dim=-1)
        learned_q = F.softmax(learned_q, dim=-1)
        llm_p_norm = F.softmax(llm_p, dim=-1)
        if llm_q is not None:
            llm_q_norm = F.softmax(llm_q, dim=-1)
        
        # Compute prior quality (entropy-based)
        def entropy(p):
            return -(p * torch.log(p + 1e-8)).sum(dim=-1)
        
        h_p = entropy(llm_p_norm)  # (B,)
        h_p_max = np.log(llm_p.shape[-1])
        prior_quality_p = 1.0 - (h_p / h_p_max)  # (B,) in [0, 1]
        
        # Adaptive confidence: conf * prior_quality
        adaptive_conf_p = llm_conf * prior_quality_p  # (B,)
        
        # Soft constraint: KL divergence regularization
        kl_loss_p = F.kl_div(
            torch.log(learned_p + 1e-8),
            llm_p_norm,
            reduction='none'
        ).sum(dim=-1)  # (B,)
        
        # Weight by adaptive confidence
        kl_loss_p = (kl_loss_p * adaptive_conf_p).mean()
        
        # Scale constraint (if available)
        kl_loss_q = 0.0
        if llm_q is not None:
            h_q = entropy(llm_q_norm)
            h_q_max = np.log(llm_q.shape[-1])
            prior_quality_q = 1.0 - (h_q / h_q_max)
            adaptive_conf_q = llm_conf * prior_quality_q
            
            kl_loss_q = F.kl_div(
                torch.log(learned_q + 1e-8),
                llm_q_norm,
                reduction='none'
            ).sum(dim=-1)
            kl_loss_q = (kl_loss_q * adaptive_conf_q).mean()
        
        # Total loss (weighted by lambda_k)
        total_loss = self.lambda_k * (kl_loss_p + kl_loss_q)
        
        return total_loss


class MSADLinearKExConImprovedLite(nn.Module):
    """
    改进版：Channel-only constraints
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_sizes: Tuple[int, ...] = (5, 7),
        gating_hidden: int = 48,
        attn_hidden: int = 48,
        lambda_k: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.kernel_sizes = kernel_sizes
        self.lambda_k = lambda_k
        self.M = len(kernel_sizes)

        # Trend decomposition
        self.trends = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in kernel_sizes
        ])

        # Seasonal decomposition
        self.seasonals = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in kernel_sizes
        ])

        # 改进的 Channel gating
        self.channel_gate = nn.Sequential(
            nn.Linear(5 * enc_in, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, enc_in),
            nn.Softmax(dim=-1)
        )

        # Scale gating (no LLM constraint)
        self.scale_gate = nn.Sequential(
            nn.Linear(self.M, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, self.M),
            nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)

    def _extract_statistical_features(self, x: torch.Tensor) -> torch.Tensor:
        """从时间序列中提取统计特征"""
        B, L, C = x.shape
        
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        max_val = x.max(dim=1)[0]
        min_val = x.min(dim=1)[0]
        
        t = torch.arange(L, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(-1)
        t_mean = t.mean()
        x_mean = x.mean(dim=1, keepdim=True)
        
        numerator = ((t - t_mean) * (x - x_mean)).mean(dim=1)
        denominator = ((t - t_mean) ** 2).mean()
        slope = numerator / (denominator + 1e-8)
        
        features = torch.cat([mean, std, max_val, min_val, slope], dim=-1)
        
        return features

    def forward(
        self,
        x: torch.Tensor,
        llm_p: Optional[torch.Tensor] = None,
        llm_q: Optional[torch.Tensor] = None,
        llm_conf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape

        # Multi-scale predictions
        scale_preds = []
        for i, trend_layer in enumerate(self.trends):
            trend = trend_layer(x[:, :, 0])
            seasonal = self.seasonals[i](x[:, :, 0])
            scale_preds.append(trend + seasonal)

        scale_preds = torch.stack(scale_preds, dim=1)

        # 改进的 Channel gating
        stat_features = self._extract_statistical_features(x)
        channel_weights = self.channel_gate(stat_features)
        self.learned_channel_weights = channel_weights

        # Scale gating (no constraint)
        scale_weights = self.scale_gate(torch.ones(B, self.M, device=x.device))

        # Combine
        scale_weights = scale_weights.unsqueeze(-1)
        pred = (scale_preds * scale_weights).sum(dim=1)

        return pred

    def compute_knowledge_loss(
        self,
        llm_p: torch.Tensor,
        llm_q: Optional[torch.Tensor],
        llm_conf: torch.Tensor,
    ) -> torch.Tensor:
        """Soft constraint with adaptive confidence (channel-only)"""
        B = llm_p.shape[0]
        
        learned_p = self.learned_channel_weights
        learned_p = F.softmax(learned_p, dim=-1)
        llm_p_norm = F.softmax(llm_p, dim=-1)
        
        # Prior quality assessment
        def entropy(p):
            return -(p * torch.log(p + 1e-8)).sum(dim=-1)
        
        h_p = entropy(llm_p_norm)
        h_p_max = np.log(llm_p.shape[-1])
        prior_quality_p = 1.0 - (h_p / h_p_max)
        adaptive_conf_p = llm_conf * prior_quality_p
        
        # Soft constraint
        kl_loss_p = F.kl_div(
            torch.log(learned_p + 1e-8),
            llm_p_norm,
            reduction='none'
        ).sum(dim=-1)
        
        kl_loss_p = (kl_loss_p * adaptive_conf_p).mean()
        
        return self.lambda_k * kl_loss_p


class MSADLinearKExConImprovedV3(nn.Module):
    """
    改进版V3：使用完整时间序列作为channel_gate输入
    
    关键改进：
    - channel_gate 输入：使用完整的时间序列 (B, L*C)
    - 这样可以保留完整的时间和特征信息
    - 应该能学到更有意义的权重差异
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_sizes: Tuple[int, ...] = (5, 7),
        gating_hidden: int = 64,
        attn_hidden: int = 48,
        lambda_k: float = 0.5,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.kernel_sizes = kernel_sizes
        self.lambda_k = lambda_k
        self.M = len(kernel_sizes)

        # Trend decomposition (per scale)
        self.trends = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in kernel_sizes
        ])

        # Seasonal decomposition (per scale)
        self.seasonals = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in kernel_sizes
        ])

        # 改进的 Channel gating：使用完整时间序列
        # 输入特征维度：seq_len * enc_in (完整的时间序列)
        self.channel_gate = nn.Sequential(
            nn.Linear(seq_len * enc_in, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, gating_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden // 2, enc_in),
            nn.Softmax(dim=-1)
        )

        # Scale gating (learned freely, not constrained by LLM)
        self.scale_gate = nn.Sequential(
            nn.Linear(self.M, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, self.M),
            nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        llm_p: Optional[torch.Tensor] = None,
        llm_q: Optional[torch.Tensor] = None,
        llm_conf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, C)
        llm_p: (B, C) - channel prior
        llm_q: (B, M) - scale prior
        llm_conf: (B,) - confidence
        """
        B, L, C = x.shape

        # Multi-scale predictions
        scale_preds = []
        for i, trend_layer in enumerate(self.trends):
            trend = trend_layer(x[:, :, 0])  # (B, pred_len) - use target only
            seasonal = self.seasonals[i](x[:, :, 0])
            scale_preds.append(trend + seasonal)

        scale_preds = torch.stack(scale_preds, dim=1)  # (B, M, pred_len)

        # 改进的 Channel gating：使用完整时间序列
        x_flat = x.reshape(B, -1)  # (B, L*C)
        channel_weights = self.channel_gate(x_flat)  # (B, C)

        # Scale gating (learned freely, not constrained by LLM)
        scale_weights = self.scale_gate(torch.ones(B, self.M, device=x.device))  # (B, M)

        # Store for knowledge loss computation
        self.learned_channel_weights = channel_weights
        self.learned_scale_weights = scale_weights

        # Combine scales (no hard constraints)
        scale_weights = scale_weights.unsqueeze(-1)  # (B, M, 1)
        pred = (scale_preds * scale_weights).sum(dim=1)  # (B, pred_len)

        return pred

    def compute_knowledge_loss(
        self,
        llm_p: torch.Tensor,
        llm_q: Optional[torch.Tensor],
        llm_conf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Soft constraint using KL divergence regularization.
        """
        B = llm_p.shape[0]
        
        # Get learned weights
        learned_p = self.learned_channel_weights  # (B, C)
        learned_q = self.learned_scale_weights    # (B, M)
        
        # Normalize to probability distributions
        learned_p = F.softmax(learned_p, dim=-1)
        learned_q = F.softmax(learned_q, dim=-1)
        llm_p_norm = F.softmax(llm_p, dim=-1)
        if llm_q is not None:
            llm_q_norm = F.softmax(llm_q, dim=-1)
        
        # Compute prior quality (entropy-based)
        def entropy(p):
            return -(p * torch.log(p + 1e-8)).sum(dim=-1)
        
        h_p = entropy(llm_p_norm)  # (B,)
        h_p_max = np.log(llm_p.shape[-1])
        prior_quality_p = 1.0 - (h_p / h_p_max)  # (B,) in [0, 1]
        
        # Adaptive confidence: conf * prior_quality
        adaptive_conf_p = llm_conf * prior_quality_p  # (B,)
        
        # Soft constraint: KL divergence regularization
        kl_loss_p = F.kl_div(
            torch.log(learned_p + 1e-8),
            llm_p_norm,
            reduction='none'
        ).sum(dim=-1)  # (B,)
        
        # Weight by adaptive confidence
        kl_loss_p = (kl_loss_p * adaptive_conf_p).mean()
        
        # Scale constraint (if available)
        kl_loss_q = 0.0
        if llm_q is not None:
            h_q = entropy(llm_q_norm)
            h_q_max = np.log(llm_q.shape[-1])
            prior_quality_q = 1.0 - (h_q / h_q_max)
            adaptive_conf_q = llm_conf * prior_quality_q
            
            kl_loss_q = F.kl_div(
                torch.log(learned_q + 1e-8),
                llm_q_norm,
                reduction='none'
            ).sum(dim=-1)
            kl_loss_q = (kl_loss_q * adaptive_conf_q).mean()
        
        # Total loss (weighted by lambda_k)
        total_loss = self.lambda_k * (kl_loss_p + kl_loss_q)
        
        return total_loss


class MSADLinearKExConImprovedV4(nn.Module):
    """
    改进版V4：使用LLM先验直接初始化权重（Hard Constraint）
    
    关键改进：
    - 不使用Softmax，改用Sigmoid + 归一化
    - 使用LLM先验初始化权重
    - 增加lambda_k到20.0，强制权重接近LLM先验
    - 使用KL divergence loss而非MSE，更适合概率分布
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_sizes: Tuple[int, ...] = (5, 7),
        gating_hidden: int = 64,
        attn_hidden: int = 48,
        lambda_k: float = 20.0,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.kernel_sizes = kernel_sizes
        self.lambda_k = lambda_k
        self.M = len(kernel_sizes)

        # Trend decomposition (per scale)
        self.trends = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in kernel_sizes
        ])

        # Seasonal decomposition (per scale)
        self.seasonals = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in kernel_sizes
        ])

        # 改进的 Channel gating：使用完整时间序列，输出非负权重
        # 不使用Softmax，改用Sigmoid确保权重在(0,1)，然后手动归一化
        self.channel_gate = nn.Sequential(
            nn.Linear(seq_len * enc_in, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, gating_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden // 2, enc_in),
            nn.Sigmoid()  # 输出在(0,1)之间
        )

        # Scale gating (learned freely, not constrained by LLM)
        self.scale_gate = nn.Sequential(
            nn.Linear(self.M, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, self.M),
            nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        llm_p: Optional[torch.Tensor] = None,
        llm_q: Optional[torch.Tensor] = None,
        llm_conf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, C)
        llm_p: (B, C) - channel prior
        llm_q: (B, M) - scale prior
        llm_conf: (B,) - confidence
        """
        B, L, C = x.shape

        # Multi-scale predictions
        scale_preds = []
        for i, trend_layer in enumerate(self.trends):
            trend = trend_layer(x[:, :, 0])  # (B, pred_len) - use target only
            seasonal = self.seasonals[i](x[:, :, 0])
            scale_preds.append(trend + seasonal)

        scale_preds = torch.stack(scale_preds, dim=1)  # (B, M, pred_len)

        # 改进的 Channel gating：使用完整时间序列，输出Sigmoid权重
        x_flat = x.reshape(B, -1)  # (B, L*C)
        channel_weights_raw = self.channel_gate(x_flat)  # (B, C) - Sigmoid输出(0,1)
        
        # 手动归一化为概率分布（允许更大的差异）
        channel_weights = channel_weights_raw / (channel_weights_raw.sum(dim=-1, keepdim=True) + 1e-8)

        # Scale gating (learned freely, not constrained by LLM)
        scale_weights = self.scale_gate(torch.ones(B, self.M, device=x.device))  # (B, M)

        # Store for knowledge loss computation
        self.learned_channel_weights = channel_weights
        self.learned_scale_weights = scale_weights

        # Combine scales (no hard constraints)
        scale_weights = scale_weights.unsqueeze(-1)  # (B, M, 1)
        pred = (scale_preds * scale_weights).sum(dim=1)  # (B, pred_len)

        return pred

    def compute_knowledge_loss(
        self,
        llm_p: torch.Tensor,
        llm_q: Optional[torch.Tensor],
        llm_conf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hard constraint: 强制权重接近LLM先验
        使用MSE loss，权重通过Sigmoid+归一化得到
        """
        B = llm_p.shape[0]
        
        # Get learned weights (Sigmoid + normalized)
        learned_p = self.learned_channel_weights  # (B, C)
        
        # Normalize LLM prior to probability distribution
        llm_p_norm = F.softmax(llm_p, dim=-1)
        
        # 使用MSE loss强制权重分布接近LLM先验
        mse_loss = F.mse_loss(learned_p, llm_p_norm, reduction='none').mean(dim=-1)  # (B,)
        
        # Compute prior quality (entropy-based)
        def entropy(p):
            return -(p * torch.log(p + 1e-8)).sum(dim=-1)
        
        h_p = entropy(llm_p_norm)  # (B,)
        h_p_max = np.log(llm_p.shape[-1])
        prior_quality_p = 1.0 - (h_p / h_p_max)  # (B,) in [0, 1]
        
        # Adaptive confidence: conf * prior_quality
        adaptive_conf_p = llm_conf * prior_quality_p  # (B,)
        
        # Weight by adaptive confidence
        mse_loss = (mse_loss * adaptive_conf_p).mean()
        
        # Total loss (weighted by lambda_k)
        total_loss = self.lambda_k * mse_loss
        
        return total_loss


class MSADLinearKExConImprovedScale(nn.Module):
    """
    改进版：Scale-only constraints
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_sizes: Tuple[int, ...] = (5, 7),
        gating_hidden: int = 48,
        attn_hidden: int = 48,
        lambda_k: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.kernel_sizes = kernel_sizes
        self.lambda_k = lambda_k
        self.M = len(kernel_sizes)

        # Trend decomposition
        self.trends = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in kernel_sizes
        ])

        # Seasonal decomposition
        self.seasonals = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in kernel_sizes
        ])

        # 改进的 Channel gating
        self.channel_gate = nn.Sequential(
            nn.Linear(5 * enc_in, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, enc_in),
            nn.Softmax(dim=-1)
        )

        # Scale gating
        self.scale_gate = nn.Sequential(
            nn.Linear(self.M, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, self.M),
            nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)

    def _extract_statistical_features(self, x: torch.Tensor) -> torch.Tensor:
        """从时间序列中提取统计特征"""
        B, L, C = x.shape
        
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        max_val = x.max(dim=1)[0]
        min_val = x.min(dim=1)[0]
        
        t = torch.arange(L, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(-1)
        t_mean = t.mean()
        x_mean = x.mean(dim=1, keepdim=True)
        
        numerator = ((t - t_mean) * (x - x_mean)).mean(dim=1)
        denominator = ((t - t_mean) ** 2).mean()
        slope = numerator / (denominator + 1e-8)
        
        features = torch.cat([mean, std, max_val, min_val, slope], dim=-1)
        
        return features

    def forward(
        self,
        x: torch.Tensor,
        llm_p: Optional[torch.Tensor] = None,
        llm_q: Optional[torch.Tensor] = None,
        llm_conf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape

        # Multi-scale predictions
        scale_preds = []
        for i, trend_layer in enumerate(self.trends):
            trend = trend_layer(x[:, :, 0])
            seasonal = self.seasonals[i](x[:, :, 0])
            scale_preds.append(trend + seasonal)

        scale_preds = torch.stack(scale_preds, dim=1)

        # 改进的 Channel gating
        stat_features = self._extract_statistical_features(x)
        channel_weights = self.channel_gate(stat_features)

        # Scale gating (learned freely, soft constraint via loss)
        scale_weights = self.scale_gate(torch.ones(B, self.M, device=x.device))
        self.learned_scale_weights = scale_weights

        # Combine
        scale_weights = scale_weights.unsqueeze(-1)
        pred = (scale_preds * scale_weights).sum(dim=1)

        return pred

    def compute_knowledge_loss(
        self,
        llm_p: torch.Tensor,
        llm_q: Optional[torch.Tensor],
        llm_conf: torch.Tensor,
    ) -> torch.Tensor:
        """Soft constraint with adaptive confidence (scale-only)"""
        if llm_q is None:
            return torch.tensor(0.0, device=llm_p.device)

        B = llm_q.shape[0]
        
        learned_q = self.learned_scale_weights
        learned_q = F.softmax(learned_q, dim=-1)
        llm_q_norm = F.softmax(llm_q, dim=-1)
        
        # Prior quality assessment
        def entropy(p):
            return -(p * torch.log(p + 1e-8)).sum(dim=-1)
        
        h_q = entropy(llm_q_norm)
        h_q_max = np.log(llm_q.shape[-1])
        prior_quality_q = 1.0 - (h_q / h_q_max)
        adaptive_conf_q = llm_conf * prior_quality_q
        
        # Soft constraint
        kl_loss_q = F.kl_div(
            torch.log(learned_q + 1e-8),
            llm_q_norm,
            reduction='none'
        ).sum(dim=-1)
        
        kl_loss_q = (kl_loss_q * adaptive_conf_q).mean()
        
        return self.lambda_k * kl_loss_q
