"""
改进版V5：直接使用LLM先验初始化，可学习的混合系数

关键改进：
- 权重 = α * LLM_prior + (1-α) * learned_weights
- α是可学习的混合系数，初始化为1.0（完全相信LLM先验）
- 逐步学习调整权重，但始终保持与LLM先验的联系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class MSADLinearKExConImprovedV5(nn.Module):
    """
    改进版V5：LLM先验混合模型
    
    关键改进：
    - 使用LLM先验作为基础权重
    - 学习一个混合系数α，控制LLM先验和学习权重的比例
    - 权重 = α * LLM_prior + (1-α) * learned_weights
    """
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        kernel_sizes: Tuple[int, ...] = (5, 7),
        gating_hidden: int = 64,
        attn_hidden: int = 48,
        lambda_k: float = 1.0,
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

        # Channel gating：学习权重
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

        # 混合系数：控制LLM先验和学习权重的比例
        # 初始化为1.0（完全相信LLM先验）
        self.alpha = nn.Parameter(torch.ones(1) * 0.8)  # 初始化为0.8

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
            trend = trend_layer(x[:, :, 0])
            seasonal = self.seasonals[i](x[:, :, 0])
            scale_preds.append(trend + seasonal)

        scale_preds = torch.stack(scale_preds, dim=1)  # (B, M, pred_len)

        # 学习的权重
        x_flat = x.reshape(B, -1)  # (B, L*C)
        learned_weights = self.channel_gate(x_flat)  # (B, C)

        # 混合LLM先验和学习权重
        if llm_p is not None:
            llm_p_norm = F.softmax(llm_p, dim=-1)  # (B, C)
            # 混合：α * LLM_prior + (1-α) * learned_weights
            alpha_clipped = torch.sigmoid(self.alpha)  # 确保α在(0,1)之间
            channel_weights = alpha_clipped * llm_p_norm + (1 - alpha_clipped) * learned_weights
        else:
            channel_weights = learned_weights

        # Scale gating (learned freely, not constrained by LLM)
        scale_weights = self.scale_gate(torch.ones(B, self.M, device=x.device))  # (B, M)

        # Store for knowledge loss computation
        self.learned_channel_weights = channel_weights
        self.learned_scale_weights = scale_weights
        self.llm_p_norm = llm_p_norm if llm_p is not None else None

        # Combine scales
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
        知识对齐loss：鼓励权重接近LLM先验
        """
        B = llm_p.shape[0]
        
        # Get learned weights
        learned_p = self.learned_channel_weights  # (B, C)
        llm_p_norm = self.llm_p_norm  # (B, C)
        
        # 使用KL divergence衡量权重与LLM先验的距离
        kl_loss = F.kl_div(
            torch.log(learned_p + 1e-8),
            llm_p_norm,
            reduction='none'
        ).sum(dim=-1)  # (B,)
        
        # Compute prior quality (entropy-based)
        def entropy(p):
            return -(p * torch.log(p + 1e-8)).sum(dim=-1)
        
        h_p = entropy(llm_p_norm)  # (B,)
        h_p_max = np.log(llm_p.shape[-1])
        prior_quality_p = 1.0 - (h_p / h_p_max)  # (B,) in [0, 1]
        
        # Adaptive confidence: conf * prior_quality
        adaptive_conf_p = llm_conf * prior_quality_p  # (B,)
        
        # Weight by adaptive confidence
        kl_loss = (kl_loss * adaptive_conf_p).mean()
        
        # Total loss (weighted by lambda_k)
        total_loss = self.lambda_k * kl_loss
        
        return total_loss
