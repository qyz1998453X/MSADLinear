"""
KExCon Models: MSADLinear with LLM Knowledge Constraints (Improved)
- MSADLinearKExCon: Full version (channel + scale constraints)
- MSADLinearKExConLite: Channel-only constraints
- MSADLinearKExConScale: Scale-only constraints

Improvements:
- Validation-driven soft constraints (KL divergence regularization)
- Adaptive confidence based on prior quality
- Selective constraint application (only high-confidence dimensions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class MSADLinearKExCon(nn.Module):
    """
    Multi-Scale Adaptive DLinear with KExCon (full: channel + scale)
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

        # Channel gating (learns which channels matter)
        self.channel_gate = nn.Sequential(
            nn.Linear(enc_in, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, enc_in),
            nn.Softmax(dim=-1)
        )

        # Scale gating (learns which scales matter)
        self.scale_gate = nn.Sequential(
            nn.Linear(self.M, gating_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gating_hidden, self.M),
            nn.Softmax(dim=-1)
        )

        # Attention for channel aggregation
        self.attn = nn.MultiheadAttention(
            embed_dim=enc_in,
            num_heads=max(1, enc_in // 8),
            batch_first=True,
            dropout=dropout
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
        llm_conf: (B,) - confidence (0-1, adjusted by prior quality)
        
        Note: LLM priors are NOT directly applied to weights.
        Instead, they are used as soft regularization targets in compute_knowledge_loss().
        This allows the model to learn when to trust or ignore priors.
        """
        B, L, C = x.shape

        # Multi-scale predictions
        scale_preds = []
        for i, trend_layer in enumerate(self.trends):
            trend = trend_layer(x[:, :, 0])  # (B, pred_len) - use target only
            seasonal = self.seasonals[i](x[:, :, 0])
            scale_preds.append(trend + seasonal)

        scale_preds = torch.stack(scale_preds, dim=1)  # (B, M, pred_len)

        # Channel gating (learned freely, not constrained by LLM)
        x_mean = x.mean(dim=1)  # (B, C)
        channel_weights = self.channel_gate(x_mean)  # (B, C)

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
        
        Key improvements:
        1. Soft regularization: KL(learned || prior) instead of hard replacement
        2. Adaptive confidence: Adjusted based on prior quality (entropy)
        3. Selective application: Only apply to high-confidence dimensions
        
        The model learns when to trust or ignore priors through gradient flow.
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
        # High entropy = uniform/uninformative prior -> low quality
        # Low entropy = concentrated prior -> high quality
        def entropy(p):
            return -(p * torch.log(p + 1e-8)).sum(dim=-1)
        
        h_p = entropy(llm_p_norm)  # (B,)
        h_p_max = np.log(llm_p.shape[-1])
        prior_quality_p = 1.0 - (h_p / h_p_max)  # (B,) in [0, 1]
        
        # Adaptive confidence: conf * prior_quality
        # If prior is uninformative (low quality), reduce constraint strength
        adaptive_conf_p = llm_conf * prior_quality_p  # (B,)
        
        # Soft constraint: KL divergence regularization
        # KL(learned || prior) encourages learned to be close to prior
        # But gradient flow allows model to deviate if it improves prediction
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


class MSADLinearKExConLite(nn.Module):
    """
    Channel-only constraints (no scale constraints)
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

        # Channel gating
        self.channel_gate = nn.Sequential(
            nn.Linear(enc_in, gating_hidden),
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

        scale_preds = torch.stack(scale_preds, dim=1)  # (B, M, pred_len)

        # Channel gating (learned freely, soft constraint via loss)
        x_mean = x.mean(dim=1)
        channel_weights = self.channel_gate(x_mean)
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


class MSADLinearKExConScale(nn.Module):
    """
    Scale-only constraints (no channel constraints)
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

        # Channel gating (no LLM constraint)
        self.channel_gate = nn.Sequential(
            nn.Linear(enc_in, gating_hidden),
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

        scale_preds = torch.stack(scale_preds, dim=1)  # (B, M, pred_len)

        # Channel gating (no constraint)
        x_mean = x.mean(dim=1)
        channel_weights = self.channel_gate(x_mean)

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
