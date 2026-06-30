"""
SOFTS (Selective Observation-based Forecasting for Time Series) - PyTorch
轻量实现，支持纯时序（无外生），核心思路：
- 局部打补丁（patch）+ 重要性评分，选取 Top-k 关键补丁做注意力，降低长序列计算。
- 多尺度卷积特征，融合局部趋势/波动。
- 残差 + 前馈，输出多步回归。

输入: (B, L, C)  B=batch, L=序列长度, C=特征维
输出: (B, pred_horizon)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchExtract(nn.Module):
    """将时间维分块成 patch，pool 成 patch 表示。"""

    def __init__(self, patch_len: int, stride: Optional[int] = None, pool: str = "mean"):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride or patch_len
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        return: (B, P, C)  P为补丁数
        """
        B, L, C = x.shape
        # unfold: (B, C, patch_len, P)
        patches = x.transpose(1, 2).unfold(dimension=2, size=self.patch_len, step=self.stride)
        # (B, C, patch_len, P) -> (B, P, patch_len, C)
        patches = patches.permute(0, 3, 2, 1)
        if self.pool == "mean":
            pooled = patches.mean(dim=2)
        elif self.pool == "max":
            pooled = patches.max(dim=2).values
        else:
            raise ValueError("Unsupported pool")
        return pooled  # (B, P, C)


class ImportanceScorer(nn.Module):
    """对 patch 做重要性评分，用于 Top-k 选择。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, d_model)
        return self.proj(x).squeeze(-1)  # (B, P)


class LocalConv(nn.Module):
    """多尺度 depthwise 卷积融合局部特征。"""

    def __init__(self, d_model: int, kernels=(3, 5, 7), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model),
                    nn.Conv1d(d_model, d_model, 1),
                    nn.GELU(),
                )
                for k in kernels
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model * len(kernels), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, d_model)
        outs = []
        for branch in self.branches:
            y = branch(x.transpose(1, 2)).transpose(1, 2)  # (B, P, d_model)
            outs.append(y)
        h = torch.cat(outs, dim=-1)
        h = self.proj(h)
        return self.dropout(h)


class SelectiveAttention(nn.Module):
    """仅在 Top-k patch 上做注意力，减少计算。"""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, topk: int = 8):
        super().__init__()
        self.topk = topk
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        x: (B, P, d_model)
        scores: (B, P)
        """
        B, P, _ = x.shape
        k = min(self.topk, P)
        # 选 Top-k
        idx = scores.topk(k, dim=1).indices  # (B, k)
        # gather
        batch_indices = torch.arange(B, device=x.device).unsqueeze(-1)
        selected = x[batch_indices, idx]  # (B, k, d_model)
        # self-attn on selected
        h, _ = self.attn(selected, selected, selected)
        h = self.dropout(h)
        # scatter back (简化版：仅返回更新后的 selected，其他保持原样)
        out = x.clone()
        out[batch_indices, idx] = self.norm(out[batch_indices, idx] + h)
        return out


class SOFTSBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, topk: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = ImportanceScorer(d_model)
        self.local = LocalConv(d_model, dropout=dropout)
        self.attn = SelectiveAttention(d_model, n_heads=n_heads, dropout=dropout, topk=topk)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, d_model)
        scores = self.scorer(x)
        h_local = self.local(x)
        x = x + h_local
        x = self.attn(x, scores)
        x = x + self.ff(self.norm(x))
        return x


class SOFTS(nn.Module):
    """
    纯时序版 SOFTS，默认输出 pred_horizon 步的回归结果。
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        topk: int = 8,
        patch_len: int = 8,
        patch_stride: Optional[int] = None,
        dropout: float = 0.1,
        pred_horizon: int = 1,
        pooling: str = "mean",  # 对 patch 结果的汇聚
    ):
        super().__init__()
        self.patch = PatchExtract(patch_len, stride=patch_stride, pool="mean")
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [SOFTSBlock(d_model, n_heads, topk, dropout) for _ in range(n_layers)]
        )
        self.pooling = pooling
        self.readout = nn.Linear(d_model, pred_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, input_dim)
        return: (B, pred_horizon)
        """
        # patching
        x = self.patch(x)  # (B, P, C)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)

        if self.pooling == "last":
            h = x[:, -1, :]
        else:
            h = x.mean(dim=1)
        return self.readout(h)


__all__ = ["SOFTS", "SOFTSBlock"]

