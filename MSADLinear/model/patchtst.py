"""
PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers

This module implements PatchTST model based on:
"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
by Nie et al., ICLR 2023

Key features:
- Patch-based tokenization (divide time series into patches)
- Channel independence (each variable is processed independently)
- Transformer encoder for patch-level representation
- Linear projection for prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """
    Patch Embedding: Convert time series patches into embeddings.
    """
    
    def __init__(self, d_model, patch_len, stride, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # Linear projection for patches
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = nn.Parameter(torch.randn(1, 1000, d_model))  # Max 1000 patches
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, n_vars) or (B, L) for univariate
        Returns:
            x: (B, num_patches, d_model)
            num_patches: number of patches
        """
        # Handle univariate case
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L, 1)
        
        B, L, n_vars = x.shape
        
        # Create patches using unfold
        # x: (B, L, n_vars) -> (B, n_vars, L) -> patches
        x = x.permute(0, 2, 1)  # (B, n_vars, L)
        
        # Padding
        x = self.padding_patch_layer(x)  # (B, n_vars, L + stride)
        
        # Unfold to create patches
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B, n_vars, num_patches, patch_len)
        
        # Reshape: (B, n_vars, num_patches, patch_len) -> (B * n_vars, num_patches, patch_len)
        B, n_vars, num_patches, patch_len = patches.shape
        patches = patches.permute(0, 2, 1, 3).contiguous()  # (B, num_patches, n_vars, patch_len)
        patches = patches.view(B * num_patches, n_vars, patch_len)  # (B * num_patches, n_vars, patch_len)
        
        # For channel independence, process each variable separately
        # Reshape to (B * num_patches * n_vars, patch_len)
        patches = patches.view(B * num_patches * n_vars, patch_len)  # (B * num_patches * n_vars, patch_len)
        
        # Embed patches
        patch_emb = self.value_embedding(patches)  # (B * num_patches * n_vars, d_model)
        
        # Reshape back: (B * num_patches * n_vars, d_model) -> (B, num_patches, n_vars, d_model)
        patch_emb = patch_emb.view(B, num_patches, n_vars, self.d_model)
        
        # For channel independence, we process each variable independently
        # Reshape to (B * n_vars, num_patches, d_model)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous()  # (B, n_vars, num_patches, d_model)
        patch_emb = patch_emb.view(B * n_vars, num_patches, self.d_model)  # (B * n_vars, num_patches, d_model)
        
        # Add positional embedding
        patch_emb = patch_emb + self.position_embedding[:, :num_patches, :]
        patch_emb = self.dropout(patch_emb)
        
        return patch_emb, num_patches


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            x: (B, L, d_model)
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Feed-forward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out
        
        return x


class PatchTST(nn.Module):
    """
    PatchTST: Patch-based Time Series Transformer.
    """
    
    def __init__(self, c_in, context_window, target_window, patch_len=16, stride=8,
                 d_model=128, n_heads=8, e_layers=3, d_ff=256, dropout=0.1,
                 channel_independence=True):
        """
        Args:
            c_in: Number of input variables (features)
            context_window: Input sequence length
            target_window: Prediction horizon
            patch_len: Length of each patch
            stride: Stride for patch creation
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            channel_independence: Whether to process each variable independently
        """
        super(PatchTST, self).__init__()
        self.c_in = c_in
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.channel_independence = channel_independence
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        
        # Transformer encoder
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        ])
        
        # Output projection
        # We'll use adaptive pooling or mean pooling to handle variable num_patches
        # For channel independence, we use mean pooling over patches then linear projection
        if channel_independence:
            # Use mean pooling over patches, then linear projection
            self.head = nn.Linear(d_model, target_window)
        else:
            # Not using channel independence - would need to handle differently
            self.head = nn.Linear(d_model, target_window)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, C) where L is context_window, C is c_in
        Returns:
            output: (B, target_window)
        """
        B, L, C = x.shape
        
        # Patch embedding
        x_patch, num_patches = self.patch_embedding(x)  # (B * C, num_patches, d_model) if channel_independence
        
        # Transformer encoder
        for layer in self.encoder:
            x_patch = layer(x_patch)  # (B * C, num_patches, d_model)
        
        # Process patches for each variable
        if self.channel_independence:
            # x_patch: (B * C, num_patches, d_model)
            # Use mean pooling over patches to get fixed-size representation
            x_pooled = x_patch.mean(dim=1)  # (B * C, d_model) - mean over patches
            
            # Project to target_window
            output = self.head(x_pooled)  # (B * C, target_window)
            
            # Reshape back: (B * C, target_window) -> (B, C, target_window)
            output = output.view(B, C, self.target_window)
            
            # For multivariate, we typically predict only the target variable
            # If C > 1, take the first channel (target variable)
            # If C == 1, we still need to squeeze
            if C > 1:
                output = output[:, 0, :]  # (B, target_window)
            else:
                output = output.squeeze(1)  # (B, target_window)
        else:
            # Not using channel independence
            # x_patch: (B, num_patches, d_model)
            x_pooled = x_patch.mean(dim=1)  # (B, d_model) - mean over patches
            output = self.head(x_pooled)  # (B, target_window)
        
        return output


class PatchTSTModel(nn.Module):
    """
    Wrapper for PatchTST adapted to match other models' interface.
    For single-step prediction.
    """
    
    def __init__(self, input_size, output_size=1, d_model=32, n_heads=4,
                 e_layers=2, d_ff=128, dropout=0.1, seq_len=7, patch_len=4, stride=2):
        """
        Args:
            input_size: Number of input features
            output_size: Output size (1 for single-step)
            d_model: Model dimension (reduced for small sequences)
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            seq_len: Input sequence length
            patch_len: Length of each patch (adjusted for short sequences)
            stride: Stride for patch creation
        """
        super(PatchTSTModel, self).__init__()
        
        # Adjust patch_len and stride for short sequences
        if seq_len <= 7:
            patch_len = min(patch_len, seq_len // 2) if seq_len > 2 else seq_len
            stride = min(stride, patch_len) if patch_len > 1 else 1
        
        self.patchtst = PatchTST(
            c_in=input_size,
            context_window=seq_len,
            target_window=output_size,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            channel_independence=True
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, output_size)
        """
        return self.patchtst(x)


__all__ = ["PatchTST", "PatchTSTModel", "PatchEmbedding"]

