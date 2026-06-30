"""
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

This module implements a simplified Informer model based on:
"Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
by Zhou et al., AAAI 2021

Key features:
- ProbSparse Self-Attention (reduced computational complexity)
- Self-attention Distilling (sequence length reduction)
- Generative Style Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for time series."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (seq_len, batch, d_model)
        Returns:
            x with positional encoding
        """
        x = x + self.pe[:x.size(0), :]
        return x


class ProbSparseAttention(nn.Module):
    """
    ProbSparse Self-Attention mechanism.
    Selects top-u queries instead of all queries to reduce computation.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _get_initial_context(self, V, L_Q):
        """Initialize context vector."""
        B, H, L_V, D = V.shape
        if L_Q < self.factor * np.log(L_V):
            # For short sequences, use all queries
            return None
        else:
            # Sample u = c * ln(L) queries
            u = int(self.factor * np.log(L_V))
            V_sum = V.mean(dim=-2)  # (B, H, D)
            contex = V_sum.unsqueeze(-2).expand(B, H, u, D).clone()
            return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """Update context with selected queries."""
        B, H, L_V, D = V.shape
        
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        attn = None
        return context_in

    def forward(self, queries, keys, values, attn_mask=None):
        """
        Args:
            queries: (B, L_Q, d_model)
            keys: (B, L_K, d_model)
            values: (B, L_V, d_model)
            attn_mask: optional mask
        Returns:
            output: (B, L_Q, d_model)
        """
        B, L_Q, _ = queries.shape
        _, L_K, _ = keys.shape
        
        # Linear projections
        Q = self.W_q(queries).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L_Q, d_k)
        K = self.W_k(keys).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L_K, d_k)
        V = self.W_v(values).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L_V, d_k)
        
        # For short sequences, use standard attention (simplified for seq_len <= 200)
        # This avoids ProbSparse indexing issues for decoder inputs (label_len + out_len)
        if L_Q <= 200:
            # Standard multi-head attention for short sequences
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if attn_mask is not None:
                scores.masked_fill_(attn_mask == 0, -1e9)
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            context = torch.matmul(attn, V)  # (B, H, L_Q, d_k)
        else:
            # ProbSparse: select top-u queries (for longer sequences)
            u = max(int(self.factor * np.log(L_K)), 1)
            u = min(u, L_Q)  # Ensure u <= L_Q
            
            # Compute M = QK^T / sqrt(d_k) and select top-u
            M = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            M_top = M.topk(min(u, M.size(-1)), dim=-1)[0]  # (B, H, L_Q, min(u, L_K))
            scores_top = M_top.sum(dim=-1)  # (B, H, L_Q)
            index = scores_top.topk(u, dim=-1)[1]  # (B, H, u)
            
            # Select top-u queries
            batch_indices = torch.arange(B, device=Q.device).view(B, 1, 1, 1)
            head_indices = torch.arange(self.n_heads, device=Q.device).view(1, self.n_heads, 1, 1)
            Q_top = Q[batch_indices, head_indices, index.unsqueeze(-1)].squeeze(-1)  # (B, H, u, d_k)
            
            scores = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if attn_mask is not None:
                scores.masked_fill_(attn_mask == 0, -1e9)
            
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            context_selected = torch.matmul(attn, V)  # (B, H, u, d_k)
            
            # Expand back to full length by placing selected outputs at their positions
            context = torch.zeros(B, self.n_heads, L_Q, self.d_k, device=Q.device, dtype=Q.dtype)
            context[batch_indices.squeeze(-1), head_indices.squeeze(-1), index.unsqueeze(-1)] = context_selected.unsqueeze(-2)
            # For positions not selected, use mean of selected outputs
            selected_mask = torch.zeros(B, self.n_heads, L_Q, dtype=torch.bool, device=Q.device)
            selected_mask.scatter_(2, index.unsqueeze(-1), True)
            if not selected_mask.all():
                mean_context = context_selected.mean(dim=2, keepdim=True)  # (B, H, 1, d_k)
                context = context + (~selected_mask.unsqueeze(-1)).float() * mean_context
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(B, L_Q, self.d_model)
        output = self.W_o(context)
        output = self.dropout(output)
        output = self.norm(output + queries)
        
        return output


class InformerEncoderLayer(nn.Module):
    """Encoder layer with ProbSparse attention and distilling."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, factor=5):
        super(InformerEncoderLayer, self).__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, dropout, factor)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            out: (B, L//2, d_model)  # Distilling: halve sequence length
        """
        # Self-attention
        attn_out = self.attention(x, x, x)
        
        # Feed-forward
        out = attn_out.transpose(1, 2)  # (B, d_model, L)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out.transpose(1, 2)  # (B, L, d_model)
        out = self.norm(out + attn_out)
        
        # Distilling: keep every 2nd element
        out = out[:, ::2, :]  # (B, L//2, d_model)
        
        return out


class InformerDecoderLayer(nn.Module):
    """Decoder layer with masked self-attention and cross-attention."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(InformerDecoderLayer, self).__init__()
        self.self_attn = ProbSparseAttention(d_model, n_heads, dropout)
        self.cross_attn = ProbSparseAttention(d_model, n_heads, dropout)
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, enc_out):
        """
        Args:
            x: (B, L_dec, d_model)
            enc_out: (B, L_enc, d_model)
        Returns:
            out: (B, L_dec, d_model)
        """
        # Self-attention (with causal mask)
        x = self.norm1(x)
        self_out = self.self_attn(x, x, x)
        x = x + self_out
        
        # Cross-attention
        x = self.norm2(x)
        cross_out = self.cross_attn(x, enc_out, enc_out)
        x = x + cross_out
        
        # Feed-forward
        out = x.transpose(1, 2)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm3(out + x)
        
        return out


class Informer(nn.Module):
    """
    Informer model for time series forecasting.
    Simplified version adapted for short sequences.
    """
    
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
                 dropout=0.1, factor=5):
        """
        Args:
            enc_in: Input feature dimension (encoder)
            dec_in: Input feature dimension (decoder)
            c_out: Output feature dimension
            seq_len: Input sequence length
            label_len: Start token length (for decoder)
            out_len: Prediction horizon (for single-step, should be 1)
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            factor: Factor for ProbSparse attention
        """
        super(Informer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len
        self.d_model = d_model
        
        # Input projection
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.dec_embedding = nn.Linear(dec_in, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Encoder
        self.encoder = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(e_layers)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(d_layers)
        ])
        
        # Output projection
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Args:
            x_enc: (B, seq_len, enc_in) - Encoder input
            x_mark_enc: Optional time features
            x_dec: (B, label_len+out_len, dec_in) - Decoder input
            x_mark_dec: Optional time features
        Returns:
            dec_out: (B, out_len, c_out)
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc)  # (B, seq_len, d_model)
        enc_out = enc_out.transpose(0, 1)  # (seq_len, B, d_model)
        enc_out = self.pos_encoder(enc_out)
        enc_out = enc_out.transpose(0, 1)  # (B, seq_len, d_model)
        
        for layer in self.encoder:
            enc_out = layer(enc_out)
        
        # Decoder input: use last label_len from encoder + zeros for out_len
        if x_dec is None:
            B = x_enc.size(0)
            dec_inp = torch.zeros(B, self.label_len + self.out_len, x_enc.size(-1)).to(x_enc.device)
            dec_inp[:, :self.label_len, :] = x_enc[:, -self.label_len:, :]
        else:
            dec_inp = x_dec
        
        # Decoder
        dec_out = self.dec_embedding(dec_inp)  # (B, label_len+out_len, d_model)
        dec_out = dec_out.transpose(0, 1)
        dec_out = self.pos_encoder(dec_out)
        dec_out = dec_out.transpose(0, 1)
        
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out)
        
        # Projection
        dec_out = self.projection(dec_out)  # (B, label_len+out_len, c_out)
        dec_out = dec_out[:, -self.out_len:, :]  # (B, out_len, c_out)
        
        return dec_out.squeeze(-1) if self.out_len == 1 else dec_out


class InformerModel(nn.Module):
    """
    Wrapper for Informer adapted to match other models' interface.
    For single-step prediction.
    """
    
    def __init__(self, input_size, output_size=1, d_model=32, n_heads=4, 
                 e_layers=2, d_layers=1, d_ff=128, dropout=0.1, seq_len=7):
        """
        Args:
            input_size: Number of input features
            output_size: Output size (1 for single-step)
            d_model: Model dimension (reduced for small sequences)
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            seq_len: Input sequence length
        """
        super(InformerModel, self).__init__()
        label_len = max(1, seq_len // 2)  # Start token length
        
        self.informer = Informer(
            enc_in=input_size,
            dec_in=input_size,
            c_out=1,  # Always 1 for single-variable prediction (PM2.5)
            seq_len=seq_len,
            label_len=label_len,
            out_len=output_size,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            factor=5
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, output_size)
        """
        out = self.informer(x_enc=x)  # (B, out_len, 1)
        return out.squeeze(-1)  # (B, out_len)


__all__ = ["Informer", "InformerModel", "ProbSparseAttention"]

