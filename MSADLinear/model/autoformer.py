"""
Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

This module implements Autoformer model based on:
"Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
by Wu et al., NeurIPS 2021

Key features:
- Decomposition-based Auto-Correlation mechanism (replaces self-attention)
- Series Decomposition Block (trend + seasonal components)
- Auto-Correlation mechanism for capturing periodic dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SeriesDecomp(nn.Module):
    """
    Series Decomposition Block.
    Decomposes time series into trend and seasonal components using moving average.
    """
    
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = MovingAverage(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            trend: (B, L, D)
            seasonal: (B, L, D)
        """
        moving_mean = self.moving_avg(x)
        trend = moving_mean
        seasonal = x - moving_mean
        return trend, seasonal


class MovingAverage(nn.Module):
    """Moving average block to highlight the trend of time series."""
    
    def __init__(self, kernel_size, stride=1):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            moving_avg: (B, L, D)
        """
        # Pad on both sides
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # Apply moving average
        x_padded = x_padded.permute(0, 2, 1)  # (B, D, L)
        moving_avg = self.avg(x_padded)  # (B, D, L')
        moving_avg = moving_avg.permute(0, 2, 1)  # (B, L', D)
        
        return moving_avg


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


class AutoCorrelation(nn.Module):
    """
    Auto-Correlation mechanism.
    Replaces self-attention with auto-correlation for capturing periodic dependencies.
    """
    
    def __init__(self, d_model, n_heads, factor=1, dropout=0.1):
        super(AutoCorrelation, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        B, L, H, D = values.shape
        # find top k
        top_k = int(self.factor * math.log(L))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # (B, L)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k)[1]  # (top_k,)
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)  # (B, top_k)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)  # (B, top_k)
        # aggregation
        tmp_values = values  # (B, L, H, D)
        delays_agg = torch.zeros_like(values).float()
        for i, topk_index in enumerate(index):
            pattern = torch.roll(tmp_values, -int(topk_index), -2)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, H, D))
        return delays_agg
    
    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation
        This is for the inference phase.
        """
        B, L, H, D = values.shape
        top_k = int(self.factor * math.log(L))
        # index init
        init_index = torch.arange(L).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(B, 1, 1, 1).to(values.device)
        # find top k
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # (B, L)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)  # (B, top_k)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 1)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(3)
            pattern = torch.gather(tmp_values, dim=1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(3).repeat(1, 1, H, D))
        return delays_agg
    
    def forward(self, queries, keys, values):
        """
        Args:
            queries: (B, L_Q, d_model)
            keys: (B, L_K, d_model)
            values: (B, L_V, d_model)
        Returns:
            out: (B, L_Q, d_model)
        """
        B, L_Q, _ = queries.shape
        _, L_K, _ = keys.shape
        
        # Linear projections
        Q = self.W_q(queries).view(B, L_Q, self.n_heads, self.d_k)  # (B, L_Q, H, d_k)
        K = self.W_k(keys).view(B, L_K, self.n_heads, self.d_k)  # (B, L_K, H, d_k)
        V = self.W_v(values).view(B, L_K, self.n_heads, self.d_k)  # (B, L_V, H, d_k)
        
        # For short sequences (seq_len <= 10), use simplified aggregation
        if L_Q <= 10:
            # Simplified autocorrelation for short sequences
            # Use standard scaled dot-product as approximation (standard attention)
            # Transpose for matmul: (B, H, L_Q, d_k) @ (B, H, d_k, L_K) = (B, H, L_Q, L_K)
            Q_t = Q.permute(0, 2, 1, 3)  # (B, H, L_Q, d_k)
            K_t = K.permute(0, 2, 3, 1)  # (B, H, d_k, L_K)
            scores = torch.matmul(Q_t, K_t) / math.sqrt(self.d_k)  # (B, H, L_Q, L_K)
            attn_weights = torch.softmax(scores, dim=-1)  # (B, H, L_Q, L_K)
            attn_weights = self.dropout(attn_weights)
            
            V_t = V.permute(0, 2, 1, 3)  # (B, H, L_K, d_k)
            out = torch.matmul(attn_weights, V_t)  # (B, H, L_Q, d_k)
            out = out.permute(0, 2, 1, 3)  # (B, L_Q, H, d_k)
        else:
            # Compute autocorrelation using FFT for longer sequences
            # Pad to same length for FFT
            max_len = max(L_Q, L_K)
            Q_padded = F.pad(Q, (0, 0, 0, 0, 0, max_len - L_Q))  # (B, max_len, H, d_k)
            K_padded = F.pad(K, (0, 0, 0, 0, 0, max_len - L_K))  # (B, max_len, H, d_k)
            
            Q_fft = torch.fft.rfft(Q_padded.permute(0, 2, 3, 1).contiguous(), dim=-1)  # (B, H, d_k, max_len//2+1)
            K_fft = torch.fft.rfft(K_padded.permute(0, 2, 3, 1).contiguous(), dim=-1)  # (B, H, d_k, max_len//2+1)
            
            # Cross-correlation in frequency domain
            corr = torch.fft.irfft(Q_fft * torch.conj(K_fft), dim=-1, n=max_len)  # (B, H, d_k, max_len)
            corr = corr.permute(0, 3, 1, 2)  # (B, max_len, H, d_k)
            corr = corr[:, :L_Q, :, :]  # (B, L_Q, H, d_k)
            
            # Time delay aggregation
            if self.training:
                out = self.time_delay_agg_training(V, corr)
            else:
                out = self.time_delay_agg_inference(V, corr)
            # Time delay aggregation for longer sequences
            if self.training:
                out = self.time_delay_agg_training(V, corr)
            else:
                out = self.time_delay_agg_inference(V, corr)
        
        # Reshape and project
        out = out.contiguous().view(B, L_Q, self.d_model)
        out = self.W_o(out)
        out = self.dropout(out)
        
        return out


class EncoderLayer(nn.Module):
    """Autoformer Encoder Layer with Auto-Correlation and Decomposition."""
    
    def __init__(self, d_model, n_heads, d_ff, moving_avg=25, factor=1, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.auto_corr = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (B, L, d_model)
            attn_mask: optional mask (not used in auto-correlation)
        Returns:
            out: (B, L, d_model)
        """
        # Auto-Correlation
        x_norm = self.norm1(x)
        auto_corr_out = self.auto_corr(x_norm, x_norm, x_norm)
        x = x + auto_corr_out
        
        # Decomposition
        trend1, seasonal1 = self.decomp1(x)
        
        # Feed-forward
        x = seasonal1
        x_norm = self.norm2(x)
        x_trans = x_norm.transpose(1, 2)  # (B, d_model, L)
        x_trans = self.conv1(x_trans)
        x_trans = F.gelu(x_trans)
        x_trans = self.dropout(x_trans)
        x_trans = self.conv2(x_trans)
        x_trans = x_trans.transpose(1, 2)  # (B, L, d_model)
        
        # Decomposition
        trend2, seasonal2 = self.decomp2(x + x_trans)
        
        return trend1 + trend2, seasonal2


class DecoderLayer(nn.Module):
    """Autoformer Decoder Layer."""
    
    def __init__(self, d_model, n_heads, d_ff, moving_avg=25, factor=1, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.auto_corr1 = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.auto_corr2 = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        """
        Args:
            x: (B, L_dec, d_model)
            cross: (B, L_enc, d_model)
        Returns:
            trend: (B, L_dec, d_model)
            seasonal: (B, L_dec, d_model)
        """
        # Self Auto-Correlation
        x_norm = self.norm1(x)
        auto_corr_out = self.auto_corr1(x_norm, x_norm, x_norm)
        x = x + auto_corr_out
        trend1, seasonal1 = self.decomp1(x)
        
        # Cross Auto-Correlation
        x_norm = self.norm2(seasonal1)
        cross_corr_out = self.auto_corr2(x_norm, cross, cross)
        seasonal1 = seasonal1 + cross_corr_out
        trend2, seasonal2 = self.decomp2(seasonal1)
        
        # Feed-forward
        x = seasonal2
        x_norm = self.norm3(x)
        x_trans = x_norm.transpose(1, 2)
        x_trans = self.conv1(x_trans)
        x_trans = F.gelu(x_trans)
        x_trans = self.dropout(x_trans)
        x_trans = self.conv2(x_trans)
        x_trans = x_trans.transpose(1, 2)
        trend3, seasonal3 = self.decomp3(x + x_trans)
        
        return trend1 + trend2 + trend3, seasonal3


class Autoformer(nn.Module):
    """
    Autoformer model for time series forecasting.
    """
    
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
                 moving_avg=25, factor=1, dropout=0.1):
        """
        Args:
            enc_in: Input feature dimension (encoder)
            dec_in: Input feature dimension (decoder)
            c_out: Output feature dimension
            seq_len: Input sequence length
            label_len: Start token length (for decoder)
            out_len: Prediction horizon
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            moving_avg: Kernel size for moving average
            factor: Factor for auto-correlation
            dropout: Dropout rate
        """
        super(Autoformer, self).__init__()
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
            EncoderLayer(d_model, n_heads, d_ff, moving_avg, factor, dropout)
            for _ in range(e_layers)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, moving_avg, factor, dropout)
            for _ in range(d_layers)
        ])
        
        # Output projection
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Args:
            x_enc: (B, seq_len, enc_in)
            x_dec: (B, label_len+out_len, dec_in)
        Returns:
            dec_out: (B, out_len, c_out)
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc)  # (B, seq_len, d_model)
        enc_out = enc_out.transpose(0, 1)  # (seq_len, B, d_model)
        enc_out = self.pos_encoder(enc_out)
        enc_out = enc_out.transpose(0, 1)  # (B, seq_len, d_model)
        
        enc_trend = None
        for layer in self.encoder:
            enc_trend, enc_out = layer(enc_out)
            if enc_trend is None:
                enc_trend = torch.zeros_like(enc_out)
        
        # Decoder input
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
        
        dec_trend = None
        for layer in self.decoder:
            dec_trend, dec_out = layer(dec_out, enc_out)
            if dec_trend is None:
                dec_trend = torch.zeros_like(dec_out)
        
        # Final projection
        dec_out = self.projection(dec_out)  # (B, label_len+out_len, c_out)
        dec_out = dec_out[:, -self.out_len:, :]  # (B, out_len, c_out)
        
        # Add trend
        if dec_trend is not None:
            trend_proj = self.projection(dec_trend[:, -self.out_len:, :])
            dec_out = dec_out + trend_proj
        
        return dec_out.squeeze(-1) if self.out_len == 1 else dec_out


class AutoformerModel(nn.Module):
    """
    Wrapper for Autoformer adapted to match other models' interface.
    For single-step prediction.
    """
    
    def __init__(self, input_size, output_size=1, d_model=32, n_heads=4,
                 e_layers=2, d_layers=1, d_ff=128, dropout=0.1, seq_len=7, moving_avg=3):
        """
        Args:
            input_size: Number of input features
            output_size: Output size (1 for single-step)
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            seq_len: Input sequence length
            moving_avg: Kernel size for moving average (adjusted for short sequences)
        """
        super(AutoformerModel, self).__init__()
        label_len = max(1, seq_len // 2)
        # Adjust moving_avg for short sequences
        moving_avg = min(moving_avg, seq_len // 2) if seq_len > 4 else 3
        
        self.autoformer = Autoformer(
            enc_in=input_size,
            dec_in=input_size,
            c_out=output_size,
            seq_len=seq_len,
            label_len=label_len,
            out_len=output_size,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            moving_avg=moving_avg,
            factor=1,
            dropout=dropout
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, output_size)
        """
        return self.autoformer(x_enc=x)


__all__ = ["Autoformer", "AutoformerModel", "AutoCorrelation", "SeriesDecomp"]

