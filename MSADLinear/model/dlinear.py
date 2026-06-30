"""
DLinear: Decomposition Linear Model for Time Series Forecasting

This module implements DLinear model based on:
"Are Transformers Effective for Time Series Forecasting?"
by Zeng et al., AAAI 2023

Key features:
- Series Decomposition: Separate trend and seasonal components
- Linear layers for trend and seasonal prediction
- Simple but effective architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    """
    Moving average block to extract the trend component.
    """
    
    def __init__(self, kernel_size, stride=1):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, C) or (B, C, L)
        Returns:
            output: (B, L, C) or (B, C, L) depending on input
        """
        # Handle (B, L, C) format
        if x.dim() == 3 and x.shape[1] != self.kernel_size:
            # Convert to (B, C, L) for pooling
            x = x.permute(0, 2, 1)  # (B, C, L)
            # Padding
            front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
            end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
            x = torch.cat([front, x, end], dim=-1)
            x = self.avg(x)  # (B, C, L')
            x = x.permute(0, 2, 1)  # (B, L', C)
        else:
            # Already in (B, C, L) format
            front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
            end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
            x = torch.cat([front, x, end], dim=-1)
            x = self.avg(x)
        
        return x


class SeriesDecomposition(nn.Module):
    """
    Series decomposition block.
    """
    
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, C)
        Returns:
            trend: (B, L, C)
            seasonal: (B, L, C)
        """
        # Extract trend using moving average
        trend = self.moving_avg(x)  # (B, L', C)
        
        # Adjust length if needed
        if trend.shape[1] != x.shape[1]:
            # Interpolate or pad to match original length
            if trend.shape[1] < x.shape[1]:
                # Interpolate
                trend = F.interpolate(
                    trend.permute(0, 2, 1), 
                    size=x.shape[1], 
                    mode='linear', 
                    align_corners=False
                ).permute(0, 2, 1)
            else:
                # Crop
                trend = trend[:, :x.shape[1], :]
        
        # Seasonal component = original - trend
        seasonal = x - trend
        
        return trend, seasonal


class DLinear(nn.Module):
    """
    DLinear: Decomposition Linear Model.
    """
    
    def __init__(self, seq_len, pred_len, individual=False, enc_in=1, kernel_size=25):
        """
        Args:
            seq_len: Input sequence length
            pred_len: Prediction horizon
            individual: If True, each channel has its own linear layer
            enc_in: Number of input channels (features)
            kernel_size: Kernel size for moving average (trend extraction)
        """
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in
        
        # Adjust kernel_size for short sequences
        if kernel_size > seq_len:
            kernel_size = max(3, seq_len // 2) if seq_len > 2 else seq_len
        
        # Series decomposition
        self.decomposition = SeriesDecomposition(kernel_size)
        
        # Linear layers for trend and seasonal
        if individual:
            # Each channel has its own linear layer
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
        else:
            # Shared linear layer for all channels
            self.Linear_Trend = nn.Linear(seq_len, pred_len)
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, C) where L is seq_len, C is enc_in
        Returns:
            output: (B, pred_len) - prediction for target variable
        """
        B, L, C = x.shape
        
        # Decompose into trend and seasonal
        trend, seasonal = self.decomposition(x)  # Both: (B, L, C)
        
        # Process trend component
        if self.individual:
            trend_out = []
            for i in range(C):
                # (B, L) -> (B, pred_len)
                trend_i = self.Linear_Trend[i](trend[:, :, i])
                trend_out.append(trend_i)
            trend_out = torch.stack(trend_out, dim=-1)  # (B, pred_len, C)
        else:
            # (B, L, C) -> (B, C, L) -> process each channel -> (B, C, pred_len) -> (B, pred_len, C)
            trend = trend.permute(0, 2, 1)  # (B, C, L)
            trend_out = []
            for i in range(C):
                trend_i = self.Linear_Trend(trend[:, i, :])  # (B, pred_len)
                trend_out.append(trend_i)
            trend_out = torch.stack(trend_out, dim=-1)  # (B, pred_len, C)
        
        # Process seasonal component
        if self.individual:
            seasonal_out = []
            for i in range(C):
                seasonal_i = self.Linear_Seasonal[i](seasonal[:, :, i])
                seasonal_out.append(seasonal_i)
            seasonal_out = torch.stack(seasonal_out, dim=-1)  # (B, pred_len, C)
        else:
            seasonal = seasonal.permute(0, 2, 1)  # (B, C, L)
            seasonal_out = []
            for i in range(C):
                seasonal_i = self.Linear_Seasonal(seasonal[:, i, :])  # (B, pred_len)
                seasonal_out.append(seasonal_i)
            seasonal_out = torch.stack(seasonal_out, dim=-1)  # (B, pred_len, C)
        
        # Combine trend and seasonal
        output = trend_out + seasonal_out  # (B, pred_len, C)
        
        # For multivariate input, we typically predict only the target variable (first channel)
        if C > 1:
            output = output[:, :, 0]  # (B, pred_len)
        else:
            output = output.squeeze(-1)  # (B, pred_len)
        
        return output


class DLinearModel(nn.Module):
    """
    Wrapper for DLinear adapted to match other models' interface.
    For single-step prediction (pred_len=1).
    """
    
    def __init__(self, input_size, output_size=1, seq_len=7, individual=False, kernel_size=3):
        """
        Args:
            input_size: Number of input features
            output_size: Output size (1 for single-step)
            seq_len: Input sequence length
            individual: If True, each channel has its own linear layer
            kernel_size: Kernel size for moving average (adjusted for short sequences)
        """
        super(DLinearModel, self).__init__()
        
        # Adjust kernel_size for short sequences
        if kernel_size > seq_len:
            kernel_size = max(3, seq_len // 2) if seq_len > 2 else seq_len
        
        self.dlinear = DLinear(
            seq_len=seq_len,
            pred_len=output_size,
            individual=individual,
            enc_in=input_size,
            kernel_size=kernel_size
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, output_size)
        """
        return self.dlinear(x)


class ChannelAttention(nn.Module):
    """
    轻量级通道注意力模块
    用于增强多变量时间序列中不同特征的重要性建模
    """
    def __init__(self, num_channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.num_channels = num_channels
        
        # 使用全局平均池化和最大池化捕获通道统计信息
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 共享的MLP网络
        hidden = max(1, num_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels, bias=False)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, L)
        Returns:
            attention weights: (B, C, 1)
        """
        # 全局平均池化和最大池化
        avg_out = self.mlp(self.avg_pool(x).squeeze(-1))  # (B, C)
        max_out = self.mlp(self.max_pool(x).squeeze(-1))  # (B, C)
        
        # 融合并应用sigmoid
        attention = torch.sigmoid(avg_out + max_out).unsqueeze(-1)  # (B, C, 1)
        
        return attention


class DLinearWithAttention(nn.Module):
    """
    DLinear + 通道注意力增强版
    在DLinear基础上添加轻量级通道注意力，提升多变量特征建模能力
    """
    
    def __init__(self, seq_len, pred_len, individual=False, enc_in=1, kernel_size=25, use_attention=True):
        """
        Args:
            seq_len: Input sequence length
            pred_len: Prediction horizon
            individual: If True, each channel has its own linear layer
            enc_in: Number of input channels (features)
            kernel_size: Kernel size for moving average (trend extraction)
            use_attention: Whether to use channel attention
        """
        super(DLinearWithAttention, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in
        self.use_attention = use_attention
        
        # Adjust kernel_size for short sequences
        if kernel_size > seq_len:
            kernel_size = max(3, seq_len // 2) if seq_len > 2 else seq_len
        
        # Series decomposition
        self.decomposition = SeriesDecomposition(kernel_size)
        
        # Channel attention (only if enc_in > 1)
        if use_attention and enc_in > 1:
            self.channel_attention = ChannelAttention(enc_in, reduction=4)
        else:
            self.channel_attention = None
        
        # Linear layers for trend and seasonal
        if individual:
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(enc_in)
            ])
        else:
            self.Linear_Trend = nn.Linear(seq_len, pred_len)
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, C) where L is seq_len, C is enc_in
        Returns:
            output: (B, pred_len) - prediction for target variable
        """
        B, L, C = x.shape
        
        # Decompose into trend and seasonal
        trend, seasonal = self.decomposition(x)  # Both: (B, L, C)
        
        # Apply channel attention if enabled
        if self.channel_attention is not None:
            # (B, L, C) -> (B, C, L)
            x_permuted = x.permute(0, 2, 1)
            attention_weights = self.channel_attention(x_permuted)  # (B, C, 1)
            
            # Apply attention to trend and seasonal
            trend = trend.permute(0, 2, 1)  # (B, C, L)
            seasonal = seasonal.permute(0, 2, 1)  # (B, C, L)
            
            trend = trend * attention_weights  # (B, C, L)
            seasonal = seasonal * attention_weights  # (B, C, L)
            
            trend = trend.permute(0, 2, 1)  # (B, L, C)
            seasonal = seasonal.permute(0, 2, 1)  # (B, L, C)
        
        # Process trend component
        if self.individual:
            trend_out = []
            for i in range(C):
                trend_i = self.Linear_Trend[i](trend[:, :, i])
                trend_out.append(trend_i)
            trend_out = torch.stack(trend_out, dim=-1)  # (B, pred_len, C)
        else:
            trend = trend.permute(0, 2, 1)  # (B, C, L)
            trend_out = []
            for i in range(C):
                trend_i = self.Linear_Trend(trend[:, i, :])  # (B, pred_len)
                trend_out.append(trend_i)
            trend_out = torch.stack(trend_out, dim=-1)  # (B, pred_len, C)
        
        # Process seasonal component
        if self.individual:
            seasonal_out = []
            for i in range(C):
                seasonal_i = self.Linear_Seasonal[i](seasonal[:, :, i])
                seasonal_out.append(seasonal_i)
            seasonal_out = torch.stack(seasonal_out, dim=-1)  # (B, pred_len, C)
        else:
            seasonal = seasonal.permute(0, 2, 1)  # (B, C, L)
            seasonal_out = []
            for i in range(C):
                seasonal_i = self.Linear_Seasonal(seasonal[:, i, :])  # (B, pred_len)
                seasonal_out.append(seasonal_i)
            seasonal_out = torch.stack(seasonal_out, dim=-1)  # (B, pred_len, C)
        
        # Combine trend and seasonal
        output = trend_out + seasonal_out  # (B, pred_len, C)
        
        # For multivariate input, predict only the target variable (first channel)
        if C > 1:
            output = output[:, :, 0]  # (B, pred_len)
        else:
            output = output.squeeze(-1)  # (B, pred_len)
        
        return output


class DLinearModelWithAttention(nn.Module):
    """
    Wrapper for DLinearWithAttention adapted to match other models' interface.
    For single-step prediction (pred_len=1).
    """
    
    def __init__(self, input_size, output_size=1, seq_len=7, individual=False, kernel_size=3, use_attention=True):
        """
        Args:
            input_size: Number of input features
            output_size: Output size (1 for single-step)
            seq_len: Input sequence length
            individual: If True, each channel has its own linear layer
            kernel_size: Kernel size for moving average
            use_attention: Whether to use channel attention
        """
        super(DLinearModelWithAttention, self).__init__()
        
        # Adjust kernel_size for short sequences
        if kernel_size > seq_len:
            kernel_size = max(3, seq_len // 2) if seq_len > 2 else seq_len
        
        self.dlinear = DLinearWithAttention(
            seq_len=seq_len,
            pred_len=output_size,
            individual=individual,
            enc_in=input_size,
            kernel_size=kernel_size,
            use_attention=use_attention
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, output_size)
        """
        return self.dlinear(x)


__all__ = ["DLinear", "DLinearModel", "DLinearWithAttention", "DLinearModelWithAttention", 
           "SeriesDecomposition", "MovingAverage", "ChannelAttention"]

