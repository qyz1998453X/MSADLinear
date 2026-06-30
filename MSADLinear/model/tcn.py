"""
TCN (Temporal Convolutional Network) for Time Series Forecasting
"""
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.init as init

# Chomp1d
class Chomp1d(nn.Module):
    """
    Remove right-side padding to preserve causality.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        # x: (B, C, L)
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

# Temporal Block
class TemporalBlock(nn.Module):
    """
    Two-layer causal dilated convolution block with residual connection.
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super().__init__()

        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.conv1.weight, 0.0, 0.01)
        init.normal_(self.conv2.weight, 0.0, 0.01)
        if self.downsample is not None:
            init.normal_(self.downsample.weight, 0.0, 0.01)

    def forward(self, x):
        """
        x: (B, C, L)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# Temporal Convolutional Network (TCN backbone)
class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilation.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        self.num_inputs = num_inputs

        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Input:
            x: (B, L, C)
        Output:
            (B, C_out, L)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B,L,C), got {x.shape}")

        # enforce (B, C, L)
        if x.size(-1) == self.num_inputs:
            x = x.transpose(1, 2).contiguous()
        elif x.size(1) == self.num_inputs:
            pass
        else:
            raise ValueError(
                f"Cannot infer input layout for shape {x.shape}, "
                f"expected feature dim={self.num_inputs}"
            )

        return self.network(x)


# Final TCN Forecasting Model
class TCN(nn.Module):
    """
    TCN model for multi-step time series forecasting.

    Strategy:
    - Use last hidden time step
    - Linear projection to prediction horizon
    """

    def __init__(
        self,
        input_size,
        output_size=1,
        num_channels=None,
        kernel_size=2,
        dropout=0.2,
    ):
        super().__init__()

        if num_channels is None:
            num_channels = [32, 32, 32]

        self.input_size = input_size
        self.output_size = output_size

        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """
        x: (B, L, C)
        return: (B, output_size)
        """
        y = self.tcn(x)          # (B, C_out, L)
        y = y[:, :, -1]          # last time step
        out = self.linear(y)     # (B, output_size)
        return out

__all__ = ["TCN", "TemporalConvNet", "TemporalBlock"]
