"""
LSTM Model for Time Series Forecasting

This module implements a standard LSTM model as a baseline for time series prediction.
Supports multivariate input and single/multi-step forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """
    Standard LSTM model for time series forecasting.
    
    Architecture:
        - Input layer: receives sequence of features
        - LSTM layers: stacked LSTM layers with dropout
        - Fully connected layer: maps to output dimension
        
    Args:
        input_size (int): Number of input features (multivariate)
        hidden_size (int): Number of hidden units in LSTM layer (default: 64)
        num_layers (int): Number of stacked LSTM layers (default: 2)
        output_size (int): Number of output features (default: 1)
        dropout (float): Dropout rate between LSTM layers (default: 0.2)
        bidirectional (bool): Whether to use bidirectional LSTM (default: False)
    """
    
    def __init__(self, 
                 input_size, 
                 hidden_size=64, 
                 num_layers=2, 
                 output_size=1,
                 dropout=0.2,
                 bidirectional=False):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Determine the LSTM output dimension
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected output layer
        self.fc = nn.Linear(lstm_output_dim, output_size)
        
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # x shape: (batch_size, seq_len, input_size)
        
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        # hidden: tuple of (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the output of the last time step
        # Take the last time step output: (batch_size, hidden_size * num_directions)
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layer
        output = self.fc(last_output)
        
        # output shape: (batch_size, output_size)
        return output
    
    def predict(self, x):
        """
        Make predictions (same as forward, but sets model to eval mode).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, output_size)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def create_lstm_model(input_size, 
                      hidden_size=64, 
                      num_layers=2, 
                      output_size=1,
                      dropout=0.2,
                      bidirectional=False,
                      device='cpu'):
    """
    Factory function to create and initialize an LSTM model.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden units (default: 64)
        num_layers (int): Number of LSTM layers (default: 2)
        output_size (int): Number of output features (default: 1)
        dropout (float): Dropout rate (default: 0.2)
        bidirectional (bool): Use bidirectional LSTM (default: False)
        device (str): Device to place model on (default: 'cpu')
        
    Returns:
        LSTMModel: Initialized LSTM model
    """
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    model = model.to(device)
    return model


# Default configuration for baseline LSTM (as mentioned in experimental settings)
DEFAULT_LSTM_CONFIG = {
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': False
}

