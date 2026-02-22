"""
Project INTERCEPTOR — Custom LSTM Policy Network
================================================

Custom feature extractor for Stable-Baselines3 that uses an LSTM
to encode temporal maneuver history. This allows the RL agent to
observe and predict the target's evasive patterns over time.

The architecture:
    Input (12-dim) → Linear(128) → ReLU → LSTM(128, 2 layers) → Linear(64) → Output

This is used as the feature extractor inside RecurrentPPO's policy.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class GuidanceLSTMExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using LSTM for temporal encoding.
    
    Processes the 12-dimensional observation through a linear projection,
    then feeds it through a multi-layer LSTM to capture the history of
    target maneuvers.
    """

    def __init__(
        self, 
        observation_space: gym.spaces.Space,
        features_dim: int = 64,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(GuidanceLSTMExtractor, self).__init__(
            observation_space, features_dim
        )
        
        input_dim = observation_space.shape[0]  # 12
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, lstm_hidden_size),
            nn.LayerNorm(lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM temporal encoder
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        
        # Output projection to feature dimension
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better long-term memory
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM feature extractor.
        
        Args:
            observations: Tensor of shape (batch, obs_dim) or (batch, seq, obs_dim)
            
        Returns:
            Features tensor of shape (batch, features_dim)
        """
        # Handle both single-step and sequence inputs
        if observations.dim() == 2:
            # Single timestep: (batch, obs_dim) → (batch, 1, obs_dim)
            observations = observations.unsqueeze(1)
        
        # Project input
        projected = self.input_proj(observations)  # (batch, seq, hidden)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(projected)  # (batch, seq, hidden)
        
        # Take the last timestep's output
        last_output = lstm_out[:, -1, :]  # (batch, hidden)
        
        # Project to feature space
        features = self.output_proj(last_output)  # (batch, features_dim)
        
        return features


class GuidanceTransformerExtractor(BaseFeaturesExtractor):
    """
    Alternative feature extractor using a Transformer encoder.
    
    For agents that need longer-range temporal attention 
    (e.g., detecting periodic maneuver patterns).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super(GuidanceTransformerExtractor, self).__init__(
            observation_space, features_dim
        )
        
        input_dim = observation_space.shape[0]
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)
        
        projected = self.input_proj(observations)
        encoded = self.transformer(projected)
        last_output = encoded[:, -1, :]
        features = self.output_proj(last_output)
        
        return features
