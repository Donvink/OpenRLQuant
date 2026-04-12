"""
train/policy_network.py
────────────────────────
Custom policy network: Transformer temporal encoder + multi-asset fusion.

Architecture:
  Input: [batch, lookback, n_features * n_stocks] + [portfolio_state]
         ↓ reshape per-stock
  TransformerEncoder: captures temporal dependencies in each stock's features
         ↓ pool → stock embeddings [batch, n_stocks, d_model]
  CrossAttentionFusion: stocks attend to each other (inter-asset relationships)
         ↓ flatten
  MLP head → Actor (action logits) + Critic (state value)

Why Transformer over LSTM:
  - Parallelizable training (faster)
  - Longer effective memory (attention vs. vanishing gradient)
  - Multi-head attention = multiple "market regime detectors"
  - Position encoding captures day-of-week / month seasonality
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

logger = logging.getLogger(__name__)


# ─── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    Injects time-step information without learned parameters.
    """

    def __init__(self, d_model: int, max_len: int = 252, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─── Per-Stock Temporal Encoder ───────────────────────────────────────────────

class StockTemporalEncoder(nn.Module):
    """
    Processes one stock's feature sequence → fixed-size embedding.

    Flow:
      [batch, seq_len, n_feat] → linear projection → TransformerEncoder → pool → [batch, d_model]
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 252,
    ):
        super().__init__()
        self.d_model = d_model

        # Project raw features → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # [batch, seq, features] convention
            norm_first=True,    # Pre-LN for better stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Learnable [CLS] token — aggregates sequence info
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, n_features]
        Returns: [batch, d_model]
        """
        batch = x.size(0)

        # Project to d_model
        x = self.input_proj(x)  # [batch, seq, d_model]

        # Prepend CLS token
        cls = self.cls_token.expand(batch, -1, -1)  # [batch, 1, d_model]
        x = torch.cat([cls, x], dim=1)              # [batch, seq+1, d_model]

        x = self.pos_enc(x)
        x = self.transformer(x)   # [batch, seq+1, d_model]

        # CLS token output as sequence summary
        return x[:, 0, :]         # [batch, d_model]


# ─── Cross-Asset Attention ────────────────────────────────────────────────────

class CrossAssetAttention(nn.Module):
    """
    Multi-head attention over stock embeddings.
    Each stock attends to all other stocks → captures correlations.

    This models "when NVDA spikes, AMD tends to follow" implicitly.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, n_stocks, d_model]
        Returns: [batch, n_stocks, d_model]
        """
        # Self-attention over stocks
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


# ─── Full Features Extractor (SB3-compatible) ─────────────────────────────────

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible features extractor.
    Takes flat observation vector, reconstructs per-stock windows, applies Transformer.

    Registered with SB3 via policy_kwargs={"features_extractor_class": TransformerFeaturesExtractor}
    """

    def __init__(
        self,
        observation_space: gym.Space,
        n_stocks: int,
        n_features: int,
        lookback: int,
        portfolio_state_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        n_cross_asset_layers: int = 1,
        dropout: float = 0.1,
    ):
        # features_dim is the output dimension of this extractor
        features_dim = d_model * n_stocks + portfolio_state_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.n_stocks = n_stocks
        self.n_features = n_features
        self.lookback = lookback
        self.portfolio_state_dim = portfolio_state_dim
        self.d_model = d_model

        # One shared temporal encoder (weight-sharing across stocks)
        # Alternative: separate encoders per stock (more params, potentially better)
        self.temporal_encoder = StockTemporalEncoder(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_transformer_layers,
            dropout=dropout,
        )

        # Cross-asset attention layers
        self.cross_asset = nn.Sequential(*[
            CrossAssetAttention(d_model, n_heads, dropout)
            for _ in range(n_cross_asset_layers)
        ])

        # Portfolio state encoder
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(portfolio_state_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, portfolio_state_dim),
        )

        self._initialize_weights()
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"TransformerFeaturesExtractor: "
            f"{n_params:,} params | "
            f"output_dim={features_dim}"
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [batch, obs_dim]  (flat observation from TradingEnv)
        Returns: [batch, features_dim]
        """
        batch = obs.size(0)
        temporal_dim = self.lookback * self.n_features * self.n_stocks

        # Split obs into temporal features and portfolio state
        temporal_flat = obs[:, :temporal_dim]
        portfolio_state = obs[:, temporal_dim:]

        # Reshape temporal: [batch, n_stocks, lookback, n_features]
        temporal = temporal_flat.view(batch, self.n_stocks, self.lookback, self.n_features)

        # Encode each stock independently: [batch * n_stocks, lookback, n_features]
        temporal_2d = temporal.reshape(batch * self.n_stocks, self.lookback, self.n_features)
        stock_embs = self.temporal_encoder(temporal_2d)  # [batch * n_stocks, d_model]
        stock_embs = stock_embs.view(batch, self.n_stocks, self.d_model)

        # Cross-asset attention
        for layer in self.cross_asset:
            stock_embs = layer(stock_embs)

        # Flatten stock embeddings
        stock_flat = stock_embs.reshape(batch, self.n_stocks * self.d_model)

        # Encode portfolio state
        port_enc = self.portfolio_encoder(portfolio_state)

        return torch.cat([stock_flat, port_enc], dim=-1)


# ─── MLP Extractor (lightweight baseline) ─────────────────────────────────────

class MLPFeaturesExtractor(BaseFeaturesExtractor):
    """
    Simpler MLP-based features extractor.
    Much faster to train, good for initial experiments and debugging.
    Use this FIRST to validate the pipeline, then switch to Transformer.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        net_arch: List[int] = None,
        dropout: float = 0.1,
    ):
        net_arch = net_arch or [512, 256, 128]
        features_dim = net_arch[-1]
        super().__init__(observation_space, features_dim=features_dim)

        layers = []
        in_dim = int(np.prod(observation_space.shape))
        for out_dim in net_arch:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─── Custom ActorCriticPolicy ─────────────────────────────────────────────────

class TradingPolicy(ActorCriticPolicy):
    """
    Full PPO policy with Transformer features extractor.

    Usage with SB3:
        model = PPO(
            TradingPolicy,
            env,
            policy_kwargs={
                "features_extractor_class": TransformerFeaturesExtractor,
                "features_extractor_kwargs": {
                    "n_stocks": 8, "n_features": 50, "lookback": 30, ...
                },
                "net_arch": dict(pi=[128, 64], vf=[128, 64]),
                "activation_fn": nn.GELU,
            },
        )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ─── Helper: build policy_kwargs for SB3 ─────────────────────────────────────

def make_transformer_policy_kwargs(
    n_stocks: int,
    n_features: int,
    lookback: int,
    portfolio_state_dim: int,
    d_model: int = 128,
    n_heads: int = 4,
    n_transformer_layers: int = 2,
    use_transformer: bool = True,
) -> dict:
    """
    Build policy_kwargs dict for SB3's PPO/SAC.

    Args:
        use_transformer: if False, uses simpler MLP extractor (faster for debugging)
    """
    if use_transformer:
        return {
            "features_extractor_class": TransformerFeaturesExtractor,
            "features_extractor_kwargs": {
                "n_stocks": n_stocks,
                "n_features": n_features,
                "lookback": lookback,
                "portfolio_state_dim": portfolio_state_dim,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_transformer_layers": n_transformer_layers,
            },
            "net_arch": dict(pi=[128, 64], vf=[256, 128]),
            "activation_fn": nn.GELU,
            "share_features_extractor": True,  # Actor and Critic share the Transformer
        }
    else:
        return {
            "features_extractor_class": MLPFeaturesExtractor,
            "features_extractor_kwargs": {
                "net_arch": [512, 256, 128],
                "dropout": 0.1,
            },
            "net_arch": dict(pi=[64], vf=[64]),
            "activation_fn": nn.GELU,
        }
