"""
Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

Based on: "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting"
by Wu et al. (2021)

Key innovations:
- Auto-correlation mechanism replacing self-attention
- Series decomposition architecture
- Deep decomposition with trend-cyclical decomposition
- Efficient for capturing long-term dependencies
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast
from torch.utils.data import DataLoader
import numpy as np


class SeriesDecomposition(nn.Module):
    """
    Series Decomposition Block.

    Decomposes time series into trend and seasonal components
    using moving average.
    """

    def __init__(self, kernel_size: int = 25):
        """
        Initialize series decomposition.

        Args:
            kernel_size: Kernel size for moving average
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Decompose series into trend and seasonal components.

        Args:
            x: Input tensor (batch, seq_len, features)

        Returns:
            Tuple of (seasonal, trend)
        """
        # Padding for edge effects
        batch_size, seq_len, features = x.shape

        # Transpose for AvgPool1d: (batch, features, seq_len)
        x_transposed = x.transpose(1, 2)

        # Moving average for trend
        trend = self.avg(x_transposed)
        trend = trend[:, :, :seq_len]  # Trim to original length

        # Transpose back
        trend = trend.transpose(1, 2)

        # Seasonal is residual
        seasonal = x - trend

        return seasonal, trend


class AutoCorrelation(nn.Module):
    """
    Auto-Correlation mechanism.

    Replaces self-attention with auto-correlation to discover period-based
    dependencies efficiently.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        factor: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize auto-correlation.

        Args:
            d_model: Model dimension
            num_heads: Number of heads
            factor: Number of top correlations to use
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.factor = factor

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def time_delay_agg(
        self,
        values: Tensor,
        corr: Tensor,
    ) -> Tensor:
        """
        Time delay aggregation based on correlation.

        Args:
            values: Value tensor (batch, num_heads, seq_len, d_k)
            corr: Correlation tensor (batch, num_heads, seq_len)

        Returns:
            Aggregated values (batch, num_heads, seq_len, d_k)
        """
        batch_size, num_heads, seq_len, d_k = values.shape

        # Find top-k delays based on correlation
        top_k = self.factor
        if top_k == 0:
            top_k = seq_len

        # Get top-k correlation indices
        weights, delays = torch.topk(corr, top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # Initialize output
        output = torch.zeros_like(values)

        # Aggregate with time delays
        for i in range(top_k):
            # Get delay for each position
            delay = delays[:, :, i]  # (batch, num_heads)

            # Roll values by delay
            for b in range(batch_size):
                for h in range(num_heads):
                    d = delay[b, h].item()
                    if d > 0:
                        rolled = torch.roll(values[b, h], shifts=int(d), dims=0)
                        output[b, h] += weights[b, h, i].view(1, 1) * rolled

        return output

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with auto-correlation.

        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional mask (not used in auto-correlation)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.shape

        # Linear projections and reshape
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Auto-correlation using FFT
        # Compute in frequency domain
        Q_fft = torch.fft.rfft(Q, dim=2)
        K_fft = torch.fft.rfft(K, dim=2)

        # Auto-correlation: Q * conj(K)
        corr_fft = Q_fft * torch.conj(K_fft)

        # Back to time domain
        corr = torch.fft.irfft(corr_fft, n=seq_len, dim=2)

        # Normalize correlation
        corr = corr / seq_len

        # Time delay aggregation
        if self.factor > 0:
            # Use top-k correlations
            output = self.time_delay_agg(V, corr.mean(dim=-1))
        else:
            # Use all correlations (standard convolution)
            # Compute weighted sum based on correlation
            corr_weights = F.softmax(corr, dim=2)
            output = torch.einsum('bhsd,bhsl->bhld', V, corr_weights)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        output = self.dropout(output)

        return output


class AutoformerEncoderLayer(nn.Module):
    """
    Single Autoformer encoder layer with auto-correlation and decomposition.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        moving_avg: int = 25,
        factor: int = 1,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        """
        Initialize encoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            moving_avg: Moving average window for decomposition
            factor: Factor for auto-correlation
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()

        self.auto_corr = AutoCorrelation(
            d_model=d_model,
            num_heads=num_heads,
            factor=factor,
            dropout=dropout,
        )

        self.decomp1 = SeriesDecomposition(kernel_size=moving_avg)
        self.decomp2 = SeriesDecomposition(kernel_size=moving_avg)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional mask

        Returns:
            Tuple of (seasonal_output, trend_output)
        """
        # Auto-correlation
        attn_output = self.auto_corr(x, x, x, mask)
        x = x + self.dropout(attn_output)

        # Decomposition 1
        seasonal, trend1 = self.decomp1(x)

        # Feed-forward on seasonal component
        ff_output = self.ff(seasonal)
        seasonal = seasonal + self.dropout(ff_output)

        # Decomposition 2
        seasonal, trend2 = self.decomp2(seasonal)

        # Aggregate trends
        trend = trend1 + trend2

        return seasonal, trend


class AutoformerDecoderLayer(nn.Module):
    """
    Single Autoformer decoder layer with auto-correlation and cross-correlation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        moving_avg: int = 25,
        factor: int = 1,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        """
        Initialize decoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            moving_avg: Moving average window for decomposition
            factor: Factor for auto-correlation
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()

        self.self_corr = AutoCorrelation(
            d_model=d_model,
            num_heads=num_heads,
            factor=factor,
            dropout=dropout,
        )

        self.cross_corr = AutoCorrelation(
            d_model=d_model,
            num_heads=num_heads,
            factor=factor,
            dropout=dropout,
        )

        self.decomp1 = SeriesDecomposition(kernel_size=moving_avg)
        self.decomp2 = SeriesDecomposition(kernel_size=moving_avg)
        self.decomp3 = SeriesDecomposition(kernel_size=moving_avg)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

        # Trend projection
        self.trend_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: Tensor,
        memory_seasonal: Tensor,
        memory_trend: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Decoder input (batch, tgt_len, d_model)
            memory_seasonal: Encoder seasonal output (batch, src_len, d_model)
            memory_trend: Encoder trend output (batch, src_len, d_model)
            mask: Optional mask

        Returns:
            Tuple of (seasonal_output, trend_output)
        """
        # Self auto-correlation
        self_corr_output = self.self_corr(x, x, x, mask)
        x = x + self.dropout(self_corr_output)

        # Decomposition 1
        seasonal, trend1 = self.decomp1(x)

        # Cross auto-correlation with encoder
        cross_corr_output = self.cross_corr(seasonal, memory_seasonal, memory_seasonal)
        seasonal = seasonal + self.dropout(cross_corr_output)

        # Decomposition 2
        seasonal, trend2 = self.decomp2(seasonal)

        # Feed-forward on seasonal
        ff_output = self.ff(seasonal)
        seasonal = seasonal + self.dropout(ff_output)

        # Decomposition 3
        seasonal, trend3 = self.decomp3(seasonal)

        # Aggregate trends
        # Add encoder trend contribution
        trend = trend1 + trend2 + trend3
        trend = trend + self.trend_proj(memory_trend[:, -trend.size(1):, :])

        return seasonal, trend


class Autoformer(nn.Module):
    """
    Autoformer for long-term time series forecasting.

    Uses auto-correlation mechanism and series decomposition
    for efficient modeling of long-term dependencies.
    """

    def __init__(
        self,
        # Input configuration
        input_dim: int = 1,
        output_dim: int = 1,

        # Architecture configuration
        d_model: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        num_heads: int = 8,
        d_ff: int = 512,
        moving_avg: int = 25,
        factor: int = 1,
        dropout: float = 0.1,
        activation: str = 'gelu',

        # Forecasting configuration
        encoder_steps: int = 96,
        decoder_steps: int = 24,
        label_len: int = 48,

        # Training configuration
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize Autoformer.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            d_model: Model dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of heads
            d_ff: Feed-forward dimension
            moving_avg: Moving average window for decomposition
            factor: Factor for auto-correlation (0 for all correlations)
            dropout: Dropout rate
            activation: Activation function
            encoder_steps: Number of historical steps
            decoder_steps: Number of future steps to predict
            label_len: Length of start token
            learning_rate: Learning rate
            device: Device to run on
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.label_len = label_len
        self.learning_rate = learning_rate
        self.device = device

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Linear(output_dim, d_model)

        # Positional encoding
        self.pos_encoding = self._get_positional_encoding(
            max(encoder_steps, decoder_steps + label_len), d_model
        )

        # Initial decomposition
        self.decomp = SeriesDecomposition(kernel_size=moving_avg)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                moving_avg=moving_avg,
                factor=factor,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            AutoformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                moving_avg=moving_avg,
                factor=factor,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_decoder_layers)
        ])

        # Output projections for seasonal and trend
        self.seasonal_proj = nn.Linear(d_model, output_dim)
        self.trend_proj = nn.Linear(d_model, output_dim)

        self.to(device)

    def _get_positional_encoding(self, max_len: int, d_model: int) -> Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(
        self,
        src: Tensor,
        tgt: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            src: Source sequence (batch, encoder_steps, input_dim)
            tgt: Target sequence (batch, decoder_steps, output_dim)

        Returns:
            Predictions (batch, decoder_steps, output_dim)
        """
        batch_size = src.size(0)

        # Encode
        src_embedded = self.input_embedding(src)
        src_embedded = src_embedded + self.pos_encoding[:, :src.size(1), :].to(src.device)

        # Initial decomposition
        seasonal, trend = self.decomp(src_embedded)

        # Encoder
        for layer in self.encoder_layers:
            seasonal, layer_trend = layer(seasonal)
            trend = trend + layer_trend

        memory_seasonal = seasonal
        memory_trend = trend

        # Prepare decoder input
        if tgt is None:
            # Inference: use last label_len from encoder + zeros
            label = src[:, -self.label_len:, :self.output_dim]
            zeros = torch.zeros(batch_size, self.decoder_steps, self.output_dim, device=src.device)
            tgt = torch.cat([label, zeros], dim=1)

        tgt_embedded = self.decoder_embedding(tgt)
        tgt_len = tgt.size(1)
        tgt_embedded = tgt_embedded + self.pos_encoding[:, :tgt_len, :].to(tgt.device)

        # Initial decomposition for decoder
        seasonal, trend = self.decomp(tgt_embedded)

        # Decoder
        for layer in self.decoder_layers:
            seasonal, layer_trend = layer(seasonal, memory_seasonal, memory_trend)
            trend = trend + layer_trend

        # Project to output
        seasonal_output = self.seasonal_proj(seasonal)
        trend_output = self.trend_proj(trend)

        # Combine seasonal and trend
        output = seasonal_output + trend_output

        # Return only the prediction part
        return output[:, -self.decoder_steps:, :]

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            # Prepare decoder input
            label = src[:, -self.label_len:, :self.output_dim]
            decoder_input = torch.cat([label, tgt], dim=1)

            if scaler is not None:
                with autocast(device_type='cuda' if 'cuda' in self.device else 'cpu'):
                    predictions = self(src, decoder_input)
                    loss = F.mse_loss(predictions, tgt)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = self(src, decoder_input)
                loss = F.mse_loss(predictions, tgt)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {'loss': total_loss / num_batches}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.eval()
        total_loss = 0
        total_mae = 0
        total_mse = 0
        total_mape = 0
        total_directional_accuracy = 0
        num_batches = 0

        for batch in dataloader:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            predictions = self(src, None)

            loss = F.mse_loss(predictions, tgt)
            mae = F.l1_loss(predictions, tgt)
            mse = F.mse_loss(predictions, tgt)
            mape = torch.mean(torch.abs((tgt - predictions) / (tgt + 1e-8))) * 100

            # Directional accuracy
            if tgt.size(1) > 1:
                target_direction = (tgt[:, 1:] > tgt[:, :-1]).float()
                pred_direction = (predictions[:, 1:] > predictions[:, :-1]).float()
                directional_accuracy = (target_direction == pred_direction).float().mean()
            else:
                directional_accuracy = torch.tensor(0.0)

            total_loss += loss.item()
            total_mae += mae.item()
            total_mse += mse.item()
            total_mape += mape.item()
            total_directional_accuracy += directional_accuracy.item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches,
            'mse': total_mse / num_batches,
            'rmse': math.sqrt(total_mse / num_batches),
            'mape': total_mape / num_batches,
            'directional_accuracy': total_directional_accuracy / num_batches,
        }

    @torch.no_grad()
    def predict(self, src: Tensor) -> Tensor:
        """Make predictions."""
        self.eval()
        return self(src, None)

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'd_model': self.d_model,
                'encoder_steps': self.encoder_steps,
                'decoder_steps': self.decoder_steps,
                'label_len': self.label_len,
            }
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'], device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint.get('optimizer_state_dict'), checkpoint.get('epoch', 0)
