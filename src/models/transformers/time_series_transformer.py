"""
Full Transformer architecture for time series forecasting.

This module implements a classical Transformer adapted for time series with:
- Time-aware positional encoding
- Encoder-decoder architecture
- Causal masking for autoregressive prediction
- Multi-horizon forecasting
- Uncertainty quantification
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


class PositionalEncoding(nn.Module):
    """
    Positional encoding for time series.

    Supports both:
    1. Sinusoidal encoding (original Transformer)
    2. Learned encoding
    3. Time-feature encoding (hour, day, month, etc.)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        encoding_type: str = 'sinusoidal',
    ):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
            encoding_type: Type of encoding ('sinusoidal', 'learned', or 'time')
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.encoding_type = encoding_type

        if encoding_type == 'sinusoidal':
            # Create sinusoidal positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)

        elif encoding_type == 'learned':
            # Learned positional embedding
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

        elif encoding_type == 'time':
            # Time features will be added externally
            self.pe = None

    def forward(self, x: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            positions: Optional position indices (batch, seq_len)

        Returns:
            Output with positional encoding (batch, seq_len, d_model)
        """
        if self.encoding_type == 'time':
            # Assume time features are already in x
            return self.dropout(x)

        seq_len = x.size(1)

        if positions is not None:
            # Use specific positions
            pe = self.pe[:, positions, :]
        else:
            # Use sequential positions
            pe = self.pe[:, :seq_len, :]

        x = x + pe
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.

    Consists of:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Residual connections and layer normalization
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        """
        Initialize encoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu')
        """
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder stack.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        """
        Initialize encoder.

        Args:
            num_layers: Number of encoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through all encoder layers.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Encoded output (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer with causal masking.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        """
        Initialize decoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()

        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention to encoder
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Decoder input (batch, tgt_len, d_model)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Causal mask for decoder
            memory_mask: Mask for encoder memory

        Returns:
            Decoder output (batch, tgt_len, d_model)
        """
        # Causal self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=False)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Cross-attention to encoder
        attn_output, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask, need_weights=False)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        # Feed-forward
        ff_output = self.ff(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)

        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder stack.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        """
        Initialize decoder.

        Args:
            num_layers: Number of decoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through all decoder layers.

        Args:
            x: Decoder input (batch, tgt_len, d_model)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Causal mask for decoder
            memory_mask: Mask for encoder memory

        Returns:
            Decoded output (batch, tgt_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x


class TimeSeriesTransformer(nn.Module):
    """
    Full Transformer for time series forecasting.

    Encoder-decoder architecture with:
    - Positional encoding
    - Multi-head attention
    - Causal masking for autoregressive prediction
    - Multi-horizon forecasting
    - Uncertainty quantification via ensemble or quantile regression
    """

    def __init__(
        self,
        # Input configuration
        input_dim: int = 1,
        output_dim: int = 1,

        # Architecture configuration
        d_model: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu',

        # Positional encoding
        max_len: int = 5000,
        encoding_type: str = 'sinusoidal',

        # Forecasting configuration
        encoder_steps: int = 30,
        decoder_steps: int = 10,

        # Uncertainty quantification
        use_quantiles: bool = False,
        quantiles: List[float] = [0.1, 0.5, 0.9],

        # Training configuration
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize Time Series Transformer.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            d_model: Model dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            activation: Activation function
            max_len: Maximum sequence length
            encoding_type: Type of positional encoding
            encoder_steps: Number of historical steps
            decoder_steps: Number of future steps to predict
            use_quantiles: Whether to use quantile regression
            quantiles: Quantiles for uncertainty estimation
            learning_rate: Learning rate for training
            device: Device to run on
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.use_quantiles = use_quantiles
        self.quantiles = quantiles
        self.learning_rate = learning_rate
        self.device = device

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.output_embedding = nn.Linear(output_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            encoding_type=encoding_type,
        )
        self.pos_decoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            encoding_type=encoding_type,
        )

        # Encoder
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

        # Decoder
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

        # Output projection
        if use_quantiles:
            self.output_proj = nn.ModuleList([
                nn.Linear(d_model, output_dim)
                for _ in range(len(quantiles))
            ])
        else:
            self.output_proj = nn.Linear(d_model, output_dim)

        self.to(device)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generate causal mask for decoder.

        Args:
            sz: Size of mask

        Returns:
            Causal mask (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(self.device)

    def forward(
        self,
        src: Tensor,
        tgt: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            src: Source sequence (batch, encoder_steps, input_dim)
            tgt: Target sequence for teacher forcing (batch, decoder_steps, output_dim)
                 If None, uses autoregressive generation
            src_mask: Optional source mask
            tgt_mask: Optional target mask (will be generated if None)

        Returns:
            Predictions (batch, decoder_steps, output_dim) or
                       (batch, decoder_steps, output_dim, num_quantiles)
        """
        batch_size = src.size(0)

        # Encode source
        src_embedded = self.input_embedding(src)
        src_embedded = self.pos_encoder(src_embedded)
        memory = self.encoder(src_embedded, src_mask)

        # Prepare target
        if tgt is None:
            # Autoregressive generation
            # Start with last value from encoder or zeros
            tgt = src[:, -1:, :self.output_dim]  # (batch, 1, output_dim)

            # Generate causal mask
            tgt_mask = self.generate_square_subsequent_mask(self.decoder_steps)

            outputs = []
            for i in range(self.decoder_steps):
                # Embed target
                tgt_embedded = self.output_embedding(tgt)
                tgt_embedded = self.pos_decoder(tgt_embedded)

                # Decode
                current_mask = tgt_mask[:i+1, :i+1] if tgt_mask is not None else None
                decoder_output = self.decoder(tgt_embedded, memory, current_mask)

                # Project to output
                if self.use_quantiles:
                    output = torch.stack([
                        proj(decoder_output[:, -1:, :])
                        for proj in self.output_proj
                    ], dim=-1)  # (batch, 1, output_dim, num_quantiles)
                else:
                    output = self.output_proj(decoder_output[:, -1:, :])  # (batch, 1, output_dim)

                outputs.append(output)

                # Update target for next step (use median if quantiles)
                if self.use_quantiles:
                    median_idx = len(self.quantiles) // 2
                    next_input = output[..., median_idx]
                else:
                    next_input = output

                tgt = torch.cat([tgt, next_input], dim=1)

            # Stack outputs
            output = torch.cat(outputs, dim=1)  # (batch, decoder_steps, output_dim[, num_quantiles])

        else:
            # Teacher forcing
            tgt_embedded = self.output_embedding(tgt)
            tgt_embedded = self.pos_decoder(tgt_embedded)

            # Generate causal mask if not provided
            if tgt_mask is None:
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

            # Decode
            decoder_output = self.decoder(tgt_embedded, memory, tgt_mask)

            # Project to output
            if self.use_quantiles:
                output = torch.stack([
                    proj(decoder_output)
                    for proj in self.output_proj
                ], dim=-1)  # (batch, decoder_steps, output_dim, num_quantiles)
            else:
                output = self.output_proj(decoder_output)  # (batch, decoder_steps, output_dim)

        return output

    def quantile_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute quantile loss.

        Args:
            predictions: Predicted quantiles (batch, time, dim, num_quantiles)
            targets: Ground truth (batch, time, dim)

        Returns:
            Quantile loss
        """
        targets = targets.unsqueeze(-1)  # (batch, time, dim, 1)
        errors = targets - predictions

        quantiles = torch.tensor(self.quantiles, device=predictions.device).view(1, 1, 1, -1)

        loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
        return loss.mean()

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            scaler: GradScaler for mixed precision
            teacher_forcing_ratio: Probability of using teacher forcing (1.0 = always)

        Returns:
            Training metrics
        """
        self.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            # Move to device
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)

            # Decide whether to use teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

            # Forward pass
            if scaler is not None:
                with autocast(device_type='cuda' if 'cuda' in self.device else 'cpu'):
                    predictions = self(src, tgt if use_teacher_forcing else None)

                    if self.use_quantiles:
                        loss = self.quantile_loss(predictions, tgt)
                    else:
                        loss = F.mse_loss(predictions, tgt)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = self(src, tgt if use_teacher_forcing else None)

                if self.use_quantiles:
                    loss = self.quantile_loss(predictions, tgt)
                else:
                    loss = F.mse_loss(predictions, tgt)

                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {'loss': total_loss / num_batches}

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            dataloader: Validation data loader

        Returns:
            Validation metrics
        """
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

            predictions = self(src, None)  # Autoregressive generation

            # Get point predictions
            if self.use_quantiles:
                median_idx = len(self.quantiles) // 2
                point_predictions = predictions[..., median_idx]
                loss = self.quantile_loss(predictions, tgt)
            else:
                point_predictions = predictions
                loss = F.mse_loss(predictions, tgt)

            # Metrics
            mae = F.l1_loss(point_predictions, tgt)
            mse = F.mse_loss(point_predictions, tgt)
            mape = torch.mean(torch.abs((tgt - point_predictions) / (tgt + 1e-8))) * 100

            # Directional accuracy
            if tgt.size(1) > 1:
                target_direction = (tgt[:, 1:] > tgt[:, :-1]).float()
                pred_direction = (point_predictions[:, 1:] > point_predictions[:, :-1]).float()
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
    def predict(
        self,
        src: Tensor,
        return_quantiles: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Make predictions.

        Args:
            src: Source sequence (batch, encoder_steps, input_dim)
            return_quantiles: Whether to return quantile bounds

        Returns:
            Predictions or (median, lower, upper) if return_quantiles
        """
        self.eval()
        predictions = self(src, None)

        if self.use_quantiles and return_quantiles:
            median_idx = len(self.quantiles) // 2
            median = predictions[..., median_idx]
            lower = predictions[..., 0]
            upper = predictions[..., -1]
            return median, lower, upper
        elif self.use_quantiles:
            median_idx = len(self.quantiles) // 2
            return predictions[..., median_idx]
        else:
            return predictions

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'encoder_steps': self.encoder_steps,
                'decoder_steps': self.decoder_steps,
                'use_quantiles': self.use_quantiles,
                'quantiles': self.quantiles,
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
