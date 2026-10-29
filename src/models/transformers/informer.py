"""
Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

Based on: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
by Zhou et al. (2021)

Key innovations:
- ProbSparse self-attention (O(L log L) complexity)
- Self-attention distilling for dominant features
- Generative decoder for one-step prediction
- Efficient for long sequences (1000+ time steps)
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


class ProbSparseAttention(nn.Module):
    """
    ProbSparse Self-Attention mechanism.

    Selects top-u queries based on sparsity measurement to reduce
    computational complexity from O(L^2) to O(L log L).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        factor: int = 5,
        dropout: float = 0.1,
    ):
        """
        Initialize ProbSparse attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            factor: Sampling factor for queries
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

    def _prob_QK(
        self,
        Q: Tensor,
        K: Tensor,
        sample_k: int,
        n_top: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute top-u queries based on sparsity measurement.

        Args:
            Q: Queries (batch, num_heads, L_Q, d_k)
            K: Keys (batch, num_heads, L_K, d_k)
            sample_k: Number of keys to sample for measurement
            n_top: Number of top queries to select

        Returns:
            Tuple of (Q_reduced, M_top indices)
        """
        batch_size, num_heads, L_K, d_k = K.shape
        _, _, L_Q, _ = Q.shape

        # Sample keys
        K_expand = K.unsqueeze(-3).expand(batch_size, num_heads, L_Q, L_K, d_k)

        # Random sampling of keys
        index_sample = torch.randint(0, L_K, (L_Q, sample_k), device=K.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # Compute Q @ K^T for sampled keys
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2),
            K_sample.transpose(-2, -1)
        ).squeeze(-2)  # (batch, num_heads, L_Q, sample_k)

        # Sparsity measurement: max - mean
        M = Q_K_sample.max(dim=-1)[0] - Q_K_sample.mean(dim=-1)

        # Select top-u queries
        M_top = M.topk(n_top, dim=-1, sorted=False)[1]

        return M_top

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with ProbSparse attention.

        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, L_Q, _ = query.shape
        _, L_K, _ = key.shape

        # Linear projections and reshape
        Q = self.W_q(query).view(batch_size, L_Q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, L_K, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, L_K, self.num_heads, self.d_k).transpose(1, 2)

        # ProbSparse attention mechanism
        u = self.factor * math.ceil(math.log(L_K))  # c * ln(L_K)
        u = min(u, L_Q)  # Don't select more than available

        # Sample for sparsity measurement
        u_sample = self.factor * math.ceil(math.log(L_K))
        u_sample = min(u_sample, L_K)

        if u < L_Q:
            # Use ProbSparse attention
            M_top = self._prob_QK(Q, K, u_sample, u)

            # Gather top queries
            Q_reduced = torch.gather(
                Q,
                dim=2,
                index=M_top.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
            )  # (batch, num_heads, u, d_k)

            # Compute attention for top queries
            scores = torch.matmul(Q_reduced, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            if mask is not None:
                # Gather corresponding mask rows
                mask_reduced = torch.gather(
                    mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1),
                    dim=2,
                    index=M_top.unsqueeze(-1).expand(-1, -1, -1, L_K)
                )
                scores = scores.masked_fill(mask_reduced == 0, -1e9)

            attention = F.softmax(scores, dim=-1)
            attention = self.dropout(attention)

            # Apply attention to values
            context = torch.matmul(attention, V)  # (batch, num_heads, u, d_k)

            # Create output tensor and scatter results
            output = torch.zeros(
                batch_size, self.num_heads, L_Q, self.d_k,
                device=query.device, dtype=query.dtype
            )
            output = output.scatter(
                dim=2,
                index=M_top.unsqueeze(-1).expand(-1, -1, -1, self.d_k),
                src=context
            )

            # For non-top queries, use mean of values
            V_mean = V.mean(dim=2, keepdim=True).expand(-1, -1, L_Q, -1)
            output = output + V_mean * (output.sum(dim=-1, keepdim=True) == 0).float()

        else:
            # Standard attention for short sequences
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

            attention = F.softmax(scores, dim=-1)
            attention = self.dropout(attention)
            output = torch.matmul(attention, V)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, L_Q, self.d_model)
        output = self.W_o(output)

        return output


class Distilling(nn.Module):
    """
    Distilling operation for self-attention.

    Uses 1D convolution and max pooling to distill dominant features
    and reduce sequence length by half.
    """

    def __init__(self, d_model: int):
        """
        Initialize distilling layer.

        Args:
            d_model: Model dimension
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Distilled output (batch, seq_len//2, d_model)
        """
        # Transpose for conv1d: (batch, d_model, seq_len)
        x = x.transpose(1, 2)

        # Apply convolution and pooling
        x = self.conv(x)
        x = self.activation(x)
        x = self.maxpool(x)

        # Transpose back: (batch, seq_len//2, d_model)
        x = x.transpose(1, 2)
        x = self.norm(x)

        return x


class InformerEncoderLayer(nn.Module):
    """
    Single Informer encoder layer with ProbSparse attention and distilling.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        factor: int = 5,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_distilling: bool = True,
    ):
        """
        Initialize encoder layer.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            factor: Sampling factor for ProbSparse attention
            dropout: Dropout rate
            activation: Activation function
            use_distilling: Whether to use distilling
        """
        super().__init__()

        self.attention = ProbSparseAttention(
            d_model=d_model,
            num_heads=num_heads,
            factor=factor,
            dropout=dropout,
        )

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

        self.use_distilling = use_distilling
        if use_distilling:
            self.distilling = Distilling(d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len or seq_len//2, d_model)
        """
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        # Distilling
        if self.use_distilling:
            x = self.distilling(x)

        return x


class Informer(nn.Module):
    """
    Informer model for efficient long-sequence time series forecasting.

    Uses ProbSparse attention and distilling for O(L log L) complexity.
    """

    def __init__(
        self,
        # Input configuration
        input_dim: int = 1,
        output_dim: int = 1,

        # Architecture configuration
        d_model: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        num_heads: int = 8,
        d_ff: int = 512,
        factor: int = 5,
        dropout: float = 0.1,
        activation: str = 'gelu',

        # Forecasting configuration
        encoder_steps: int = 96,
        decoder_steps: int = 24,
        label_len: int = 48,  # Start token length

        # Training configuration
        learning_rate: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize Informer.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            d_model: Model dimension
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            factor: Sampling factor for ProbSparse attention
            dropout: Dropout rate
            activation: Activation function
            encoder_steps: Number of historical steps
            decoder_steps: Number of future steps to predict
            label_len: Length of start token (overlap with encoder)
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

        # Encoder layers (last layer without distilling)
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                factor=factor,
                dropout=dropout,
                activation=activation,
                use_distilling=(i < num_encoder_layers - 1),
            )
            for i in range(num_encoder_layers)
        ])

        # Decoder layers (ProbSparse self-attention + cross-attention)
        self.decoder_layers = nn.ModuleList([
            self._create_decoder_layer(
                d_model, num_heads, d_ff, factor, dropout, activation
            )
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)

        self.to(device)

    def _get_positional_encoding(self, max_len: int, d_model: int) -> Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _create_decoder_layer(
        self, d_model, num_heads, d_ff, factor, dropout, activation
    ):
        """Create a decoder layer with self-attention and cross-attention."""
        return nn.ModuleDict({
            'self_attn': ProbSparseAttention(d_model, num_heads, factor, dropout),
            'cross_attn': ProbSparseAttention(d_model, num_heads, factor, dropout),
            'ff': nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            ),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model),
            'norm3': nn.LayerNorm(d_model),
            'dropout1': nn.Dropout(dropout),
            'dropout2': nn.Dropout(dropout),
            'dropout3': nn.Dropout(dropout),
        })

    def generate_causal_mask(self, sz: int) -> Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(self.device)

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
                 For inference, provide label_len from end of src

        Returns:
            Predictions (batch, decoder_steps, output_dim)
        """
        batch_size = src.size(0)

        # Encode
        src_embedded = self.input_embedding(src)
        src_embedded = src_embedded + self.pos_encoding[:, :src.size(1), :].to(src.device)

        memory = src_embedded
        for layer in self.encoder_layers:
            memory = layer(memory)

        # Prepare decoder input
        if tgt is None:
            # Inference: use last label_len from encoder + zeros for prediction
            label = src[:, -self.label_len:, :self.output_dim]
            zeros = torch.zeros(batch_size, self.decoder_steps, self.output_dim, device=src.device)
            tgt = torch.cat([label, zeros], dim=1)

        tgt_embedded = self.decoder_embedding(tgt)
        tgt_len = tgt.size(1)
        tgt_embedded = tgt_embedded + self.pos_encoding[:, :tgt_len, :].to(tgt.device)

        # Decode
        decoder_output = tgt_embedded
        causal_mask = self.generate_causal_mask(tgt_len)

        for layer in self.decoder_layers:
            # Self-attention with causal mask
            self_attn_output = layer['self_attn'](
                decoder_output, decoder_output, decoder_output, causal_mask
            )
            decoder_output = decoder_output + layer['dropout1'](self_attn_output)
            decoder_output = layer['norm1'](decoder_output)

            # Cross-attention to encoder
            cross_attn_output = layer['cross_attn'](
                decoder_output, memory, memory, None
            )
            decoder_output = decoder_output + layer['dropout2'](cross_attn_output)
            decoder_output = layer['norm2'](decoder_output)

            # Feed-forward
            ff_output = layer['ff'](decoder_output)
            decoder_output = decoder_output + layer['dropout3'](ff_output)
            decoder_output = layer['norm3'](decoder_output)

        # Project to output
        output = self.output_proj(decoder_output)

        # Return only the prediction part (not the label part)
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

            # Prepare decoder input (label + target)
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

            predictions = self(src, None)  # Inference mode

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
