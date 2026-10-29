"""
Temporal Fusion Transformer (TFT) for multi-horizon time series forecasting.

Based on: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
by Lim et al. (2021)

This implementation includes:
- Variable Selection Networks for feature importance
- Gated Residual Networks (GRN) for non-linear processing
- LSTM for local processing
- Multi-head attention for long-term dependencies
- Quantile regression for uncertainty estimation
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


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) for controlling information flow.

    GLU(x) = σ(Wx + b) ⊙ (Vx + c)
    where σ is sigmoid and ⊙ is element-wise multiplication.
    """

    def __init__(self, input_dim: int, output_dim: Optional[int] = None, dropout: float = 0.1):
        """
        Initialize GLU.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension (defaults to input_dim)
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim

        self.fc = nn.Linear(input_dim, self.output_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, ..., input_dim)

        Returns:
            Output tensor of shape (batch, ..., output_dim)
        """
        x = self.fc(x)
        x = self.dropout(x)
        gate, value = torch.chunk(x, 2, dim=-1)
        return torch.sigmoid(gate) * value


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) for flexible non-linear processing.

    Applies:
    1. Linear transformation
    2. ELU activation
    3. Linear transformation
    4. Gating + skip connection
    5. Layer normalization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize GRN.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (defaults to input_dim)
            context_dim: Optional context dimension for conditional processing
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.context_dim = context_dim

        # Primary pathway
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)

        # Context pathway (if applicable)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)

        # Gating and skip
        self.glu = GatedLinearUnit(self.output_dim, self.output_dim, dropout=dropout)

        if input_dim != self.output_dim:
            self.skip_fc = nn.Linear(input_dim, self.output_dim)
        else:
            self.skip_fc = None

        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, ..., input_dim)
            context: Optional context tensor of shape (batch, ..., context_dim)

        Returns:
            Output tensor of shape (batch, ..., output_dim)
        """
        # Skip connection
        if self.skip_fc is not None:
            skip = self.skip_fc(x)
        else:
            skip = x

        # Primary pathway
        hidden = self.elu(self.fc1(x))

        # Add context if provided
        if context is not None and self.context_dim is not None:
            hidden = hidden + self.context_fc(context)

        hidden = self.fc2(hidden)

        # Gating and residual
        gated = self.glu(hidden)
        output = self.layer_norm(skip + gated)

        return output


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network for feature importance and selection.

    Learns soft weights for each input variable, allowing the model to
    identify which features are most important for prediction.
    """

    def __init__(
        self,
        input_dim: int,
        num_vars: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        """
        Initialize Variable Selection Network.

        Args:
            input_dim: Dimension of each variable
            num_vars: Number of input variables
            hidden_dim: Hidden dimension for GRN
            dropout: Dropout rate
            context_dim: Optional context dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        # Flatten all variables
        self.flattened_grn = GatedResidualNetwork(
            input_dim=num_vars * input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_vars,
            context_dim=context_dim,
            dropout=dropout,
        )

        # Individual variable GRNs
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_vars)
        ])

    def forward(
        self,
        variables: Tensor,
        context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            variables: Input variables of shape (batch, time, num_vars, input_dim)
            context: Optional context of shape (batch, time, context_dim)

        Returns:
            Tuple of:
                - Weighted output: (batch, time, hidden_dim)
                - Variable weights: (batch, time, num_vars)
        """
        batch_size, time_steps, num_vars, input_dim = variables.shape

        # Flatten variables for weight calculation
        flattened = variables.reshape(batch_size, time_steps, -1)

        # Calculate variable weights
        weights = self.flattened_grn(flattened, context)
        weights = F.softmax(weights, dim=-1)  # (batch, time, num_vars)

        # Process each variable through its GRN
        processed_vars = []
        for i, grn in enumerate(self.variable_grns):
            var = variables[:, :, i, :]  # (batch, time, input_dim)
            processed = grn(var)  # (batch, time, hidden_dim)
            processed_vars.append(processed)

        # Stack and weight
        processed_vars = torch.stack(processed_vars, dim=2)  # (batch, time, num_vars, hidden_dim)
        weighted_vars = torch.sum(
            processed_vars * weights.unsqueeze(-1),
            dim=2
        )  # (batch, time, hidden_dim)

        return weighted_vars, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretability features.

    Standard multi-head attention with additional support for
    attention weight extraction and visualization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Store for interpretability

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            query: Query tensor (batch, time, d_model)
            key: Key tensor (batch, time, d_model)
            value: Value tensor (batch, time, d_model)
            mask: Optional attention mask (batch, time, time)

        Returns:
            Output tensor (batch, time, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Store attention weights for interpretability
        self.attention_weights = attention.detach()

        # Apply attention to values
        context = torch.matmul(attention, V)

        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_o(context)

        return output


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon time series forecasting.

    Architecture:
    1. Variable selection for static, known, and observed inputs
    2. LSTM encoder for local temporal processing
    3. Multi-head attention for long-term dependencies
    4. Position-wise feed-forward with gating
    5. Quantile outputs for uncertainty estimation
    """

    def __init__(
        self,
        # Input configuration
        num_static_vars: int = 0,
        num_known_vars: int = 0,
        num_observed_vars: int = 1,
        input_dim: int = 1,

        # Architecture configuration
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,

        # Forecasting configuration
        encoder_steps: int = 30,
        decoder_steps: int = 10,
        quantiles: List[float] = [0.1, 0.5, 0.9],

        # Training configuration
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize Temporal Fusion Transformer.

        Args:
            num_static_vars: Number of static covariates
            num_known_vars: Number of known future covariates
            num_observed_vars: Number of observed variables to predict
            input_dim: Dimension of each input variable
            hidden_dim: Hidden dimension throughout the network
            lstm_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            encoder_steps: Number of historical time steps
            decoder_steps: Number of future time steps to predict
            quantiles: Quantiles for uncertainty estimation
            learning_rate: Learning rate for training
            device: Device to run on ('cuda' or 'cpu')
        """
        super().__init__()

        self.num_static_vars = num_static_vars
        self.num_known_vars = num_known_vars
        self.num_observed_vars = num_observed_vars
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.quantiles = quantiles
        self.learning_rate = learning_rate
        self.device = device

        # Variable selection networks
        if num_static_vars > 0:
            self.static_selection = VariableSelectionNetwork(
                input_dim=input_dim,
                num_vars=num_static_vars,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            self.static_context_grn = GatedResidualNetwork(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )

        total_encoder_vars = num_known_vars + num_observed_vars
        if total_encoder_vars > 0:
            self.encoder_selection = VariableSelectionNetwork(
                input_dim=input_dim,
                num_vars=total_encoder_vars,
                hidden_dim=hidden_dim,
                dropout=dropout,
                context_dim=hidden_dim if num_static_vars > 0 else None,
            )

        if num_known_vars > 0:
            self.decoder_selection = VariableSelectionNetwork(
                input_dim=input_dim,
                num_vars=num_known_vars,
                hidden_dim=hidden_dim,
                dropout=dropout,
                context_dim=hidden_dim if num_static_vars > 0 else None,
            )

        # LSTM encoder-decoder
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        # Post-LSTM gating
        self.post_lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim, dropout=dropout)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)

        # Multi-head attention
        self.attention = InterpretableMultiHeadAttention(
            d_model=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.attention_gate = GatedLinearUnit(hidden_dim, hidden_dim, dropout=dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)

        # Position-wise feed-forward
        self.position_wise_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # Output layers for each quantile
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, num_observed_vars)
            for _ in range(len(quantiles))
        ])

        self.to(device)

    def forward(
        self,
        static_vars: Optional[Tensor] = None,
        encoder_known_vars: Optional[Tensor] = None,
        encoder_observed_vars: Optional[Tensor] = None,
        decoder_known_vars: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            static_vars: Static covariates (batch, num_static_vars, input_dim)
            encoder_known_vars: Known past covariates (batch, encoder_steps, num_known_vars, input_dim)
            encoder_observed_vars: Observed past values (batch, encoder_steps, num_observed_vars, input_dim)
            decoder_known_vars: Known future covariates (batch, decoder_steps, num_known_vars, input_dim)

        Returns:
            Dictionary containing:
                - predictions: Quantile predictions (batch, decoder_steps, num_observed_vars, num_quantiles)
                - attention_weights: Attention weights for interpretability
                - variable_weights: Variable selection weights
        """
        batch_size = (
            static_vars.size(0) if static_vars is not None
            else encoder_known_vars.size(0) if encoder_known_vars is not None
            else encoder_observed_vars.size(0)
        )

        # Static variable processing
        static_context = None
        static_weights = None
        if static_vars is not None and self.num_static_vars > 0:
            # Expand static vars to match encoder time steps
            static_expanded = static_vars.unsqueeze(1).expand(
                batch_size, self.encoder_steps, self.num_static_vars, self.input_dim
            )
            static_selected, static_weights = self.static_selection(static_expanded)
            static_context = self.static_context_grn(static_selected[:, 0, :])  # Use first timestep

        # Encoder variable selection
        encoder_vars = []
        if encoder_known_vars is not None:
            encoder_vars.append(encoder_known_vars)
        if encoder_observed_vars is not None:
            encoder_vars.append(encoder_observed_vars)

        if encoder_vars:
            encoder_vars = torch.cat(encoder_vars, dim=2)  # Concatenate along variable dimension
            encoder_selected, encoder_weights = self.encoder_selection(
                encoder_vars,
                static_context.unsqueeze(1).expand(-1, self.encoder_steps, -1) if static_context is not None else None
            )
        else:
            raise ValueError("Must provide at least encoder_known_vars or encoder_observed_vars")

        # Decoder variable selection (known future variables)
        if decoder_known_vars is not None and self.num_known_vars > 0:
            decoder_selected, decoder_weights = self.decoder_selection(
                decoder_known_vars,
                static_context.unsqueeze(1).expand(-1, self.decoder_steps, -1) if static_context is not None else None
            )
        else:
            # If no known future variables, use zeros
            decoder_selected = torch.zeros(
                batch_size, self.decoder_steps, self.hidden_dim,
                device=self.device
            )

        # LSTM processing
        lstm_encoder_output, (hidden, cell) = self.lstm_encoder(encoder_selected)
        lstm_decoder_output, _ = self.lstm_decoder(decoder_selected, (hidden, cell))

        # Gate and add skip connection
        lstm_output = torch.cat([lstm_encoder_output, lstm_decoder_output], dim=1)
        gated_lstm = self.post_lstm_gate(lstm_output)
        gated_lstm = self.post_lstm_norm(lstm_output + gated_lstm)

        # Multi-head attention
        attention_output = self.attention(
            query=gated_lstm,
            key=gated_lstm,
            value=gated_lstm,
        )
        attention_gated = self.attention_gate(attention_output)
        attention_output = self.attention_norm(gated_lstm + attention_gated)

        # Position-wise processing
        output = self.position_wise_grn(attention_output)

        # Extract decoder portion
        decoder_output = output[:, -self.decoder_steps:, :]

        # Generate quantile predictions
        predictions = []
        for output_layer in self.output_layers:
            pred = output_layer(decoder_output)  # (batch, decoder_steps, num_observed_vars)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=-1)  # (batch, decoder_steps, num_observed_vars, num_quantiles)

        return {
            'predictions': predictions,
            'attention_weights': self.attention.attention_weights,
            'encoder_variable_weights': encoder_weights if encoder_vars else None,
            'decoder_variable_weights': decoder_weights if decoder_known_vars is not None else None,
            'static_variable_weights': static_weights,
        }

    def quantile_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Compute quantile loss for training.

        Args:
            predictions: Predicted quantiles (batch, time, vars, num_quantiles)
            targets: Ground truth (batch, time, vars)

        Returns:
            Quantile loss (scalar)
        """
        targets = targets.unsqueeze(-1)  # (batch, time, vars, 1)
        errors = targets - predictions  # (batch, time, vars, num_quantiles)

        quantiles = torch.tensor(self.quantiles, device=predictions.device).view(1, 1, 1, -1)

        loss = torch.max(
            quantiles * errors,
            (quantiles - 1) * errors
        )

        return loss.mean()

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            scaler: GradScaler for mixed precision training

        Returns:
            Dictionary of training metrics
        """
        self.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()}

            # Forward pass with mixed precision
            if scaler is not None:
                with autocast(device_type='cuda' if 'cuda' in self.device else 'cpu'):
                    outputs = self(**{k: v for k, v in batch.items() if k != 'targets'})
                    loss = self.quantile_loss(outputs['predictions'], batch['targets'])

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self(**{k: v for k, v in batch.items() if k != 'targets'})
                loss = self.quantile_loss(outputs['predictions'], batch['targets'])
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.eval()
        total_loss = 0
        total_mae = 0
        total_mse = 0
        total_mape = 0
        total_directional_accuracy = 0
        num_batches = 0

        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()}

            outputs = self(**{k: v for k, v in batch.items() if k != 'targets'})
            targets = batch['targets']

            # Use median prediction (quantile 0.5)
            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            predictions = outputs['predictions'][..., median_idx]

            # Compute metrics
            loss = self.quantile_loss(outputs['predictions'], targets)
            mae = F.l1_loss(predictions, targets)
            mse = F.mse_loss(predictions, targets)

            # MAPE
            mape = torch.mean(torch.abs((targets - predictions) / (targets + 1e-8))) * 100

            # Directional accuracy (for next step prediction)
            if targets.size(1) > 1:
                target_direction = (targets[:, 1:] > targets[:, :-1]).float()
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
    def predict(
        self,
        static_vars: Optional[Tensor] = None,
        encoder_known_vars: Optional[Tensor] = None,
        encoder_observed_vars: Optional[Tensor] = None,
        decoder_known_vars: Optional[Tensor] = None,
        return_quantiles: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Make predictions.

        Args:
            static_vars: Static covariates
            encoder_known_vars: Known past covariates
            encoder_observed_vars: Observed past values
            decoder_known_vars: Known future covariates
            return_quantiles: If True, return all quantiles; else return median only

        Returns:
            If return_quantiles is True:
                Tuple of (median, lower_bound, upper_bound)
            Else:
                Median predictions
        """
        self.eval()

        outputs = self(
            static_vars=static_vars,
            encoder_known_vars=encoder_known_vars,
            encoder_observed_vars=encoder_observed_vars,
            decoder_known_vars=decoder_known_vars,
        )

        predictions = outputs['predictions']

        if return_quantiles:
            # Return median and bounds
            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            median = predictions[..., median_idx]
            lower = predictions[..., 0]  # Lowest quantile
            upper = predictions[..., -1]  # Highest quantile
            return median, lower, upper
        else:
            # Return median only
            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            return predictions[..., median_idx]

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer to save
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': {
                'num_static_vars': self.num_static_vars,
                'num_known_vars': self.num_known_vars,
                'num_observed_vars': self.num_observed_vars,
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'lstm_layers': self.lstm_layers,
                'num_heads': self.num_heads,
                'dropout': self.dropout,
                'encoder_steps': self.encoder_steps,
                'decoder_steps': self.decoder_steps,
                'quantiles': self.quantiles,
            }
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model on

        Returns:
            Tuple of (model, optimizer_state_dict, epoch)
        """
        checkpoint = torch.load(path, map_location=device)

        model = cls(**checkpoint['config'], device=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        epoch = checkpoint.get('epoch', 0)

        return model, optimizer_state, epoch
