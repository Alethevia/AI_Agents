"""
Attention-based models and mechanisms for time series forecasting.

This module provides various attention mechanisms optimized for time series:
- Multi-head self-attention
- Temporal attention (time-aware)
- Spatial attention (for multivariate series)
- Cross-attention (for multivariate dependencies)
- Attention visualization utilities
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Implements the standard multi-head attention from "Attention is All You Need"
    with optimizations for time series data.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            qkv_bias: Whether to use bias in Q, K, V projections
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # For visualization
        self.attention_weights = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional attention mask (batch, seq_len, seq_len)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor (batch, seq_len, d_model)
            If return_attention is True, also returns attention weights
        """
        batch_size, seq_len, _ = query.size()

        # Linear projections and reshape for multi-head
        # (batch, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax and dropout
        attention = F.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)

        # Store for visualization
        self.attention_weights = attention.detach()

        # Apply attention to values
        context = torch.matmul(attention, V)

        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.W_o(context)
        output = self.dropout(output)

        if return_attention:
            return output, attention
        return output


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism that considers time differences.

    This attention mechanism incorporates temporal information by
    modulating attention scores based on time differences between elements.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_time_delta: int = 100,
        dropout: float = 0.1,
    ):
        """
        Initialize temporal attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            max_time_delta: Maximum time difference to consider
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_time_delta = max_time_delta

        # Standard attention components
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Temporal bias embeddings
        self.temporal_bias = nn.Embedding(max_time_delta * 2 + 1, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        time_indices: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            time_indices: Time indices (batch, seq_len) or None
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.size()

        # Linear projections
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add temporal bias
        if time_indices is not None:
            # Compute time differences
            time_diff = time_indices.unsqueeze(2) - time_indices.unsqueeze(1)  # (batch, seq_len, seq_len)
            time_diff = torch.clamp(time_diff, -self.max_time_delta, self.max_time_delta)
            time_diff = time_diff + self.max_time_delta  # Shift to positive indices

            # Get temporal bias
            temporal_bias = self.temporal_bias(time_diff)  # (batch, seq_len, seq_len, num_heads)
            temporal_bias = temporal_bias.permute(0, 3, 1, 2)  # (batch, num_heads, seq_len, seq_len)

            scores = scores + temporal_bias

        # Apply mask
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax and attention
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        self.attention_weights = attention.detach()

        # Apply to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output


class SpatialAttention(nn.Module):
    """
    Spatial attention for multivariate time series.

    Learns dependencies between different variables/assets at each time step.
    """

    def __init__(
        self,
        num_vars: int,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize spatial attention.

        Args:
            num_vars: Number of variables/assets
            d_model: Model dimension per variable
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        self.num_heads = num_heads

        # Attention over variables
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Per-variable embeddings
        self.var_embeddings = nn.Parameter(torch.randn(num_vars, d_model))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, num_vars, d_model)
            mask: Optional mask (batch, num_vars, num_vars)

        Returns:
            Output tensor (batch, seq_len, num_vars, d_model)
        """
        batch_size, seq_len, num_vars, d_model = x.size()

        # Process each time step
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch, num_vars, d_model)

            # Add variable embeddings
            x_t = x_t + self.var_embeddings.unsqueeze(0)

            # Apply attention over variables
            attended = self.attention(x_t, x_t, x_t, mask=mask)

            # Residual connection and layer norm
            output = self.layer_norm(x_t + self.dropout(attended))
            outputs.append(output)

        # Stack time steps
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, num_vars, d_model)

        return output


class CrossAttention(nn.Module):
    """
    Cross-attention for attending to different time series.

    Useful for multivariate forecasting where one series attends to others.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(
        self,
        target: Tensor,
        source: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            target: Target series (batch, target_len, d_model)
            source: Source series to attend to (batch, source_len, d_model)
            mask: Optional attention mask (batch, target_len, source_len)

        Returns:
            Output tensor (batch, target_len, d_model)
        """
        batch_size, target_len, _ = target.size()
        source_len = source.size(1)

        # Query from target, key and value from source
        Q = self.W_q(target).view(batch_size, target_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(source).view(batch_size, source_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(source).view(batch_size, source_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        self.attention_weights = attention.detach()

        # Apply attention
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, target_len, self.d_model
        )
        output = self.W_o(context)

        return output


class AttentionVisualizer:
    """
    Utilities for visualizing attention weights.

    Provides methods to visualize and analyze attention patterns
    for model interpretability.
    """

    @staticmethod
    def plot_attention_heatmap(
        attention_weights: Tensor,
        title: str = "Attention Weights",
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'viridis',
    ) -> plt.Figure:
        """
        Plot attention weights as a heatmap.

        Args:
            attention_weights: Attention weights (seq_len, seq_len) or (num_heads, seq_len, seq_len)
            title: Plot title
            x_labels: Labels for x-axis (key positions)
            y_labels: Labels for y-axis (query positions)
            figsize: Figure size
            cmap: Colormap

        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        if isinstance(attention_weights, Tensor):
            attention_weights = attention_weights.cpu().numpy()

        # Handle multi-head case
        if attention_weights.ndim == 3:
            num_heads = attention_weights.shape[0]
            fig, axes = plt.subplots(1, num_heads, figsize=(figsize[0] * num_heads // 2, figsize[1]))
            if num_heads == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                sns.heatmap(
                    attention_weights[i],
                    ax=ax,
                    cmap=cmap,
                    xticklabels=x_labels or 'auto',
                    yticklabels=y_labels or 'auto',
                    cbar=True,
                )
                ax.set_title(f"{title} - Head {i+1}")
                ax.set_xlabel("Key Position")
                ax.set_ylabel("Query Position")
        else:
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                attention_weights,
                ax=ax,
                cmap=cmap,
                xticklabels=x_labels or 'auto',
                yticklabels=y_labels or 'auto',
                cbar=True,
            )
            ax.set_title(title)
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_temporal_attention_pattern(
        attention_weights: Tensor,
        time_steps: Optional[np.ndarray] = None,
        query_idx: int = -1,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot how a specific query attends to different time steps.

        Args:
            attention_weights: Attention weights (seq_len, seq_len)
            time_steps: Time step labels
            query_idx: Index of query to visualize (-1 for last)
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if isinstance(attention_weights, Tensor):
            attention_weights = attention_weights.cpu().numpy()

        if attention_weights.ndim == 3:
            # Average over heads
            attention_weights = attention_weights.mean(axis=0)

        query_attention = attention_weights[query_idx]

        if time_steps is None:
            time_steps = np.arange(len(query_attention))

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(time_steps, query_attention, alpha=0.7)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"Attention Pattern for Query Position {query_idx}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def analyze_attention_statistics(
        attention_weights: Tensor,
    ) -> Dict[str, float]:
        """
        Compute statistics about attention patterns.

        Args:
            attention_weights: Attention weights (num_heads, seq_len, seq_len)

        Returns:
            Dictionary of statistics
        """
        if isinstance(attention_weights, Tensor):
            attention_weights = attention_weights.cpu().numpy()

        stats = {}

        # Average attention per head
        if attention_weights.ndim == 3:
            stats['attention_per_head'] = attention_weights.mean(axis=(1, 2)).tolist()

            # Attention entropy per head (measure of concentration)
            epsilon = 1e-10
            entropy = -np.sum(
                attention_weights * np.log(attention_weights + epsilon),
                axis=-1
            )
            stats['entropy_per_head'] = entropy.mean(axis=1).tolist()

            # Average over heads for remaining stats
            attention_weights = attention_weights.mean(axis=0)

        # Self-attention ratio (diagonal elements)
        self_attention = np.diag(attention_weights)
        stats['mean_self_attention'] = float(self_attention.mean())
        stats['std_self_attention'] = float(self_attention.std())

        # Attention span (weighted average distance)
        seq_len = attention_weights.shape[0]
        positions = np.arange(seq_len)
        attention_span = []
        for i in range(seq_len):
            span = np.sum(attention_weights[i] * np.abs(positions - i))
            attention_span.append(span)
        stats['mean_attention_span'] = float(np.mean(attention_span))
        stats['max_attention_span'] = float(np.max(attention_span))

        # Sparsity (percentage of attention weights below threshold)
        threshold = 0.01
        sparsity = (attention_weights < threshold).mean()
        stats['sparsity'] = float(sparsity)

        return stats

    @staticmethod
    def plot_attention_rollout(
        attention_layers: List[Tensor],
        figsize: Tuple[int, int] = (12, 4),
    ) -> plt.Figure:
        """
        Visualize attention rollout across multiple layers.

        Args:
            attention_layers: List of attention weights from different layers
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        num_layers = len(attention_layers)

        # Convert to numpy and average over heads
        attentions = []
        for attn in attention_layers:
            if isinstance(attn, Tensor):
                attn = attn.cpu().numpy()
            if attn.ndim == 3:
                attn = attn.mean(axis=0)
            attentions.append(attn)

        # Compute rollout (cumulative attention)
        rollout = attentions[0]
        rollouts = [rollout]

        for attn in attentions[1:]:
            rollout = np.matmul(attn, rollout)
            rollouts.append(rollout)

        # Plot
        fig, axes = plt.subplots(1, num_layers, figsize=figsize)
        if num_layers == 1:
            axes = [axes]

        for i, (ax, rollout) in enumerate(zip(axes, rollouts)):
            sns.heatmap(rollout, ax=ax, cmap='viridis', cbar=True)
            ax.set_title(f"Layer {i+1} Rollout")
            ax.set_xlabel("Key Position")
            if i == 0:
                ax.set_ylabel("Query Position")

        plt.tight_layout()
        return fig

    @staticmethod
    def get_top_attended_positions(
        attention_weights: Tensor,
        query_idx: int,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Get top-k most attended positions for a query.

        Args:
            attention_weights: Attention weights (seq_len, seq_len)
            query_idx: Query index
            top_k: Number of top positions to return

        Returns:
            List of (position, weight) tuples
        """
        if isinstance(attention_weights, Tensor):
            attention_weights = attention_weights.cpu().numpy()

        if attention_weights.ndim == 3:
            # Average over heads
            attention_weights = attention_weights.mean(axis=0)

        query_attention = attention_weights[query_idx]
        top_indices = np.argsort(query_attention)[-top_k:][::-1]

        return [(int(idx), float(query_attention[idx])) for idx in top_indices]
