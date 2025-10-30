"""
Transformer-based models for time series forecasting.

This module provides state-of-the-art transformer architectures optimized for
financial time series prediction, including:
- Temporal Fusion Transformer (TFT)
- Informer (efficient long-sequence transformer)
- Autoformer (auto-correlation based transformer)
- Time Series Transformer (classical transformer adapted for time series)
- Various attention mechanisms for multivariate analysis
"""

from .temporal_fusion_transformer import (
    TemporalFusionTransformer,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    GatedLinearUnit,
)

from .attention_models import (
    MultiHeadAttention,
    TemporalAttention,
    SpatialAttention,
    CrossAttention,
    AttentionVisualizer,
)

from .time_series_transformer import (
    TimeSeriesTransformer,
    PositionalEncoding,
    TransformerEncoder,
    TransformerDecoder,
)

from .informer import (
    Informer,
    ProbSparseAttention,
    Distilling,
)

from .autoformer import (
    Autoformer,
    AutoCorrelation,
    SeriesDecomposition,
)

__all__ = [
    # Temporal Fusion Transformer
    'TemporalFusionTransformer',
    'GatedResidualNetwork',
    'VariableSelectionNetwork',
    'GatedLinearUnit',

    # Attention Models
    'MultiHeadAttention',
    'TemporalAttention',
    'SpatialAttention',
    'CrossAttention',
    'AttentionVisualizer',

    # Time Series Transformer
    'TimeSeriesTransformer',
    'PositionalEncoding',
    'TransformerEncoder',
    'TransformerDecoder',

    # Informer
    'Informer',
    'ProbSparseAttention',
    'Distilling',

    # Autoformer
    'Autoformer',
    'AutoCorrelation',
    'SeriesDecomposition',
]

__version__ = '1.0.0'
