"""
Features Package

Comprehensive feature engineering module for AI trading system.

This package provides:
- Technical indicators extraction
- Fundamental features calculation
- Market microstructure analysis
- Sentiment features from multiple sources
- Feature transformation utilities
- Advanced feature selection methods

All modules are production-ready with:
- Vectorized operations for performance
- Batch and streaming computation support
- Missing data handling
- Multi-symbol support
- Comprehensive documentation
"""

from .extractors import (
    TechnicalIndicators,
    FundamentalFeatures,
    MarketMicrostructure,
    SentimentFeatures
)

from .transformers import FeatureTransformer
from .selectors import FeatureSelector

__all__ = [
    # Extractors
    'TechnicalIndicators',
    'FundamentalFeatures',
    'MarketMicrostructure',
    'SentimentFeatures',

    # Transformers
    'FeatureTransformer',

    # Selectors
    'FeatureSelector',
]

__version__ = '1.0.0'
__author__ = 'AI Trading System'
