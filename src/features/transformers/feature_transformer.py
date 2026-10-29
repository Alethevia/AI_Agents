"""
Feature Transformer

This module provides advanced feature transformation methods including
polynomial features, interactions, rank transformations, and cross-sectional features.

Author: AI Trading System
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures as SKLearnPoly
import warnings


class FeatureTransformer:
    """
    Production-ready feature transformation utilities.

    Supports:
    - Polynomial feature generation
    - Feature interactions
    - Rank transformations
    - Percentile features
    - Cross-sectional features
    - Time-based features
    """

    def __init__(self):
        """Initialize feature transformer."""
        self.feature_names_ = []
        self.fitted_ = False

    # ==================== Polynomial Features ====================

    @staticmethod
    def polynomial_features(data: pd.DataFrame, degree: int = 2,
                          include_bias: bool = False,
                          interaction_only: bool = False) -> pd.DataFrame:
        """
        Generate polynomial features.

        Args:
            data: Input features DataFrame
            degree: Polynomial degree (default: 2)
            include_bias: Include bias term (default: False)
            interaction_only: Only interaction terms, no powers (default: False)

        Returns:
            DataFrame with polynomial features
        """
        poly = SKLearnPoly(degree=degree, include_bias=include_bias,
                          interaction_only=interaction_only)

        # Transform data
        poly_features = poly.fit_transform(data)

        # Generate feature names
        feature_names = poly.get_feature_names_out(data.columns)

        # Create DataFrame
        df_poly = pd.DataFrame(poly_features, index=data.index, columns=feature_names)

        return df_poly

    @staticmethod
    def power_transform(data: Union[pd.Series, pd.DataFrame],
                       power: float = 2.0) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply power transformation to features.

        Args:
            data: Input data
            power: Power for transformation

        Returns:
            Power-transformed data
        """
        return np.sign(data) * np.abs(data) ** power

    @staticmethod
    def log_transform(data: Union[pd.Series, pd.DataFrame],
                     offset: float = 1.0) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply logarithmic transformation.

        Args:
            data: Input data
            offset: Offset to handle zero/negative values (default: 1.0)

        Returns:
            Log-transformed data
        """
        return np.log(np.abs(data) + offset)

    @staticmethod
    def square_root_transform(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply square root transformation.

        Args:
            data: Input data

        Returns:
            Square root transformed data
        """
        return np.sign(data) * np.sqrt(np.abs(data))

    # ==================== Interaction Features ====================

    @staticmethod
    def multiplicative_interactions(data: pd.DataFrame,
                                   feature_pairs: Optional[List[Tuple[str, str]]] = None,
                                   max_features: Optional[int] = None) -> pd.DataFrame:
        """
        Create multiplicative interaction features.

        Args:
            data: Input features DataFrame
            feature_pairs: Specific pairs to interact (None = all pairs)
            max_features: Maximum number of interaction features

        Returns:
            DataFrame with interaction features
        """
        result = pd.DataFrame(index=data.index)

        if feature_pairs is None:
            # Generate all pairwise combinations
            feature_pairs = list(combinations(data.columns, 2))

        # Limit number of features if specified
        if max_features is not None:
            feature_pairs = feature_pairs[:max_features]

        for feat1, feat2 in feature_pairs:
            interaction_name = f'{feat1}_x_{feat2}'
            result[interaction_name] = data[feat1] * data[feat2]

        return result

    @staticmethod
    def additive_interactions(data: pd.DataFrame,
                            feature_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Create additive interaction features.

        Args:
            data: Input features DataFrame
            feature_pairs: Specific pairs to interact (None = all pairs)

        Returns:
            DataFrame with additive interactions
        """
        result = pd.DataFrame(index=data.index)

        if feature_pairs is None:
            feature_pairs = list(combinations(data.columns, 2))

        for feat1, feat2 in feature_pairs:
            result[f'{feat1}_plus_{feat2}'] = data[feat1] + data[feat2]
            result[f'{feat1}_minus_{feat2}'] = data[feat1] - data[feat2]

        return result

    @staticmethod
    def ratio_features(data: pd.DataFrame,
                      feature_pairs: Optional[List[Tuple[str, str]]] = None,
                      epsilon: float = 1e-10) -> pd.DataFrame:
        """
        Create ratio features.

        Args:
            data: Input features DataFrame
            feature_pairs: Specific pairs to create ratios (None = all pairs)
            epsilon: Small constant to avoid division by zero

        Returns:
            DataFrame with ratio features
        """
        result = pd.DataFrame(index=data.index)

        if feature_pairs is None:
            feature_pairs = list(combinations(data.columns, 2))

        for feat1, feat2 in feature_pairs:
            result[f'{feat1}_div_{feat2}'] = data[feat1] / (data[feat2] + epsilon)

        return result

    @staticmethod
    def conditional_features(data: pd.DataFrame, condition_feature: str,
                           threshold: float) -> pd.DataFrame:
        """
        Create conditional features based on a threshold.

        Args:
            data: Input features DataFrame
            condition_feature: Feature to use for conditioning
            threshold: Threshold value

        Returns:
            DataFrame with conditional features
        """
        result = pd.DataFrame(index=data.index)

        condition = data[condition_feature] > threshold

        for col in data.columns:
            if col != condition_feature:
                result[f'{col}_when_{condition_feature}_high'] = data[col] * condition
                result[f'{col}_when_{condition_feature}_low'] = data[col] * (~condition)

        return result

    # ==================== Rank Transformation ====================

    @staticmethod
    def rank_transform(data: Union[pd.Series, pd.DataFrame],
                      method: str = 'average', pct: bool = True) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply rank transformation to features.

        Args:
            data: Input data
            method: Ranking method ('average', 'min', 'max', 'first', 'dense')
            pct: Return percentile ranks (0-1) instead of ranks

        Returns:
            Rank-transformed data
        """
        if isinstance(data, pd.DataFrame):
            return data.rank(method=method, pct=pct)
        else:
            return data.rank(method=method, pct=pct)

    @staticmethod
    def rolling_rank(data: Union[pd.Series, pd.DataFrame],
                    window: int = 20, pct: bool = True) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply rolling rank transformation.

        Args:
            data: Input data
            window: Rolling window size
            pct: Return percentile ranks

        Returns:
            Rolling rank-transformed data
        """
        def rank_within_window(x):
            return pd.Series(x).rank(pct=pct).iloc[-1]

        if isinstance(data, pd.DataFrame):
            return data.rolling(window=window, min_periods=1).apply(rank_within_window, raw=True)
        else:
            return data.rolling(window=window, min_periods=1).apply(rank_within_window, raw=True)

    @staticmethod
    def gaussian_rank(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply Gaussian rank transformation (inverse normal transformation).

        Args:
            data: Input data

        Returns:
            Gaussian rank-transformed data
        """
        from scipy.stats import norm

        # Get percentile ranks
        ranks = FeatureTransformer.rank_transform(data, pct=True)

        # Apply inverse normal CDF
        # Avoid exact 0 and 1 values
        ranks_clipped = ranks.clip(0.001, 0.999)

        if isinstance(ranks_clipped, pd.DataFrame):
            return ranks_clipped.apply(lambda x: norm.ppf(x))
        else:
            return norm.ppf(ranks_clipped)

    # ==================== Percentile Features ====================

    @staticmethod
    def percentile_features(data: Union[pd.Series, pd.DataFrame],
                          window: int = 252,
                          percentiles: List[float] = [0.25, 0.5, 0.75]) -> pd.DataFrame:
        """
        Calculate percentile-based features.

        Args:
            data: Input data
            window: Rolling window for percentile calculation
            percentiles: List of percentiles to calculate (0-1)

        Returns:
            DataFrame with percentile features
        """
        result = pd.DataFrame(index=data.index)

        if isinstance(data, pd.Series):
            data_df = data.to_frame()
        else:
            data_df = data

        for col in data_df.columns:
            for pct in percentiles:
                pct_name = f'{col}_pct_{int(pct*100)}'
                result[pct_name] = data_df[col].rolling(window=window, min_periods=1).quantile(pct)

                # Current value vs percentile
                current_vs_pct = data_df[col] - result[pct_name]
                result[f'{col}_vs_pct_{int(pct*100)}'] = current_vs_pct

        return result

    @staticmethod
    def percentile_rank(data: Union[pd.Series, pd.DataFrame],
                       window: int = 252) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate percentile rank within rolling window.

        Args:
            data: Input data
            window: Rolling window

        Returns:
            Percentile rank (0-100)
        """
        def calc_percentile(x):
            if len(x) < 2:
                return 50
            return (x < x[-1]).sum() / len(x) * 100

        if isinstance(data, pd.DataFrame):
            return data.rolling(window=window, min_periods=1).apply(calc_percentile, raw=True)
        else:
            return data.rolling(window=window, min_periods=1).apply(calc_percentile, raw=True)

    # ==================== Cross-Sectional Features ====================

    @staticmethod
    def cross_sectional_rank(data: pd.DataFrame, group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate cross-sectional rank across symbols.

        Args:
            data: DataFrame with features (rows=time, cols=features or multi-index)
            group_by: Optional grouping column (e.g., 'sector')

        Returns:
            DataFrame with cross-sectional ranks
        """
        if group_by is not None and group_by in data.columns:
            # Rank within groups
            ranked = data.groupby(group_by).rank(pct=True)
        else:
            # Rank across all symbols
            ranked = data.rank(axis=1, pct=True)

        return ranked

    @staticmethod
    def cross_sectional_zscore(data: pd.DataFrame, group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate cross-sectional z-scores.

        Args:
            data: DataFrame with features
            group_by: Optional grouping column

        Returns:
            DataFrame with z-scores
        """
        if group_by is not None and group_by in data.columns:
            # Z-score within groups
            mean = data.groupby(group_by).transform('mean')
            std = data.groupby(group_by).transform('std')
        else:
            # Z-score across all
            mean = data.mean(axis=1)
            std = data.std(axis=1)

        zscore = (data.subtract(mean, axis=0)) / (std.replace(0, 1))
        return zscore.fillna(0)

    @staticmethod
    def cross_sectional_demean(data: pd.DataFrame, group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Remove cross-sectional mean (neutralize market effect).

        Args:
            data: DataFrame with features
            group_by: Optional grouping column

        Returns:
            DataFrame with demeaned features
        """
        if group_by is not None and group_by in data.columns:
            mean = data.groupby(group_by).transform('mean')
        else:
            mean = data.mean(axis=1)

        demeaned = data.subtract(mean, axis=0)
        return demeaned

    @staticmethod
    def market_neutral_features(data: pd.DataFrame, market_proxy: pd.Series,
                               window: int = 60) -> pd.DataFrame:
        """
        Create market-neutral features by removing market beta.

        Args:
            data: DataFrame with features
            market_proxy: Market return proxy
            window: Window for beta calculation

        Returns:
            DataFrame with market-neutral features
        """
        result = pd.DataFrame(index=data.index)

        for col in data.columns:
            # Calculate rolling beta
            covariance = data[col].rolling(window=window, min_periods=1).cov(market_proxy)
            market_variance = market_proxy.rolling(window=window, min_periods=1).var()
            beta = covariance / (market_variance + 1e-10)

            # Remove market component
            market_component = beta * market_proxy
            result[f'{col}_neutral'] = data[col] - market_component

        return result

    # ==================== Time-Based Features ====================

    @staticmethod
    def lagged_features(data: Union[pd.Series, pd.DataFrame],
                       lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged features.

        Args:
            data: Input data
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        result = pd.DataFrame(index=data.index)

        if isinstance(data, pd.Series):
            for lag in lags:
                result[f'lag_{lag}'] = data.shift(lag)
        else:
            for col in data.columns:
                for lag in lags:
                    result[f'{col}_lag_{lag}'] = data[col].shift(lag)

        return result

    @staticmethod
    def lead_features(data: Union[pd.Series, pd.DataFrame],
                     leads: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Create lead (forward-looking) features.

        Args:
            data: Input data
            leads: List of lead periods

        Returns:
            DataFrame with lead features
        """
        result = pd.DataFrame(index=data.index)

        if isinstance(data, pd.Series):
            for lead in leads:
                result[f'lead_{lead}'] = data.shift(-lead)
        else:
            for col in data.columns:
                for lead in leads:
                    result[f'{col}_lead_{lead}'] = data[col].shift(-lead)

        return result

    @staticmethod
    def difference_features(data: Union[pd.Series, pd.DataFrame],
                          periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Create difference features.

        Args:
            data: Input data
            periods: List of difference periods

        Returns:
            DataFrame with difference features
        """
        result = pd.DataFrame(index=data.index)

        if isinstance(data, pd.Series):
            for period in periods:
                result[f'diff_{period}'] = data.diff(period)
                result[f'pct_change_{period}'] = data.pct_change(period)
        else:
            for col in data.columns:
                for period in periods:
                    result[f'{col}_diff_{period}'] = data[col].diff(period)
                    result[f'{col}_pct_change_{period}'] = data[col].pct_change(period)

        return result

    @staticmethod
    def rolling_statistics(data: Union[pd.Series, pd.DataFrame],
                         windows: List[int] = [5, 10, 20, 60],
                         statistics: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Create rolling statistical features.

        Args:
            data: Input data
            windows: List of window sizes
            statistics: List of statistics to calculate

        Returns:
            DataFrame with rolling statistics
        """
        result = pd.DataFrame(index=data.index)

        if isinstance(data, pd.Series):
            data_df = data.to_frame('value')
        else:
            data_df = data

        for col in data_df.columns:
            for window in windows:
                rolling = data_df[col].rolling(window=window, min_periods=1)

                for stat in statistics:
                    if stat == 'mean':
                        result[f'{col}_mean_{window}'] = rolling.mean()
                    elif stat == 'std':
                        result[f'{col}_std_{window}'] = rolling.std()
                    elif stat == 'min':
                        result[f'{col}_min_{window}'] = rolling.min()
                    elif stat == 'max':
                        result[f'{col}_max_{window}'] = rolling.max()
                    elif stat == 'median':
                        result[f'{col}_median_{window}'] = rolling.median()
                    elif stat == 'skew':
                        result[f'{col}_skew_{window}'] = rolling.skew()
                    elif stat == 'kurt':
                        result[f'{col}_kurt_{window}'] = rolling.kurt()

        return result

    # ==================== Batch Transform ====================

    def transform_features(self, data: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Apply multiple transformations based on configuration.

        Args:
            data: Input features DataFrame
            config: Configuration dictionary specifying transformations

        Returns:
            DataFrame with transformed features
        """
        result = data.copy()

        try:
            # Polynomial features
            if config.get('polynomial', {}).get('enabled', False):
                degree = config['polynomial'].get('degree', 2)
                poly_features = self.polynomial_features(data, degree=degree)
                result = pd.concat([result, poly_features], axis=1)

            # Interaction features
            if config.get('interactions', {}).get('enabled', False):
                max_features = config['interactions'].get('max_features', 100)
                interactions = self.multiplicative_interactions(
                    data, max_features=max_features
                )
                result = pd.concat([result, interactions], axis=1)

            # Rank transformation
            if config.get('rank', {}).get('enabled', False):
                window = config['rank'].get('window', 20)
                rank_features = self.rolling_rank(data, window=window)
                rank_features.columns = [f'{col}_rank' for col in rank_features.columns]
                result = pd.concat([result, rank_features], axis=1)

            # Lagged features
            if config.get('lags', {}).get('enabled', False):
                lags = config['lags'].get('periods', [1, 2, 3, 5, 10])
                lagged = self.lagged_features(data, lags=lags)
                result = pd.concat([result, lagged], axis=1)

            # Rolling statistics
            if config.get('rolling_stats', {}).get('enabled', False):
                windows = config['rolling_stats'].get('windows', [5, 10, 20])
                stats = config['rolling_stats'].get('statistics', ['mean', 'std'])
                rolling_stats = self.rolling_statistics(data, windows=windows, statistics=stats)
                result = pd.concat([result, rolling_stats], axis=1)

        except Exception as e:
            warnings.warn(f"Error in feature transformation: {str(e)}")

        return result

    def fit_transform(self, data: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Fit transformer and transform data.

        Args:
            data: Input features DataFrame
            config: Optional configuration dictionary

        Returns:
            Transformed features DataFrame
        """
        if config is None:
            config = {
                'polynomial': {'enabled': False},
                'interactions': {'enabled': False},
                'rank': {'enabled': True, 'window': 20},
                'lags': {'enabled': True, 'periods': [1, 5, 10]},
                'rolling_stats': {'enabled': True, 'windows': [10, 20], 'statistics': ['mean', 'std']}
            }

        self.feature_names_ = data.columns.tolist()
        self.fitted_ = True

        return self.transform_features(data, config)
