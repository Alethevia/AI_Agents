"""
Technical Indicators Feature Extractor

This module provides a comprehensive set of technical indicators for financial time series data.
All functions are vectorized for performance and handle missing data gracefully.

Author: AI Trading System
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, List
from scipy.signal import argrelextrema
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import warnings


class TechnicalIndicators:
    """
    Production-ready technical indicators calculator.

    Supports:
    - Batch and streaming computation
    - Multiple symbols simultaneously
    - Missing data handling
    - Vectorized operations for performance
    """

    def __init__(self, symbol: Optional[str] = None):
        """
        Initialize technical indicators calculator.

        Args:
            symbol: Optional symbol identifier for tracking
        """
        self.symbol = symbol

    # ==================== Moving Averages ====================

    @staticmethod
    def sma(data: Union[pd.Series, pd.DataFrame], period: int) -> Union[pd.Series, pd.DataFrame]:
        """
        Simple Moving Average (SMA).

        Args:
            data: Price data (Series or DataFrame)
            period: Lookback period

        Returns:
            SMA values
        """
        if isinstance(data, pd.DataFrame):
            return data.rolling(window=period, min_periods=1).mean()
        return data.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def ema(data: Union[pd.Series, pd.DataFrame], period: int,
            adjust: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """
        Exponential Moving Average (EMA).

        Args:
            data: Price data
            period: Lookback period
            adjust: Use adjusted EMA calculation

        Returns:
            EMA values
        """
        return data.ewm(span=period, adjust=adjust, min_periods=1).mean()

    @staticmethod
    def wma(data: Union[pd.Series, pd.DataFrame], period: int) -> Union[pd.Series, pd.DataFrame]:
        """
        Weighted Moving Average (WMA).

        Args:
            data: Price data
            period: Lookback period

        Returns:
            WMA values
        """
        weights = np.arange(1, period + 1)

        def calc_wma(x):
            if len(x) < period:
                w = np.arange(1, len(x) + 1)
                return np.sum(x * w) / np.sum(w) if len(x) > 0 else np.nan
            return np.sum(x * weights) / np.sum(weights)

        if isinstance(data, pd.DataFrame):
            return data.rolling(window=period).apply(calc_wma, raw=True)
        return data.rolling(window=period).apply(calc_wma, raw=True)

    @staticmethod
    def vwap(price: pd.Series, volume: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP).

        Args:
            price: Typical price (H+L+C)/3 or close price
            volume: Volume data
            period: Optional rolling period (None for cumulative)

        Returns:
            VWAP values
        """
        pv = price * volume

        if period is None:
            # Cumulative VWAP (reset daily if index is datetime)
            if isinstance(price.index, pd.DatetimeIndex):
                return (pv.groupby(price.index.date).cumsum() /
                       volume.groupby(price.index.date).cumsum())
            return pv.cumsum() / volume.cumsum()
        else:
            # Rolling VWAP
            return (pv.rolling(window=period, min_periods=1).sum() /
                   volume.rolling(window=period, min_periods=1).sum())

    # ==================== Momentum Indicators ====================

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI).

        Args:
            data: Price data
            period: Lookback period (default: 14)

        Returns:
            RSI values (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)  # Neutral RSI for first values

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26,
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD).

        Args:
            data: Price data
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = data.ewm(span=fast, adjust=False, min_periods=1).mean()
        ema_slow = data.ewm(span=slow, adjust=False, min_periods=1).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = k.fillna(50)  # Neutral value

        d = k.rolling(window=d_period, min_periods=1).mean()

        return k, d

    @staticmethod
    def roc(data: pd.Series, period: int = 12) -> pd.Series:
        """
        Rate of Change (ROC).

        Args:
            data: Price data
            period: Lookback period (default: 12)

        Returns:
            ROC values (percentage)
        """
        roc = ((data - data.shift(period)) / data.shift(period)) * 100
        return roc.fillna(0)

    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Momentum indicator.

        Args:
            data: Price data
            period: Lookback period (default: 10)

        Returns:
            Momentum values
        """
        return data.diff(period).fillna(0)

    # ==================== Volatility Indicators ====================

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20,
                       num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Args:
            data: Price data
            period: Moving average period (default: 20)
            num_std: Number of standard deviations (default: 2.0)

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle = data.rolling(window=period, min_periods=1).mean()
        std = data.rolling(window=period, min_periods=1).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """
        Average True Range (ATR).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period (default: 14)

        Returns:
            ATR values
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False, min_periods=1).mean()

        return atr

    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                        ema_period: int = 20, atr_period: int = 10,
                        multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            ema_period: EMA period (default: 20)
            atr_period: ATR period (default: 10)
            multiplier: ATR multiplier (default: 2.0)

        Returns:
            Tuple of (Upper channel, Middle line, Lower channel)
        """
        middle = close.ewm(span=ema_period, adjust=False, min_periods=1).mean()
        atr = TechnicalIndicators.atr(high, low, close, atr_period)

        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)

        return upper, middle, lower

    @staticmethod
    def historical_volatility(data: pd.Series, period: int = 20,
                            annualize: bool = True) -> pd.Series:
        """
        Historical Volatility.

        Args:
            data: Price data
            period: Lookback period (default: 20)
            annualize: Annualize volatility (default: True)

        Returns:
            Historical volatility values
        """
        log_returns = np.log(data / data.shift(1))
        volatility = log_returns.rolling(window=period, min_periods=1).std()

        if annualize:
            volatility = volatility * np.sqrt(252)  # Assuming daily data

        return volatility.fillna(0)

    # ==================== Volume Indicators ====================

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume (OBV).

        Args:
            close: Close prices
            volume: Volume data

        Returns:
            OBV values
        """
        direction = np.sign(close.diff())
        direction[0] = 0  # First value has no direction

        obv = (direction * volume).cumsum()
        return obv

    @staticmethod
    def cmf(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Chaikin Money Flow (CMF).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            period: Lookback period (default: 20)

        Returns:
            CMF values
        """
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)  # Handle division by zero

        mfv = mfm * volume
        cmf = (mfv.rolling(window=period, min_periods=1).sum() /
               volume.rolling(window=period, min_periods=1).sum())

        return cmf.fillna(0)

    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            period: Lookback period (default: 14)

        Returns:
            MFI values (0-100)
        """
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        price_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(price_diff > 0, 0)
        negative_flow = raw_money_flow.where(price_diff < 0, 0)

        positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()

        mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, 1e-10)))

        return mfi.fillna(50)

    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Ratio (current volume vs average volume).

        Args:
            volume: Volume data
            period: Average period (default: 20)

        Returns:
            Volume ratio values
        """
        avg_volume = volume.rolling(window=period, min_periods=1).mean()
        ratio = volume / avg_volume.replace(0, 1)

        return ratio.fillna(1)

    # ==================== Trend Indicators ====================

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index (ADX).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period (default: 14)

        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Smooth with Wilder's smoothing
        atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr)

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

    @staticmethod
    def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                      af_start: float = 0.02, af_increment: float = 0.02,
                      af_max: float = 0.2) -> pd.Series:
        """
        Parabolic SAR (Stop and Reverse).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            af_start: Starting acceleration factor (default: 0.02)
            af_increment: AF increment (default: 0.02)
            af_max: Maximum AF (default: 0.2)

        Returns:
            Parabolic SAR values
        """
        sar = close.copy()
        trend = pd.Series(1, index=close.index)  # 1 for uptrend, -1 for downtrend
        af = af_start
        ep = high.iloc[0]  # Extreme point

        sar.iloc[0] = low.iloc[0]

        for i in range(1, len(close)):
            # Update SAR
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])

            # Check for trend reversal
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])

                if low.iloc[i] < sar.iloc[i]:  # Reverse to downtrend
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = af_start
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + af_increment, af_max)
            else:  # Downtrend
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])

                if high.iloc[i] > sar.iloc[i]:  # Reverse to uptrend
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = af_start
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + af_increment, af_max)

        return sar

    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """
        Supertrend Indicator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period (default: 10)
            multiplier: ATR multiplier (default: 3.0)

        Returns:
            Tuple of (Supertrend, Trend direction)
        """
        atr = TechnicalIndicators.atr(high, low, close, period)
        hl_avg = (high + low) / 2

        # Calculate basic upper and lower bands
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)

        supertrend = pd.Series(0.0, index=close.index)
        direction = pd.Series(1, index=close.index)  # 1 for uptrend, -1 for downtrend

        for i in range(period, len(close)):
            # Update bands
            if i > period:
                if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    lower_band.iloc[i] = lower_band.iloc[i-1]

                if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i]
                else:
                    upper_band.iloc[i] = upper_band.iloc[i-1]

            # Determine trend
            if close.iloc[i] <= upper_band.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]

            # Check for trend change
            if i > period:
                if direction.iloc[i] == 1 and direction.iloc[i-1] == -1:
                    supertrend.iloc[i] = lower_band.iloc[i]
                elif direction.iloc[i] == -1 and direction.iloc[i-1] == 1:
                    supertrend.iloc[i] = upper_band.iloc[i]

        return supertrend, direction

    # ==================== Support/Resistance ====================

    @staticmethod
    def support_resistance_levels(close: pd.Series, high: pd.Series, low: pd.Series,
                                  num_levels: int = 3, window: int = 20) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels.

        Args:
            close: Close prices
            high: High prices
            low: Low prices
            num_levels: Number of levels to identify (default: 3)
            window: Window for local extrema detection (default: 20)

        Returns:
            Dictionary with 'support' and 'resistance' lists
        """
        # Find local maxima (resistance) and minima (support)
        high_arr = high.values
        low_arr = low.values

        # Use scipy to find local extrema
        resistance_idx = argrelextrema(high_arr, np.greater, order=window)[0]
        support_idx = argrelextrema(low_arr, np.less, order=window)[0]

        # Get the values at these indices
        resistance_values = high_arr[resistance_idx]
        support_values = low_arr[support_idx]

        # Cluster similar levels
        def cluster_levels(levels, tolerance=0.02):
            if len(levels) == 0:
                return []

            sorted_levels = np.sort(levels)
            clustered = []
            current_cluster = [sorted_levels[0]]

            for level in sorted_levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                    current_cluster.append(level)
                else:
                    clustered.append(np.mean(current_cluster))
                    current_cluster = [level]

            clustered.append(np.mean(current_cluster))
            return clustered

        resistance_clustered = cluster_levels(resistance_values)
        support_clustered = cluster_levels(support_values)

        # Select top N levels closest to current price
        current_price = close.iloc[-1]

        resistance_sorted = sorted(
            [r for r in resistance_clustered if r > current_price],
            key=lambda x: abs(x - current_price)
        )[:num_levels]

        support_sorted = sorted(
            [s for s in support_clustered if s < current_price],
            key=lambda x: abs(x - current_price)
        )[:num_levels]

        return {
            'resistance': resistance_sorted,
            'support': support_sorted
        }

    # ==================== Chart Patterns ====================

    @staticmethod
    def detect_double_top_bottom(high: pd.Series, low: pd.Series, close: pd.Series,
                                window: int = 20, tolerance: float = 0.03) -> pd.Series:
        """
        Detect double top and double bottom patterns.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Window for pattern detection (default: 20)
            tolerance: Price tolerance for matching peaks/troughs (default: 0.03)

        Returns:
            Series with pattern signals (1: double top, -1: double bottom, 0: no pattern)
        """
        signals = pd.Series(0, index=close.index)

        # Find local extrema
        high_arr = high.values
        low_arr = low.values

        peak_idx = argrelextrema(high_arr, np.greater, order=window//2)[0]
        trough_idx = argrelextrema(low_arr, np.less, order=window//2)[0]

        # Check for double tops
        for i in range(1, len(peak_idx)):
            if peak_idx[i] - peak_idx[i-1] < window:
                continue

            peak1 = high_arr[peak_idx[i-1]]
            peak2 = high_arr[peak_idx[i]]

            if abs(peak1 - peak2) / peak1 < tolerance:
                # Found double top
                if peak_idx[i] < len(signals):
                    signals.iloc[peak_idx[i]] = 1

        # Check for double bottoms
        for i in range(1, len(trough_idx)):
            if trough_idx[i] - trough_idx[i-1] < window:
                continue

            trough1 = low_arr[trough_idx[i-1]]
            trough2 = low_arr[trough_idx[i]]

            if abs(trough1 - trough2) / trough1 < tolerance:
                # Found double bottom
                if trough_idx[i] < len(signals):
                    signals.iloc[trough_idx[i]] = -1

        return signals

    @staticmethod
    def detect_head_shoulders(high: pd.Series, low: pd.Series, close: pd.Series,
                             window: int = 30) -> pd.Series:
        """
        Detect head and shoulders pattern.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: Window for pattern detection (default: 30)

        Returns:
            Series with pattern signals (1: head & shoulders, -1: inverse, 0: no pattern)
        """
        signals = pd.Series(0, index=close.index)

        high_arr = high.values
        low_arr = low.values

        # Find peaks for head and shoulders
        peak_idx = argrelextrema(high_arr, np.greater, order=window//3)[0]

        # Need at least 3 peaks
        for i in range(2, len(peak_idx)):
            left_shoulder_idx = peak_idx[i-2]
            head_idx = peak_idx[i-1]
            right_shoulder_idx = peak_idx[i]

            # Check spacing
            if (head_idx - left_shoulder_idx < window//2 or
                right_shoulder_idx - head_idx < window//2):
                continue

            left_shoulder = high_arr[left_shoulder_idx]
            head = high_arr[head_idx]
            right_shoulder = high_arr[right_shoulder_idx]

            # Check pattern: head higher than shoulders, shoulders similar height
            if (head > left_shoulder and head > right_shoulder and
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):

                if right_shoulder_idx < len(signals):
                    signals.iloc[right_shoulder_idx] = 1

        # Find troughs for inverse head and shoulders
        trough_idx = argrelextrema(low_arr, np.less, order=window//3)[0]

        for i in range(2, len(trough_idx)):
            left_shoulder_idx = trough_idx[i-2]
            head_idx = trough_idx[i-1]
            right_shoulder_idx = trough_idx[i]

            if (head_idx - left_shoulder_idx < window//2 or
                right_shoulder_idx - head_idx < window//2):
                continue

            left_shoulder = low_arr[left_shoulder_idx]
            head = low_arr[head_idx]
            right_shoulder = low_arr[right_shoulder_idx]

            # Check pattern: head lower than shoulders, shoulders similar height
            if (head < left_shoulder and head < right_shoulder and
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):

                if right_shoulder_idx < len(signals):
                    signals.iloc[right_shoulder_idx] = -1

        return signals

    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given dataset.

        Args:
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume)

        Returns:
            DataFrame with all technical indicators
        """
        df = data.copy()

        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = self.sma(df['close'], period)
            df[f'ema_{period}'] = self.ema(df['close'], period)

        # VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = self.vwap(typical_price, df['volume'])

        # Momentum
        df['rsi_14'] = self.rsi(df['close'], 14)
        macd_line, signal_line, histogram = self.macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram

        k, d = self.stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = k
        df['stoch_d'] = d

        df['roc_12'] = self.roc(df['close'], 12)

        # Volatility
        upper, middle, lower = self.bollinger_bands(df['close'])
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle

        df['atr_14'] = self.atr(df['high'], df['low'], df['close'], 14)
        df['historical_vol'] = self.historical_volatility(df['close'])

        # Volume
        df['obv'] = self.obv(df['close'], df['volume'])
        df['cmf'] = self.cmf(df['high'], df['low'], df['close'], df['volume'])
        df['mfi'] = self.mfi(df['high'], df['low'], df['close'], df['volume'])
        df['volume_ratio'] = self.volume_ratio(df['volume'])

        # Trend
        adx, plus_di, minus_di = self.adx(df['high'], df['low'], df['close'])
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        df['psar'] = self.parabolic_sar(df['high'], df['low'], df['close'])
        supertrend, direction = self.supertrend(df['high'], df['low'], df['close'])
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction

        # Patterns
        df['double_top_bottom'] = self.detect_double_top_bottom(df['high'], df['low'], df['close'])
        df['head_shoulders'] = self.detect_head_shoulders(df['high'], df['low'], df['close'])

        return df
