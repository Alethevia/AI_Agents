"""
Market Microstructure Features Extractor

This module provides market microstructure features for high-frequency trading analysis.
Includes order book, trade flow, and liquidity metrics.

Author: AI Trading System
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Tuple, List
import warnings


class MarketMicrostructure:
    """
    Production-ready market microstructure features calculator.

    Supports:
    - Order book analysis
    - Trade flow metrics
    - Liquidity indicators
    - Price impact estimation
    - Vectorized computation
    """

    def __init__(self, symbol: Optional[str] = None):
        """
        Initialize market microstructure calculator.

        Args:
            symbol: Optional symbol identifier
        """
        self.symbol = symbol

    # ==================== Bid-Ask Spread ====================

    @staticmethod
    def bid_ask_spread(bid: Union[float, pd.Series],
                      ask: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate bid-ask spread.

        Args:
            bid: Bid price
            ask: Ask price

        Returns:
            Absolute spread
        """
        return ask - bid

    @staticmethod
    def relative_spread(bid: Union[float, pd.Series],
                       ask: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate relative (percentage) bid-ask spread.

        Args:
            bid: Bid price
            ask: Ask price

        Returns:
            Relative spread (%)
        """
        mid_price = (bid + ask) / 2
        spread = ask - bid
        return (spread / mid_price) * 100

    @staticmethod
    def effective_spread(trade_price: Union[float, pd.Series],
                        mid_price: Union[float, pd.Series],
                        side: Union[int, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate effective spread (actual execution cost).

        Args:
            trade_price: Actual trade price
            mid_price: Mid quote price at trade time
            side: Trade side (1 for buy, -1 for sell)

        Returns:
            Effective spread
        """
        return 2 * abs(trade_price - mid_price) * side

    @staticmethod
    def quoted_spread(bid: pd.Series, ask: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate time-weighted quoted spread.

        Args:
            bid: Bid prices
            ask: Ask prices
            window: Rolling window for averaging

        Returns:
            Time-weighted quoted spread
        """
        spread = ask - ask
        return spread.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def spread_volatility(bid: pd.Series, ask: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate spread volatility.

        Args:
            bid: Bid prices
            ask: Ask prices
            window: Rolling window

        Returns:
            Standard deviation of spread
        """
        spread = ask - bid
        return spread.rolling(window=window, min_periods=1).std()

    # ==================== Order Imbalance ====================

    @staticmethod
    def order_imbalance(bid_volume: Union[float, pd.Series],
                       ask_volume: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate order book imbalance.

        Args:
            bid_volume: Total bid volume
            ask_volume: Total ask volume

        Returns:
            Order imbalance (-1 to 1, positive = more bids)
        """
        total_volume = bid_volume + ask_volume
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imbalance = (bid_volume - ask_volume) / total_volume
            if isinstance(imbalance, pd.Series):
                imbalance = imbalance.replace([np.inf, -np.inf], 0)
            return imbalance

    @staticmethod
    def depth_imbalance(bid_depth: pd.Series, ask_depth: pd.Series,
                       levels: int = 5) -> pd.Series:
        """
        Calculate multi-level order book depth imbalance.

        Args:
            bid_depth: DataFrame with bid volumes at each level
            ask_depth: DataFrame with ask volumes at each level
            levels: Number of levels to consider

        Returns:
            Weighted depth imbalance
        """
        # Weight by level (closer levels have more weight)
        weights = np.exp(-np.arange(levels) / levels)
        weights = weights / weights.sum()

        if isinstance(bid_depth, pd.DataFrame) and isinstance(ask_depth, pd.DataFrame):
            weighted_bids = (bid_depth.iloc[:, :levels] * weights).sum(axis=1)
            weighted_asks = (ask_depth.iloc[:, :levels] * weights).sum(axis=1)
        else:
            weighted_bids = bid_depth
            weighted_asks = ask_depth

        return (weighted_bids - weighted_asks) / (weighted_bids + weighted_asks + 1e-10)

    @staticmethod
    def volume_imbalance(buy_volume: pd.Series, sell_volume: pd.Series,
                        window: int = 20) -> pd.Series:
        """
        Calculate rolling volume imbalance from trades.

        Args:
            buy_volume: Buy trade volumes
            sell_volume: Sell trade volumes
            window: Rolling window

        Returns:
            Volume imbalance ratio
        """
        total_volume = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / (total_volume + 1e-10)
        return imbalance.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def trade_imbalance(trades: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate trade imbalance from trade data.

        Args:
            trades: DataFrame with 'side' and 'volume' columns
            window: Rolling window

        Returns:
            Trade imbalance
        """
        # Assume side: 1 for buy, -1 for sell
        signed_volume = trades['volume'] * trades['side']
        return signed_volume.rolling(window=window, min_periods=1).sum()

    # ==================== Trade Flow Toxicity ====================

    @staticmethod
    def vpin(trade_volume: pd.Series, trade_side: pd.Series,
            bucket_size: int = 50, num_buckets: int = 50) -> pd.Series:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN).

        Args:
            trade_volume: Trade volumes
            trade_side: Trade side (1 for buy, -1 for sell)
            bucket_size: Volume per bucket
            num_buckets: Number of buckets to average

        Returns:
            VPIN values (0-1, higher = more toxic flow)
        """
        # Create volume buckets
        cumulative_volume = trade_volume.cumsum()
        bucket_ids = (cumulative_volume / bucket_size).astype(int)

        # Calculate buy/sell volume per bucket
        buy_volume = (trade_volume * (trade_side > 0)).groupby(bucket_ids).sum()
        sell_volume = (trade_volume * (trade_side < 0)).groupby(bucket_ids).sum()

        # Calculate VPIN
        total_bucket_volume = buy_volume + sell_volume
        vpin = abs(buy_volume - sell_volume) / (total_bucket_volume + 1e-10)

        # Rolling average over buckets
        vpin_smooth = vpin.rolling(window=num_buckets, min_periods=1).mean()

        # Map back to original index
        result = pd.Series(0.0, index=trade_volume.index)
        for i, bucket_id in enumerate(bucket_ids):
            if bucket_id in vpin_smooth.index:
                result.iloc[i] = vpin_smooth.loc[bucket_id]

        return result

    @staticmethod
    def order_flow_toxicity(price_change: pd.Series, order_flow: pd.Series,
                          window: int = 20) -> pd.Series:
        """
        Calculate order flow toxicity (correlation between flow and price changes).

        Args:
            price_change: Price changes
            order_flow: Signed order flow
            window: Rolling window

        Returns:
            Toxicity measure (correlation)
        """
        toxicity = price_change.rolling(window=window, min_periods=1).corr(order_flow)
        return toxicity.fillna(0)

    @staticmethod
    def kyle_lambda(price_change: pd.Series, signed_volume: pd.Series,
                   window: int = 20) -> pd.Series:
        """
        Calculate Kyle's Lambda (price impact coefficient).

        Args:
            price_change: Price changes
            signed_volume: Signed trade volume
            window: Rolling window

        Returns:
            Kyle's lambda (price impact per unit volume)
        """
        # Rolling regression: price_change = lambda * signed_volume
        def calc_lambda(df):
            if len(df) < 2:
                return 0
            try:
                cov = np.cov(df['volume'], df['price'])[0, 1]
                var = np.var(df['volume'])
                return cov / var if var > 0 else 0
            except:
                return 0

        combined = pd.DataFrame({'price': price_change, 'volume': signed_volume})
        kyle_lambda = combined.rolling(window=window, min_periods=2).apply(
            lambda x: calc_lambda(pd.DataFrame(x.values, columns=['volume', 'price'])),
            raw=False
        )

        return kyle_lambda.iloc[:, 0] if isinstance(kyle_lambda, pd.DataFrame) else kyle_lambda

    # ==================== Price Impact ====================

    @staticmethod
    def temporary_impact(trade_price: pd.Series, mid_price: pd.Series,
                        trade_volume: pd.Series) -> pd.Series:
        """
        Calculate temporary price impact.

        Args:
            trade_price: Trade execution price
            mid_price: Mid price before trade
            trade_volume: Trade volume

        Returns:
            Temporary impact (basis points per share)
        """
        price_change = (trade_price - mid_price) / mid_price
        impact = (price_change / trade_volume) * 10000  # Convert to bps
        return impact.replace([np.inf, -np.inf], 0)

    @staticmethod
    def permanent_impact(mid_price_after: pd.Series, mid_price_before: pd.Series,
                        trade_volume: pd.Series, window: int = 10) -> pd.Series:
        """
        Calculate permanent price impact.

        Args:
            mid_price_after: Mid price after trade execution
            mid_price_before: Mid price before trade
            trade_volume: Trade volume
            window: Window to measure "after"

        Returns:
            Permanent impact
        """
        # Shift to get mid price some periods after trade
        mid_after = mid_price_after.shift(-window)
        price_change = (mid_after - mid_price_before) / mid_price_before

        impact = (price_change / trade_volume) * 10000
        return impact.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def price_impact_ratio(trade_volume: pd.Series, avg_volume: pd.Series,
                          volatility: pd.Series) -> pd.Series:
        """
        Calculate expected price impact ratio.

        Args:
            trade_volume: Trade volume
            avg_volume: Average daily volume
            volatility: Price volatility

        Returns:
            Expected price impact (%)
        """
        # Square root impact model
        volume_ratio = trade_volume / avg_volume
        impact = volatility * np.sqrt(volume_ratio)
        return impact

    @staticmethod
    def market_impact_cost(trade_volume: pd.Series, daily_volume: pd.Series,
                          volatility: pd.Series, participation_rate: float = 0.1) -> pd.Series:
        """
        Estimate market impact cost.

        Args:
            trade_volume: Trade volume
            daily_volume: Average daily volume
            volatility: Daily volatility
            participation_rate: % of daily volume per interval

        Returns:
            Estimated market impact cost (%)
        """
        # Almgren-Chriss model approximation
        volume_fraction = trade_volume / daily_volume
        time_fraction = participation_rate

        temporary = 0.5 * volatility * np.sqrt(time_fraction)
        permanent = volatility * volume_fraction

        total_impact = temporary + permanent
        return total_impact

    # ==================== Market Depth Features ====================

    @staticmethod
    def book_depth(bid_volumes: pd.Series, ask_volumes: pd.Series,
                  levels: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate order book depth.

        Args:
            bid_volumes: DataFrame/Series with bid volumes at each level
            ask_volumes: DataFrame/Series with ask volumes at each level
            levels: Number of levels to sum

        Returns:
            Tuple of (total bid depth, total ask depth)
        """
        if isinstance(bid_volumes, pd.DataFrame):
            bid_depth = bid_volumes.iloc[:, :levels].sum(axis=1)
            ask_depth = ask_volumes.iloc[:, :levels].sum(axis=1)
        else:
            bid_depth = bid_volumes
            ask_depth = ask_volumes

        return bid_depth, ask_depth

    @staticmethod
    def book_pressure(bid_volumes: pd.Series, ask_volumes: pd.Series,
                     bid_prices: pd.Series, ask_prices: pd.Series) -> pd.Series:
        """
        Calculate order book pressure.

        Args:
            bid_volumes: Bid volumes
            ask_volumes: Ask volumes
            bid_prices: Bid prices
            ask_prices: Ask prices

        Returns:
            Book pressure (positive = buying pressure)
        """
        mid_price = (bid_prices + ask_prices) / 2

        bid_value = bid_volumes * bid_prices
        ask_value = ask_volumes * ask_prices

        pressure = (bid_value - ask_value) / (bid_value + ask_value + 1e-10)
        return pressure

    @staticmethod
    def depth_improvement(bid_volumes: pd.Series, ask_volumes: pd.Series,
                         window: int = 20) -> pd.Series:
        """
        Calculate depth improvement (change in liquidity).

        Args:
            bid_volumes: Total bid volume
            ask_volumes: Total ask volume
            window: Rolling window

        Returns:
            Depth improvement ratio
        """
        total_depth = bid_volumes + ask_volumes
        depth_change = total_depth.pct_change(window)
        return depth_change.fillna(0)

    @staticmethod
    def liquidity_imbalance(bid_depth: pd.Series, ask_depth: pd.Series,
                          spread: pd.Series) -> pd.Series:
        """
        Calculate liquidity-weighted imbalance.

        Args:
            bid_depth: Total bid depth
            ask_depth: Total ask depth
            spread: Bid-ask spread

        Returns:
            Liquidity imbalance adjusted for spread
        """
        # Weight by inverse of spread (tighter spread = more liquid)
        spread_weight = 1 / (spread + 1e-10)

        weighted_bid = bid_depth * spread_weight
        weighted_ask = ask_depth * spread_weight

        imbalance = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask + 1e-10)
        return imbalance

    # ==================== Volume Profile ====================

    @staticmethod
    def volume_at_price(prices: pd.Series, volumes: pd.Series,
                       price_bins: int = 50) -> pd.DataFrame:
        """
        Calculate volume profile (volume traded at each price level).

        Args:
            prices: Trade prices
            volumes: Trade volumes
            price_bins: Number of price bins

        Returns:
            DataFrame with price levels and volumes
        """
        # Create price bins
        price_min = prices.min()
        price_max = prices.max()
        bins = np.linspace(price_min, price_max, price_bins + 1)

        # Assign each trade to a bin
        price_bins_idx = np.digitize(prices, bins)

        # Sum volume per bin
        volume_profile = pd.DataFrame({
            'price_level': bins[:-1],
            'volume': 0.0
        })

        for i in range(len(prices)):
            bin_idx = price_bins_idx[i] - 1
            if 0 <= bin_idx < len(volume_profile):
                volume_profile.loc[bin_idx, 'volume'] += volumes.iloc[i]

        return volume_profile

    @staticmethod
    def poc_level(prices: pd.Series, volumes: pd.Series,
                 price_bins: int = 50) -> float:
        """
        Calculate Point of Control (POC) - price level with highest volume.

        Args:
            prices: Trade prices
            volumes: Trade volumes
            price_bins: Number of price bins

        Returns:
            POC price level
        """
        profile = MarketMicrostructure.volume_at_price(prices, volumes, price_bins)
        poc_idx = profile['volume'].idxmax()
        return profile.loc[poc_idx, 'price_level']

    @staticmethod
    def value_area(prices: pd.Series, volumes: pd.Series,
                  value_area_pct: float = 0.7, price_bins: int = 50) -> Tuple[float, float]:
        """
        Calculate Value Area (price range containing X% of volume).

        Args:
            prices: Trade prices
            volumes: Trade volumes
            value_area_pct: Percentage of volume in value area (default: 0.7)
            price_bins: Number of price bins

        Returns:
            Tuple of (value area low, value area high)
        """
        profile = MarketMicrostructure.volume_at_price(prices, volumes, price_bins)

        # Sort by volume
        profile_sorted = profile.sort_values('volume', ascending=False)

        # Find price levels that contain value_area_pct of volume
        total_volume = profile_sorted['volume'].sum()
        target_volume = total_volume * value_area_pct

        cumulative_volume = 0
        value_area_prices = []

        for idx, row in profile_sorted.iterrows():
            cumulative_volume += row['volume']
            value_area_prices.append(row['price_level'])

            if cumulative_volume >= target_volume:
                break

        va_low = min(value_area_prices)
        va_high = max(value_area_prices)

        return va_low, va_high

    @staticmethod
    def volume_weighted_price(prices: pd.Series, volumes: pd.Series,
                            window: int = 20) -> pd.Series:
        """
        Calculate volume-weighted average price (VWAP).

        Args:
            prices: Prices
            volumes: Volumes
            window: Rolling window

        Returns:
            VWAP values
        """
        vwap = (prices * volumes).rolling(window=window, min_periods=1).sum() / \
               volumes.rolling(window=window, min_periods=1).sum()
        return vwap

    # ==================== Batch Processing ====================

    def calculate_all_microstructure(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate all market microstructure features.

        Args:
            data: Dictionary with market data (bid, ask, volumes, prices, etc.)

        Returns:
            Dictionary with all microstructure features
        """
        results = {}

        try:
            # Spread metrics
            if 'bid' in data and 'ask' in data:
                results['spread'] = self.bid_ask_spread(data['bid'], data['ask'])
                results['relative_spread'] = self.relative_spread(data['bid'], data['ask'])

                if 'bid_volume' in data and 'ask_volume' in data:
                    results['order_imbalance'] = self.order_imbalance(
                        data['bid_volume'], data['ask_volume']
                    )

            # Trade flow metrics
            if 'trade_volume' in data and 'trade_side' in data:
                results['vpin'] = self.vpin(data['trade_volume'], data['trade_side'])

            # Depth metrics
            if 'bid_volume' in data and 'ask_volume' in data:
                bid_depth, ask_depth = self.book_depth(data['bid_volume'], data['ask_volume'])
                results['bid_depth'] = bid_depth
                results['ask_depth'] = ask_depth
                results['total_depth'] = bid_depth + ask_depth

            # Volume profile
            if 'price' in data and 'volume' in data:
                results['vwap'] = self.volume_weighted_price(data['price'], data['volume'])

        except Exception as e:
            warnings.warn(f"Error calculating microstructure features: {str(e)}")

        return results
