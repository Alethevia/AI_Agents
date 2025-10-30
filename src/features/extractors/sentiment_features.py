"""
Sentiment Features Extractor

This module provides sentiment analysis features from multiple sources:
news, social media, options market, and analyst ratings.

Author: AI Trading System
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Tuple
from datetime import datetime, timedelta
import warnings


class SentimentFeatures:
    """
    Production-ready sentiment features calculator.

    Supports:
    - News sentiment analysis
    - Social media sentiment
    - Options market sentiment
    - Analyst sentiment
    - Sentiment aggregation and trends
    """

    def __init__(self, symbol: Optional[str] = None):
        """
        Initialize sentiment features calculator.

        Args:
            symbol: Optional symbol identifier
        """
        self.symbol = symbol

    # ==================== News Sentiment ====================

    @staticmethod
    def news_sentiment_score(sentiments: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate aggregated news sentiment score.

        Args:
            sentiments: Raw sentiment scores (-1 to 1)
            window: Rolling window for aggregation

        Returns:
            Aggregated sentiment score
        """
        return sentiments.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def news_sentiment_std(sentiments: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate news sentiment volatility (disagreement).

        Args:
            sentiments: Raw sentiment scores
            window: Rolling window

        Returns:
            Sentiment standard deviation
        """
        return sentiments.rolling(window=window, min_periods=1).std()

    @staticmethod
    def news_volume(timestamps: pd.Series, window: str = '1D') -> pd.Series:
        """
        Calculate news article volume.

        Args:
            timestamps: News article timestamps
            window: Time window for counting ('1D', '1H', etc.)

        Returns:
            News volume per period
        """
        if not isinstance(timestamps.index, pd.DatetimeIndex):
            timestamps.index = pd.to_datetime(timestamps.index)

        # Count news articles per period
        news_counts = timestamps.resample(window).count()
        return news_counts

    @staticmethod
    def sentiment_momentum(sentiments: pd.Series, short_window: int = 5,
                          long_window: int = 20) -> pd.Series:
        """
        Calculate sentiment momentum.

        Args:
            sentiments: Sentiment scores
            short_window: Short-term window
            long_window: Long-term window

        Returns:
            Sentiment momentum (short MA - long MA)
        """
        short_ma = sentiments.rolling(window=short_window, min_periods=1).mean()
        long_ma = sentiments.rolling(window=long_window, min_periods=1).mean()
        return short_ma - long_ma

    @staticmethod
    def sentiment_divergence(sentiment: pd.Series, price_return: pd.Series,
                           window: int = 20) -> pd.Series:
        """
        Calculate divergence between sentiment and price movements.

        Args:
            sentiment: Sentiment scores
            price_return: Price returns
            window: Rolling window

        Returns:
            Sentiment-price divergence
        """
        # Normalize both series
        sentiment_norm = (sentiment - sentiment.rolling(window).mean()) / \
                        (sentiment.rolling(window).std() + 1e-10)
        return_norm = (price_return - price_return.rolling(window).mean()) / \
                     (price_return.rolling(window).std() + 1e-10)

        divergence = sentiment_norm - return_norm
        return divergence.fillna(0)

    @staticmethod
    def weighted_sentiment(sentiments: pd.Series, sources_credibility: pd.Series) -> pd.Series:
        """
        Calculate credibility-weighted sentiment.

        Args:
            sentiments: Raw sentiment scores
            sources_credibility: Credibility scores for each source (0-1)

        Returns:
            Weighted sentiment score
        """
        weighted = sentiments * sources_credibility
        return weighted / (sources_credibility.sum() + 1e-10)

    @staticmethod
    def sentiment_surprise(current_sentiment: Union[float, pd.Series],
                          expected_sentiment: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate sentiment surprise (actual vs expected).

        Args:
            current_sentiment: Current sentiment reading
            expected_sentiment: Expected/consensus sentiment

        Returns:
            Sentiment surprise
        """
        return current_sentiment - expected_sentiment

    # ==================== Social Media Sentiment ====================

    @staticmethod
    def social_sentiment_score(mentions: pd.Series, positive: pd.Series,
                              negative: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate social media sentiment score.

        Args:
            mentions: Total mentions count
            positive: Positive mentions count
            negative: Negative mentions count
            window: Rolling window

        Returns:
            Social sentiment score (-1 to 1)
        """
        neutral = mentions - positive - negative
        sentiment = (positive - negative) / (mentions + 1e-10)

        # Smooth with rolling average
        return sentiment.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def social_buzz(mentions: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate social media buzz (mention volume relative to baseline).

        Args:
            mentions: Mention counts
            window: Window for baseline calculation

        Returns:
            Buzz ratio (1.0 = baseline)
        """
        baseline = mentions.rolling(window=window, min_periods=1).mean()
        buzz = mentions / (baseline + 1e-10)
        return buzz

    @staticmethod
    def social_sentiment_velocity(sentiment: pd.Series, periods: int = 5) -> pd.Series:
        """
        Calculate rate of change in social sentiment.

        Args:
            sentiment: Sentiment scores
            periods: Number of periods for velocity calculation

        Returns:
            Sentiment velocity
        """
        velocity = sentiment.diff(periods) / periods
        return velocity.fillna(0)

    @staticmethod
    def influencer_sentiment(sentiments: pd.Series, follower_counts: pd.Series,
                           window: int = 20) -> pd.Series:
        """
        Calculate follower-weighted influencer sentiment.

        Args:
            sentiments: Sentiment scores from influencers
            follower_counts: Number of followers for each influencer
            window: Rolling window

        Returns:
            Weighted influencer sentiment
        """
        # Weight by followers (more followers = more weight)
        weighted = sentiments * np.log1p(follower_counts)  # Log to reduce outlier effect
        total_weight = np.log1p(follower_counts).rolling(window=window, min_periods=1).sum()

        weighted_sentiment = weighted.rolling(window=window, min_periods=1).sum() / \
                           (total_weight + 1e-10)
        return weighted_sentiment

    @staticmethod
    def sentiment_dispersion(sentiments: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate sentiment dispersion (disagreement among sources).

        Args:
            sentiments: Individual sentiment scores
            window: Rolling window

        Returns:
            Sentiment dispersion (high = high disagreement)
        """
        # Use interquartile range as dispersion measure
        q75 = sentiments.rolling(window=window, min_periods=1).quantile(0.75)
        q25 = sentiments.rolling(window=window, min_periods=1).quantile(0.25)
        iqr = q75 - q25
        return iqr

    # ==================== Options Market Sentiment ====================

    @staticmethod
    def put_call_ratio(put_volume: Union[float, pd.Series],
                      call_volume: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate put/call ratio.

        Args:
            put_volume: Put option volume
            call_volume: Call option volume

        Returns:
            Put/call ratio (>1 = bearish, <1 = bullish)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratio = put_volume / call_volume
            if isinstance(ratio, pd.Series):
                ratio = ratio.replace([np.inf, -np.inf], np.nan)
            return ratio

    @staticmethod
    def put_call_ratio_oi(put_oi: Union[float, pd.Series],
                         call_oi: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate put/call ratio based on open interest.

        Args:
            put_oi: Put open interest
            call_oi: Call open interest

        Returns:
            Put/call ratio (OI)
        """
        return SentimentFeatures.put_call_ratio(put_oi, call_oi)

    @staticmethod
    def option_skew(otm_put_iv: pd.Series, otm_call_iv: pd.Series) -> pd.Series:
        """
        Calculate option skew (implied volatility skew).

        Args:
            otm_put_iv: Out-of-money put implied volatility
            otm_call_iv: Out-of-money call implied volatility

        Returns:
            Skew measure (positive = fear/bearish)
        """
        skew = otm_put_iv - otm_call_iv
        return skew

    @staticmethod
    def iv_rank(current_iv: Union[float, pd.Series],
               iv_series: pd.Series, window: int = 252) -> Union[float, pd.Series]:
        """
        Calculate implied volatility rank.

        Args:
            current_iv: Current implied volatility
            iv_series: Historical IV series
            window: Window for rank calculation

        Returns:
            IV rank (0-100 percentile)
        """
        if isinstance(current_iv, pd.Series):
            iv_min = iv_series.rolling(window=window, min_periods=1).min()
            iv_max = iv_series.rolling(window=window, min_periods=1).max()
            rank = ((current_iv - iv_min) / (iv_max - iv_min + 1e-10)) * 100
            return rank.fillna(50)
        else:
            iv_min = iv_series.tail(window).min()
            iv_max = iv_series.tail(window).max()
            return ((current_iv - iv_min) / (iv_max - iv_min + 1e-10)) * 100

    @staticmethod
    def max_pain(option_chain: pd.DataFrame, strike_col: str = 'strike',
                oi_col: str = 'open_interest') -> float:
        """
        Calculate max pain (strike with maximum loss for option writers).

        Args:
            option_chain: DataFrame with options data
            strike_col: Column name for strike prices
            oi_col: Column name for open interest

        Returns:
            Max pain strike price
        """
        strikes = option_chain[strike_col].unique()
        max_pain_strike = None
        min_total_value = float('inf')

        for strike in strikes:
            # Calculate total value if stock closes at this strike
            calls = option_chain[option_chain['type'] == 'call']
            puts = option_chain[option_chain['type'] == 'put']

            call_value = ((strikes > strike) * calls[oi_col] *
                         (strikes - strike)).sum()
            put_value = ((strikes < strike) * puts[oi_col] *
                        (strike - strikes)).sum()

            total_value = call_value + put_value

            if total_value < min_total_value:
                min_total_value = total_value
                max_pain_strike = strike

        return max_pain_strike if max_pain_strike is not None else strikes.mean()

    @staticmethod
    def gamma_exposure(option_chain: pd.DataFrame, spot_price: float,
                      strike_col: str = 'strike', gamma_col: str = 'gamma',
                      oi_col: str = 'open_interest') -> float:
        """
        Calculate market-maker gamma exposure.

        Args:
            option_chain: DataFrame with options data
            spot_price: Current spot price
            strike_col: Column name for strike prices
            gamma_col: Column name for gamma
            oi_col: Column name for open interest

        Returns:
            Total gamma exposure (negative = dealers short gamma)
        """
        # Market makers are short customer options
        # Calls: negative gamma, Puts: positive gamma for dealers
        calls = option_chain[option_chain['type'] == 'call'].copy()
        puts = option_chain[option_chain['type'] == 'put'].copy()

        call_gamma_exp = -(calls[gamma_col] * calls[oi_col] * spot_price ** 2 / 100).sum()
        put_gamma_exp = (puts[gamma_col] * puts[oi_col] * spot_price ** 2 / 100).sum()

        total_gamma = call_gamma_exp + put_gamma_exp
        return total_gamma

    # ==================== Analyst Sentiment ====================

    @staticmethod
    def analyst_rating_score(ratings: pd.Series) -> pd.Series:
        """
        Convert analyst ratings to numerical score.

        Args:
            ratings: Categorical ratings ('Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell')

        Returns:
            Numerical scores (1-5)
        """
        rating_map = {
            'Strong Buy': 5,
            'Buy': 4,
            'Hold': 3,
            'Sell': 2,
            'Strong Sell': 1
        }

        scores = ratings.map(rating_map)
        return scores.fillna(3)  # Default to Hold

    @staticmethod
    def analyst_consensus(ratings: pd.Series, window: int = 90) -> pd.Series:
        """
        Calculate analyst consensus rating.

        Args:
            ratings: Numerical analyst ratings
            window: Rolling window (days)

        Returns:
            Consensus rating
        """
        return ratings.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def analyst_revision_trend(ratings: pd.Series, window: int = 30) -> pd.Series:
        """
        Calculate analyst rating revision trend.

        Args:
            ratings: Numerical analyst ratings
            window: Window for trend calculation

        Returns:
            Revision trend (positive = upgrades, negative = downgrades)
        """
        # Calculate number of upgrades vs downgrades
        changes = ratings.diff()
        upgrades = (changes > 0).rolling(window=window, min_periods=1).sum()
        downgrades = (changes < 0).rolling(window=window, min_periods=1).sum()

        trend = (upgrades - downgrades) / (upgrades + downgrades + 1e-10)
        return trend.fillna(0)

    @staticmethod
    def price_target_implied_return(price_target: Union[float, pd.Series],
                                   current_price: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate implied return from analyst price targets.

        Args:
            price_target: Analyst price target
            current_price: Current stock price

        Returns:
            Implied return (%)
        """
        return ((price_target - current_price) / current_price) * 100

    @staticmethod
    def target_price_dispersion(price_targets: pd.Series, window: int = 90) -> pd.Series:
        """
        Calculate dispersion in analyst price targets.

        Args:
            price_targets: Analyst price targets
            window: Rolling window

        Returns:
            Coefficient of variation (std/mean)
        """
        mean = price_targets.rolling(window=window, min_periods=1).mean()
        std = price_targets.rolling(window=window, min_periods=1).std()
        cv = std / (mean + 1e-10)
        return cv.fillna(0)

    # ==================== Sentiment Change Rate ====================

    @staticmethod
    def sentiment_acceleration(sentiment: pd.Series, periods: int = 5) -> pd.Series:
        """
        Calculate sentiment acceleration (second derivative).

        Args:
            sentiment: Sentiment scores
            periods: Number of periods

        Returns:
            Sentiment acceleration
        """
        velocity = sentiment.diff(periods)
        acceleration = velocity.diff(periods)
        return acceleration.fillna(0)

    @staticmethod
    def sentiment_regime_change(sentiment: pd.Series, threshold: float = 0.5,
                               window: int = 20) -> pd.Series:
        """
        Detect sentiment regime changes.

        Args:
            sentiment: Sentiment scores
            threshold: Threshold for regime change
            window: Window for baseline calculation

        Returns:
            Regime indicator (1 = bullish, -1 = bearish, 0 = neutral)
        """
        baseline = sentiment.rolling(window=window, min_periods=1).mean()
        deviation = sentiment - baseline

        regime = pd.Series(0, index=sentiment.index)
        regime[deviation > threshold] = 1
        regime[deviation < -threshold] = -1

        return regime

    @staticmethod
    def sentiment_reversal_signal(sentiment: pd.Series, extreme_threshold: float = 0.8,
                                  window: int = 5) -> pd.Series:
        """
        Detect sentiment reversal signals (contrarian indicator).

        Args:
            sentiment: Sentiment scores (normalized -1 to 1)
            extreme_threshold: Threshold for extreme sentiment
            window: Window for reversal detection

        Returns:
            Reversal signal (1 = expect reversal down, -1 = expect reversal up, 0 = no signal)
        """
        signal = pd.Series(0, index=sentiment.index)

        # Extreme bullish sentiment -> bearish signal
        extreme_bullish = sentiment > extreme_threshold
        # Check if sentiment is decreasing
        decreasing = sentiment.diff(window) < 0
        signal[extreme_bullish & decreasing] = 1

        # Extreme bearish sentiment -> bullish signal
        extreme_bearish = sentiment < -extreme_threshold
        increasing = sentiment.diff(window) > 0
        signal[extreme_bearish & increasing] = -1

        return signal

    # ==================== Composite Sentiment ====================

    @staticmethod
    def composite_sentiment(news_sentiment: pd.Series, social_sentiment: pd.Series,
                          options_sentiment: pd.Series, analyst_sentiment: pd.Series,
                          weights: Optional[List[float]] = None) -> pd.Series:
        """
        Calculate composite sentiment from multiple sources.

        Args:
            news_sentiment: News sentiment scores
            social_sentiment: Social media sentiment scores
            options_sentiment: Options market sentiment
            analyst_sentiment: Analyst sentiment
            weights: Optional weights for each source (default: equal weight)

        Returns:
            Composite sentiment score
        """
        if weights is None:
            weights = [0.25, 0.25, 0.25, 0.25]

        # Normalize all sentiments to -1 to 1 range
        def normalize(series):
            min_val = series.min()
            max_val = series.max()
            if max_val - min_val == 0:
                return series * 0
            return 2 * (series - min_val) / (max_val - min_val) - 1

        news_norm = normalize(news_sentiment)
        social_norm = normalize(social_sentiment)
        options_norm = normalize(options_sentiment)
        analyst_norm = normalize(analyst_sentiment)

        composite = (weights[0] * news_norm +
                    weights[1] * social_norm +
                    weights[2] * options_norm +
                    weights[3] * analyst_norm)

        return composite

    # ==================== Batch Processing ====================

    def calculate_all_sentiment(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Calculate all sentiment features.

        Args:
            data: Dictionary with sentiment data from various sources

        Returns:
            Dictionary with all sentiment features
        """
        results = {}

        try:
            # News sentiment
            if 'news_sentiment' in data:
                results['news_sentiment_ma'] = self.news_sentiment_score(data['news_sentiment'])
                results['news_sentiment_std'] = self.news_sentiment_std(data['news_sentiment'])
                results['news_sentiment_momentum'] = self.sentiment_momentum(data['news_sentiment'])

            # Social sentiment
            if 'social_mentions' in data and 'social_positive' in data and 'social_negative' in data:
                results['social_sentiment'] = self.social_sentiment_score(
                    data['social_mentions'],
                    data['social_positive'],
                    data['social_negative']
                )
                results['social_buzz'] = self.social_buzz(data['social_mentions'])

            # Options sentiment
            if 'put_volume' in data and 'call_volume' in data:
                results['put_call_ratio'] = self.put_call_ratio(
                    data['put_volume'],
                    data['call_volume']
                )

            if 'put_oi' in data and 'call_oi' in data:
                results['put_call_ratio_oi'] = self.put_call_ratio_oi(
                    data['put_oi'],
                    data['call_oi']
                )

            # Analyst sentiment
            if 'analyst_ratings' in data:
                scores = self.analyst_rating_score(data['analyst_ratings'])
                results['analyst_score'] = scores
                results['analyst_consensus'] = self.analyst_consensus(scores)
                results['analyst_trend'] = self.analyst_revision_trend(scores)

            # Composite sentiment
            sentiment_sources = []
            if 'news_sentiment_ma' in results:
                sentiment_sources.append(results['news_sentiment_ma'])
            if 'social_sentiment' in results:
                sentiment_sources.append(results['social_sentiment'])
            if 'put_call_ratio' in results:
                # Invert put/call ratio (high ratio = bearish)
                sentiment_sources.append(1 - results['put_call_ratio'].clip(0, 2) / 2)
            if 'analyst_score' in results:
                # Normalize analyst score
                sentiment_sources.append((results['analyst_score'] - 3) / 2)

            if len(sentiment_sources) >= 2:
                results['composite_sentiment'] = pd.concat(sentiment_sources, axis=1).mean(axis=1)

        except Exception as e:
            warnings.warn(f"Error calculating sentiment features: {str(e)}")

        return results
