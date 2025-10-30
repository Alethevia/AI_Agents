"""
Slippage Estimation Models

This module provides various slippage estimation models including:
- Linear market impact model
- Square-root market impact model
- Volume-based slippage
- Spread-based slippage
- Historical slippage analysis
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal
from abc import ABC, abstractmethod
import numpy as np
from collections import deque


logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Current market state for slippage estimation"""
    symbol: str
    timestamp: datetime
    price: Decimal
    bid: Decimal
    ask: Decimal
    volume: Decimal
    avg_daily_volume: Decimal
    volatility: float
    spread_bps: float  # Spread in basis points

    def __post_init__(self):
        self.price = Decimal(str(self.price))
        self.bid = Decimal(str(self.bid))
        self.ask = Decimal(str(self.ask))
        self.volume = Decimal(str(self.volume))
        self.avg_daily_volume = Decimal(str(self.avg_daily_volume))

    @property
    def spread(self) -> Decimal:
        """Absolute spread"""
        return self.ask - self.bid

    @property
    def mid_price(self) -> Decimal:
        """Mid price"""
        return (self.bid + self.ask) / 2


@dataclass
class SlippageEstimate:
    """Slippage estimation result"""
    symbol: str
    side: str
    quantity: Decimal
    reference_price: Decimal
    estimated_price: Decimal
    slippage_bps: float  # Slippage in basis points
    slippage_amount: Decimal
    confidence: float  # 0-1 confidence in estimate
    model: str
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.quantity = Decimal(str(self.quantity))
        self.reference_price = Decimal(str(self.reference_price))
        self.estimated_price = Decimal(str(self.estimated_price))
        self.slippage_amount = Decimal(str(self.slippage_amount))

    @property
    def total_cost(self) -> Decimal:
        """Total execution cost including slippage"""
        return self.estimated_price * self.quantity


@dataclass
class HistoricalSlippage:
    """Historical slippage observation"""
    timestamp: datetime
    symbol: str
    side: str
    quantity: Decimal
    reference_price: Decimal
    execution_price: Decimal
    slippage_bps: float
    market_volume: Decimal
    spread_bps: float
    volatility: float

    def __post_init__(self):
        self.quantity = Decimal(str(self.quantity))
        self.reference_price = Decimal(str(self.reference_price))
        self.execution_price = Decimal(str(self.execution_price))
        self.market_volume = Decimal(str(self.market_volume))


class SlippageModel(ABC):
    """Base class for slippage models"""

    def __init__(self, name: str):
        """Initialize slippage model"""
        self.name = name

    @abstractmethod
    def estimate(
        self,
        market_state: MarketState,
        side: str,
        quantity: Decimal,
        urgency: float = 0.5
    ) -> SlippageEstimate:
        """
        Estimate slippage for an order

        Args:
            market_state: Current market state
            side: "buy" or "sell"
            quantity: Order quantity
            urgency: Urgency level (0 = passive, 1 = aggressive)

        Returns:
            SlippageEstimate
        """
        pass


class LinearImpactModel(SlippageModel):
    """
    Linear market impact model

    Slippage is proportional to order size relative to market volume
    Impact = alpha * (quantity / daily_volume)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        temporary_impact_ratio: float = 0.6,
        permanent_impact_ratio: float = 0.4
    ):
        """
        Initialize linear impact model

        Args:
            alpha: Impact coefficient (typically 0.05 - 0.3)
            temporary_impact_ratio: Fraction of impact that is temporary
            permanent_impact_ratio: Fraction of impact that is permanent
        """
        super().__init__("LinearImpact")
        self.alpha = alpha
        self.temporary_impact_ratio = temporary_impact_ratio
        self.permanent_impact_ratio = permanent_impact_ratio

    def estimate(
        self,
        market_state: MarketState,
        side: str,
        quantity: Decimal,
        urgency: float = 0.5
    ) -> SlippageEstimate:
        """Estimate slippage using linear impact model"""
        # Calculate participation rate
        participation_rate = float(quantity / market_state.avg_daily_volume)

        # Base impact (in basis points)
        base_impact_bps = self.alpha * participation_rate * 10000

        # Temporary impact (depends on urgency)
        temporary_impact = base_impact_bps * self.temporary_impact_ratio * urgency

        # Permanent impact
        permanent_impact = base_impact_bps * self.permanent_impact_ratio

        # Total impact
        total_impact_bps = temporary_impact + permanent_impact

        # Adjust for volatility
        volatility_adjustment = market_state.volatility * 100  # Scale volatility
        total_impact_bps *= (1 + volatility_adjustment)

        # Calculate slippage direction
        if side.lower() == "buy":
            slippage_bps = total_impact_bps  # Positive for buys
            estimated_price = market_state.price * (1 + total_impact_bps / 10000)
        else:
            slippage_bps = -total_impact_bps  # Negative for sells
            estimated_price = market_state.price * (1 - total_impact_bps / 10000)

        slippage_amount = abs(estimated_price - market_state.price) * quantity

        return SlippageEstimate(
            symbol=market_state.symbol,
            side=side,
            quantity=quantity,
            reference_price=market_state.price,
            estimated_price=estimated_price,
            slippage_bps=slippage_bps,
            slippage_amount=slippage_amount,
            confidence=0.7,
            model=self.name,
            components={
                "temporary_impact_bps": temporary_impact,
                "permanent_impact_bps": permanent_impact,
                "participation_rate": participation_rate,
                "volatility_adjustment": volatility_adjustment
            }
        )


class SquareRootImpactModel(SlippageModel):
    """
    Square-root market impact model

    More realistic for large orders
    Impact = alpha * sqrt(quantity / daily_volume)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        temporary_impact_ratio: float = 0.6,
        permanent_impact_ratio: float = 0.4
    ):
        """
        Initialize square-root impact model

        Args:
            alpha: Impact coefficient (typically 0.2 - 0.5)
            temporary_impact_ratio: Fraction of impact that is temporary
            permanent_impact_ratio: Fraction of impact that is permanent
        """
        super().__init__("SquareRootImpact")
        self.alpha = alpha
        self.temporary_impact_ratio = temporary_impact_ratio
        self.permanent_impact_ratio = permanent_impact_ratio

    def estimate(
        self,
        market_state: MarketState,
        side: str,
        quantity: Decimal,
        urgency: float = 0.5
    ) -> SlippageEstimate:
        """Estimate slippage using square-root impact model"""
        # Calculate participation rate
        participation_rate = float(quantity / market_state.avg_daily_volume)

        # Square-root impact
        base_impact_bps = self.alpha * np.sqrt(participation_rate) * 10000

        # Temporary impact (depends on urgency)
        temporary_impact = base_impact_bps * self.temporary_impact_ratio * urgency

        # Permanent impact
        permanent_impact = base_impact_bps * self.permanent_impact_ratio

        # Total impact
        total_impact_bps = temporary_impact + permanent_impact

        # Adjust for volatility
        volatility_adjustment = market_state.volatility * 50
        total_impact_bps *= (1 + volatility_adjustment)

        # Calculate slippage direction
        if side.lower() == "buy":
            slippage_bps = total_impact_bps
            estimated_price = market_state.price * (1 + total_impact_bps / 10000)
        else:
            slippage_bps = -total_impact_bps
            estimated_price = market_state.price * (1 - total_impact_bps / 10000)

        slippage_amount = abs(estimated_price - market_state.price) * quantity

        return SlippageEstimate(
            symbol=market_state.symbol,
            side=side,
            quantity=quantity,
            reference_price=market_state.price,
            estimated_price=estimated_price,
            slippage_bps=slippage_bps,
            slippage_amount=slippage_amount,
            confidence=0.75,
            model=self.name,
            components={
                "temporary_impact_bps": temporary_impact,
                "permanent_impact_bps": permanent_impact,
                "participation_rate": participation_rate,
                "volatility_adjustment": volatility_adjustment
            }
        )


class VolumeBasedSlippageModel(SlippageModel):
    """
    Volume-based slippage model

    Estimates slippage based on current market volume conditions
    """

    def __init__(
        self,
        low_volume_threshold: float = 0.5,
        high_volume_threshold: float = 2.0,
        base_slippage_bps: float = 5.0
    ):
        """
        Initialize volume-based model

        Args:
            low_volume_threshold: Threshold for low volume (relative to avg)
            high_volume_threshold: Threshold for high volume (relative to avg)
            base_slippage_bps: Base slippage in basis points
        """
        super().__init__("VolumeBasedSlippage")
        self.low_volume_threshold = low_volume_threshold
        self.high_volume_threshold = high_volume_threshold
        self.base_slippage_bps = base_slippage_bps

    def estimate(
        self,
        market_state: MarketState,
        side: str,
        quantity: Decimal,
        urgency: float = 0.5
    ) -> SlippageEstimate:
        """Estimate slippage based on volume conditions"""
        # Calculate current volume relative to average
        volume_ratio = float(market_state.volume / market_state.avg_daily_volume)

        # Adjust slippage based on volume
        if volume_ratio < self.low_volume_threshold:
            # Low volume - higher slippage
            volume_factor = 2.0 + (self.low_volume_threshold - volume_ratio)
        elif volume_ratio > self.high_volume_threshold:
            # High volume - lower slippage
            volume_factor = 0.5
        else:
            # Normal volume
            volume_factor = 1.0

        # Calculate participation rate
        participation_rate = float(quantity / market_state.avg_daily_volume)

        # Base slippage adjusted by volume and participation
        slippage_bps = (
            self.base_slippage_bps *
            volume_factor *
            (1 + participation_rate * 10) *
            (0.5 + urgency)
        )

        # Adjust for spread
        spread_factor = market_state.spread_bps / 10  # Normalize
        slippage_bps += spread_factor

        # Calculate price impact
        if side.lower() == "buy":
            estimated_price = market_state.price * (1 + slippage_bps / 10000)
        else:
            slippage_bps = -slippage_bps
            estimated_price = market_state.price * (1 - abs(slippage_bps) / 10000)

        slippage_amount = abs(estimated_price - market_state.price) * quantity

        return SlippageEstimate(
            symbol=market_state.symbol,
            side=side,
            quantity=quantity,
            reference_price=market_state.price,
            estimated_price=estimated_price,
            slippage_bps=slippage_bps,
            slippage_amount=slippage_amount,
            confidence=0.6,
            model=self.name,
            components={
                "volume_ratio": volume_ratio,
                "volume_factor": volume_factor,
                "participation_rate": participation_rate,
                "spread_factor": spread_factor
            }
        )


class SpreadBasedSlippageModel(SlippageModel):
    """
    Spread-based slippage model

    Estimates slippage primarily from bid-ask spread
    """

    def __init__(
        self,
        spread_crossing_probability: float = 0.5,
        depth_decay_factor: float = 0.8
    ):
        """
        Initialize spread-based model

        Args:
            spread_crossing_probability: Probability of crossing spread
            depth_decay_factor: Factor for price impact beyond best bid/ask
        """
        super().__init__("SpreadBasedSlippage")
        self.spread_crossing_probability = spread_crossing_probability
        self.depth_decay_factor = depth_decay_factor

    def estimate(
        self,
        market_state: MarketState,
        side: str,
        quantity: Decimal,
        urgency: float = 0.5
    ) -> SlippageEstimate:
        """Estimate slippage based on spread"""
        # Base cost: crossing the spread
        spread = market_state.spread
        mid_price = market_state.mid_price

        # Probability of crossing spread depends on urgency
        cross_probability = self.spread_crossing_probability * urgency

        # Base spread cost
        if side.lower() == "buy":
            # Buying at ask
            base_price = market_state.ask
            spread_cost_bps = float((base_price - mid_price) / mid_price * 10000)
        else:
            # Selling at bid
            base_price = market_state.bid
            spread_cost_bps = -float((mid_price - base_price) / mid_price * 10000)

        # Additional impact for large orders (walking the book)
        participation_rate = float(quantity / market_state.avg_daily_volume)
        depth_impact_bps = 0

        if participation_rate > 0.01:  # > 1% of daily volume
            # Estimate additional levels needed
            additional_levels = np.log1p(participation_rate * 100)
            depth_impact_bps = (
                float(spread_cost_bps) *
                additional_levels *
                self.depth_decay_factor
            )

        # Total slippage
        total_slippage_bps = (spread_cost_bps * cross_probability) + depth_impact_bps

        # Adjust for volatility
        volatility_adjustment = market_state.volatility * 20
        total_slippage_bps *= (1 + volatility_adjustment)

        # Calculate estimated price
        if side.lower() == "buy":
            estimated_price = market_state.price * (1 + total_slippage_bps / 10000)
        else:
            estimated_price = market_state.price * (1 - abs(total_slippage_bps) / 10000)

        slippage_amount = abs(estimated_price - market_state.price) * quantity

        return SlippageEstimate(
            symbol=market_state.symbol,
            side=side,
            quantity=quantity,
            reference_price=market_state.price,
            estimated_price=estimated_price,
            slippage_bps=total_slippage_bps,
            slippage_amount=slippage_amount,
            confidence=0.8,
            model=self.name,
            components={
                "spread_cost_bps": spread_cost_bps,
                "depth_impact_bps": depth_impact_bps,
                "cross_probability": cross_probability,
                "participation_rate": participation_rate,
                "volatility_adjustment": volatility_adjustment
            }
        )


class HistoricalSlippageModel(SlippageModel):
    """
    Historical slippage model

    Uses historical execution data to predict future slippage
    """

    def __init__(
        self,
        lookback_periods: int = 100,
        min_samples: int = 10
    ):
        """
        Initialize historical model

        Args:
            lookback_periods: Number of historical periods to consider
            min_samples: Minimum samples required for estimate
        """
        super().__init__("HistoricalSlippage")
        self.lookback_periods = lookback_periods
        self.min_samples = min_samples

        # Store historical observations
        self.history: Dict[str, deque] = {}

    def add_observation(self, observation: HistoricalSlippage):
        """Add a historical slippage observation"""
        if observation.symbol not in self.history:
            self.history[observation.symbol] = deque(maxlen=self.lookback_periods)

        self.history[observation.symbol].append(observation)

    def estimate(
        self,
        market_state: MarketState,
        side: str,
        quantity: Decimal,
        urgency: float = 0.5
    ) -> SlippageEstimate:
        """Estimate slippage using historical data"""
        symbol_history = self.history.get(market_state.symbol, [])

        if len(symbol_history) < self.min_samples:
            # Insufficient data - use fallback model
            fallback = SpreadBasedSlippageModel()
            estimate = fallback.estimate(market_state, side, quantity, urgency)
            estimate.model = f"{self.name} (fallback)"
            estimate.confidence = 0.3
            return estimate

        # Filter for similar conditions
        similar_observations = []
        participation_rate = float(quantity / market_state.avg_daily_volume)

        for obs in symbol_history:
            # Consider observations with similar:
            # - Side
            # - Participation rate (within 2x)
            # - Volatility (within 2x)
            obs_participation = float(obs.quantity / market_state.avg_daily_volume)

            if (
                obs.side == side and
                0.5 * participation_rate <= obs_participation <= 2.0 * participation_rate and
                0.5 * market_state.volatility <= obs.volatility <= 2.0 * market_state.volatility
            ):
                similar_observations.append(obs)

        if len(similar_observations) < self.min_samples:
            # Use all observations for the side
            similar_observations = [
                obs for obs in symbol_history
                if obs.side == side
            ]

        if not similar_observations:
            # Fallback
            fallback = SpreadBasedSlippageModel()
            estimate = fallback.estimate(market_state, side, quantity, urgency)
            estimate.model = f"{self.name} (fallback)"
            estimate.confidence = 0.3
            return estimate

        # Calculate statistics from historical data
        slippages = [obs.slippage_bps for obs in similar_observations]
        avg_slippage = np.mean(slippages)
        std_slippage = np.std(slippages)

        # Adjust for current conditions
        # If current spread is wider than historical average
        avg_historical_spread = np.mean([obs.spread_bps for obs in similar_observations])
        spread_adjustment = market_state.spread_bps / avg_historical_spread

        estimated_slippage_bps = avg_slippage * spread_adjustment

        # Adjust for urgency
        urgency_adjustment = 0.5 + (urgency * 0.5)
        estimated_slippage_bps *= urgency_adjustment

        # Calculate estimated price
        if side.lower() == "buy":
            estimated_price = market_state.price * (1 + estimated_slippage_bps / 10000)
        else:
            estimated_price = market_state.price * (1 - abs(estimated_slippage_bps) / 10000)

        slippage_amount = abs(estimated_price - market_state.price) * quantity

        # Confidence based on number of similar observations and consistency
        confidence = min(0.95, 0.5 + (len(similar_observations) / self.lookback_periods))
        if std_slippage > abs(avg_slippage):
            confidence *= 0.7  # Reduce confidence if high variance

        return SlippageEstimate(
            symbol=market_state.symbol,
            side=side,
            quantity=quantity,
            reference_price=market_state.price,
            estimated_price=estimated_price,
            slippage_bps=estimated_slippage_bps,
            slippage_amount=slippage_amount,
            confidence=confidence,
            model=self.name,
            components={
                "avg_historical_slippage_bps": avg_slippage,
                "std_slippage_bps": std_slippage,
                "similar_samples": len(similar_observations),
                "spread_adjustment": spread_adjustment,
                "urgency_adjustment": urgency_adjustment
            }
        )


class EnsembleSlippageModel(SlippageModel):
    """
    Ensemble slippage model

    Combines multiple models for more robust estimates
    """

    def __init__(
        self,
        models: Optional[List[SlippageModel]] = None,
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble model

        Args:
            models: List of slippage models to combine
            weights: Weights for each model (defaults to equal weights)
        """
        super().__init__("EnsembleSlippage")

        if models is None:
            # Default ensemble
            self.models = [
                LinearImpactModel(),
                SquareRootImpactModel(),
                VolumeBasedSlippageModel(),
                SpreadBasedSlippageModel()
            ]
        else:
            self.models = models

        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def estimate(
        self,
        market_state: MarketState,
        side: str,
        quantity: Decimal,
        urgency: float = 0.5
    ) -> SlippageEstimate:
        """Estimate slippage using ensemble of models"""
        # Get estimates from all models
        estimates = []
        for model in self.models:
            try:
                estimate = model.estimate(market_state, side, quantity, urgency)
                estimates.append(estimate)
            except Exception as e:
                logger.warning(f"Error in {model.name}: {e}")

        if not estimates:
            raise ValueError("All models failed to produce estimates")

        # Weighted average of slippage estimates
        # Weight by both model weight and confidence
        weighted_slippage = 0.0
        total_weight = 0.0

        for estimate, model_weight in zip(estimates, self.weights):
            weight = model_weight * estimate.confidence
            weighted_slippage += estimate.slippage_bps * weight
            total_weight += weight

        avg_slippage_bps = weighted_slippage / total_weight

        # Calculate ensemble estimated price
        if side.lower() == "buy":
            estimated_price = market_state.price * (1 + avg_slippage_bps / 10000)
        else:
            estimated_price = market_state.price * (1 - abs(avg_slippage_bps) / 10000)

        slippage_amount = abs(estimated_price - market_state.price) * quantity

        # Ensemble confidence (average of model confidences)
        avg_confidence = np.mean([e.confidence for e in estimates])

        # Calculate variance of estimates as uncertainty measure
        slippage_values = [e.slippage_bps for e in estimates]
        variance = np.var(slippage_values)

        # Reduce confidence if high disagreement between models
        if variance > 100:  # > 10 bps standard deviation
            avg_confidence *= 0.8

        return SlippageEstimate(
            symbol=market_state.symbol,
            side=side,
            quantity=quantity,
            reference_price=market_state.price,
            estimated_price=estimated_price,
            slippage_bps=avg_slippage_bps,
            slippage_amount=slippage_amount,
            confidence=avg_confidence,
            model=self.name,
            components={
                "model_estimates": {
                    e.model: e.slippage_bps for e in estimates
                },
                "variance": variance,
                "num_models": len(estimates)
            },
            metadata={
                "individual_estimates": estimates
            }
        )


class SlippageAnalyzer:
    """
    Slippage analyzer for performance tracking and model selection
    """

    def __init__(self):
        """Initialize slippage analyzer"""
        self.observations: List[HistoricalSlippage] = []
        self.model_errors: Dict[str, List[float]] = {}

    def add_observation(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        reference_price: Decimal,
        execution_price: Decimal,
        market_volume: Decimal,
        spread_bps: float,
        volatility: float
    ):
        """Add a slippage observation"""
        slippage_bps = float(
            (execution_price - reference_price) / reference_price * 10000
        )

        obs = HistoricalSlippage(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            reference_price=reference_price,
            execution_price=execution_price,
            slippage_bps=slippage_bps,
            market_volume=market_volume,
            spread_bps=spread_bps,
            volatility=volatility
        )

        self.observations.append(obs)

    def evaluate_model(
        self,
        model: SlippageModel,
        observations: Optional[List[HistoricalSlippage]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a slippage model against historical observations

        Returns:
            Dictionary with error metrics
        """
        if observations is None:
            observations = self.observations

        if not observations:
            return {}

        errors = []
        absolute_errors = []

        for obs in observations:
            # Create market state
            market_state = MarketState(
                symbol=obs.symbol,
                timestamp=obs.timestamp,
                price=obs.reference_price,
                bid=obs.reference_price * Decimal("0.9995"),
                ask=obs.reference_price * Decimal("1.0005"),
                volume=obs.market_volume,
                avg_daily_volume=obs.market_volume * 390,
                volatility=obs.volatility,
                spread_bps=obs.spread_bps
            )

            # Get model estimate
            try:
                estimate = model.estimate(market_state, obs.side, obs.quantity)
                error = estimate.slippage_bps - obs.slippage_bps
                errors.append(error)
                absolute_errors.append(abs(error))
            except Exception as e:
                logger.warning(f"Error evaluating model: {e}")

        if not errors:
            return {}

        return {
            "mean_error": np.mean(errors),
            "mean_absolute_error": np.mean(absolute_errors),
            "rmse": np.sqrt(np.mean(np.square(errors))),
            "median_absolute_error": np.median(absolute_errors),
            "num_observations": len(errors)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get slippage statistics"""
        if not self.observations:
            return {}

        slippages = [obs.slippage_bps for obs in self.observations]

        return {
            "total_observations": len(self.observations),
            "mean_slippage_bps": np.mean(slippages),
            "median_slippage_bps": np.median(slippages),
            "std_slippage_bps": np.std(slippages),
            "min_slippage_bps": np.min(slippages),
            "max_slippage_bps": np.max(slippages),
            "percentiles": {
                "25th": np.percentile(slippages, 25),
                "50th": np.percentile(slippages, 50),
                "75th": np.percentile(slippages, 75),
                "95th": np.percentile(slippages, 95)
            }
        }
