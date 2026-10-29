"""
Portfolio Rebalancer Module

Implements various portfolio rebalancing strategies:
- Threshold-based rebalancing
- Periodic rebalancing
- Volatility-based rebalancing
- Tax-aware rebalancing
- Transaction cost optimization
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class RebalanceMethod(Enum):
    """Supported rebalancing methods"""
    THRESHOLD = "threshold"
    PERIODIC = "periodic"
    VOLATILITY = "volatility"
    TAX_AWARE = "tax_aware"
    COST_OPTIMIZED = "cost_optimized"


@dataclass
class Position:
    """Individual position in portfolio"""
    symbol: str
    quantity: float
    current_price: float
    cost_basis: float
    acquisition_date: datetime

    @property
    def market_value(self) -> float:
        """Current market value"""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        return (self.current_price - self.cost_basis) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss percentage"""
        if self.cost_basis == 0:
            return 0.0
        return (self.current_price - self.cost_basis) / self.cost_basis


@dataclass
class RebalanceConfig:
    """Configuration for rebalancing"""
    # Threshold-based parameters
    threshold_pct: float = 0.05  # 5% deviation triggers rebalance
    threshold_absolute: Optional[float] = None  # Absolute deviation in dollars

    # Periodic parameters
    rebalance_frequency: str = "monthly"  # 'daily', 'weekly', 'monthly', 'quarterly', 'annual'
    last_rebalance_date: Optional[datetime] = None

    # Volatility-based parameters
    volatility_threshold: float = 0.20  # 20% annualized volatility
    volatility_window: int = 20  # Days for volatility calculation

    # Tax parameters
    tax_rate_short_term: float = 0.37  # Short-term capital gains tax
    tax_rate_long_term: float = 0.20  # Long-term capital gains tax
    holding_period_days: int = 365  # Days for long-term status
    tax_loss_harvesting: bool = True  # Enable tax loss harvesting
    wash_sale_days: int = 30  # Wash sale rule period

    # Transaction cost parameters
    commission_per_trade: float = 0.0  # Commission per trade
    commission_pct: float = 0.001  # 0.1% commission
    slippage_pct: float = 0.001  # 0.1% slippage
    min_trade_size: float = 100.0  # Minimum trade size in dollars

    # Portfolio constraints
    max_turnover: Optional[float] = None  # Maximum turnover per rebalance
    min_weight: float = 0.0
    max_weight: float = 1.0

    # Cash management
    target_cash_pct: float = 0.02  # 2% cash buffer
    min_cash: float = 1000.0  # Minimum cash balance


@dataclass
class RebalanceResult:
    """Result of rebalancing operation"""
    trades: Dict[str, float]  # {symbol: quantity} (positive = buy, negative = sell)
    target_weights: np.ndarray
    current_weights: np.ndarray
    estimated_cost: float
    estimated_tax: float
    turnover: float
    method: str
    timestamp: datetime
    success: bool
    message: str
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            'trades': self.trades,
            'target_weights': self.target_weights.tolist(),
            'current_weights': self.current_weights.tolist(),
            'estimated_cost': float(self.estimated_cost),
            'estimated_tax': float(self.estimated_tax),
            'turnover': float(self.turnover),
            'method': self.method,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'message': self.message,
            'metadata': self.metadata
        }


class PortfolioRebalancer:
    """
    Advanced portfolio rebalancer with multiple strategies
    """

    def __init__(
        self,
        config: Optional[RebalanceConfig] = None
    ):
        """
        Initialize portfolio rebalancer

        Args:
            config: Rebalancing configuration
        """
        self.config = config or RebalanceConfig()
        logger.info("Initialized PortfolioRebalancer")

    def check_rebalance_needed(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        method: Union[RebalanceMethod, str],
        **kwargs
    ) -> bool:
        """
        Check if rebalancing is needed

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            method: Rebalancing method
            **kwargs: Additional method-specific parameters

        Returns:
            True if rebalancing is needed
        """
        if isinstance(method, str):
            method = RebalanceMethod(method)

        if method == RebalanceMethod.THRESHOLD:
            return self._check_threshold_rebalance(current_weights, target_weights)

        elif method == RebalanceMethod.PERIODIC:
            return self._check_periodic_rebalance(kwargs.get('current_date'))

        elif method == RebalanceMethod.VOLATILITY:
            return self._check_volatility_rebalance(
                kwargs.get('returns'),
                kwargs.get('current_date')
            )

        elif method in [RebalanceMethod.TAX_AWARE, RebalanceMethod.COST_OPTIMIZED]:
            # Always check if optimization can improve
            return True

        return False

    def rebalance(
        self,
        current_positions: Dict[str, Position],
        target_weights: np.ndarray,
        symbols: List[str],
        method: Union[RebalanceMethod, str],
        total_value: float,
        **kwargs
    ) -> RebalanceResult:
        """
        Execute portfolio rebalancing

        Args:
            current_positions: Current portfolio positions
            target_weights: Target portfolio weights
            symbols: List of symbols in order matching target_weights
            method: Rebalancing method
            total_value: Total portfolio value
            **kwargs: Additional method-specific parameters

        Returns:
            RebalanceResult with trade instructions
        """
        try:
            if isinstance(method, str):
                method = RebalanceMethod(method)

            # Calculate current weights
            current_weights = self._calculate_current_weights(
                current_positions, symbols, total_value
            )

            # Route to appropriate rebalancing method
            if method == RebalanceMethod.THRESHOLD:
                trades, metadata = self._threshold_rebalance(
                    current_positions, current_weights, target_weights,
                    symbols, total_value
                )

            elif method == RebalanceMethod.PERIODIC:
                trades, metadata = self._periodic_rebalance(
                    current_positions, current_weights, target_weights,
                    symbols, total_value
                )

            elif method == RebalanceMethod.VOLATILITY:
                trades, metadata = self._volatility_rebalance(
                    current_positions, current_weights, target_weights,
                    symbols, total_value, **kwargs
                )

            elif method == RebalanceMethod.TAX_AWARE:
                trades, metadata = self._tax_aware_rebalance(
                    current_positions, current_weights, target_weights,
                    symbols, total_value, **kwargs
                )

            elif method == RebalanceMethod.COST_OPTIMIZED:
                trades, metadata = self._cost_optimized_rebalance(
                    current_positions, current_weights, target_weights,
                    symbols, total_value, **kwargs
                )

            else:
                raise ValueError(f"Unknown rebalancing method: {method}")

            # Calculate costs and turnover
            estimated_cost = self._calculate_transaction_costs(trades, current_positions)
            estimated_tax = self._calculate_tax_impact(trades, current_positions)
            turnover = self._calculate_turnover(trades, total_value)

            return RebalanceResult(
                trades=trades,
                target_weights=target_weights,
                current_weights=current_weights,
                estimated_cost=estimated_cost,
                estimated_tax=estimated_tax,
                turnover=turnover,
                method=method.value,
                timestamp=datetime.now(),
                success=True,
                message="Rebalancing completed successfully",
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Rebalancing failed: {str(e)}", exc_info=True)
            return RebalanceResult(
                trades={},
                target_weights=target_weights,
                current_weights=np.zeros_like(target_weights),
                estimated_cost=0.0,
                estimated_tax=0.0,
                turnover=0.0,
                method=method.value if isinstance(method, RebalanceMethod) else str(method),
                timestamp=datetime.now(),
                success=False,
                message=f"Rebalancing failed: {str(e)}"
            )

    async def rebalance_async(
        self,
        current_positions: Dict[str, Position],
        target_weights: np.ndarray,
        symbols: List[str],
        method: Union[RebalanceMethod, str],
        total_value: float,
        **kwargs
    ) -> RebalanceResult:
        """Async version of rebalance"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.rebalance,
            current_positions,
            target_weights,
            symbols,
            method,
            total_value,
            **kwargs
        )

    def _calculate_current_weights(
        self,
        positions: Dict[str, Position],
        symbols: List[str],
        total_value: float
    ) -> np.ndarray:
        """Calculate current portfolio weights"""
        weights = np.zeros(len(symbols))

        for i, symbol in enumerate(symbols):
            if symbol in positions:
                weights[i] = positions[symbol].market_value / total_value

        return weights

    def _check_threshold_rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> bool:
        """Check if threshold-based rebalancing is needed"""
        deviations = np.abs(current_weights - target_weights)
        max_deviation = np.max(deviations)

        return max_deviation > self.config.threshold_pct

    def _check_periodic_rebalance(
        self,
        current_date: Optional[datetime] = None
    ) -> bool:
        """Check if periodic rebalancing is needed"""
        if self.config.last_rebalance_date is None:
            return True

        current_date = current_date or datetime.now()

        frequency_map = {
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'monthly': timedelta(days=30),
            'quarterly': timedelta(days=90),
            'annual': timedelta(days=365)
        }

        delta = frequency_map.get(self.config.rebalance_frequency, timedelta(days=30))
        return (current_date - self.config.last_rebalance_date) >= delta

    def _check_volatility_rebalance(
        self,
        returns: Optional[pd.DataFrame] = None,
        current_date: Optional[datetime] = None
    ) -> bool:
        """Check if volatility-based rebalancing is needed"""
        if returns is None or returns.empty:
            return False

        # Calculate recent volatility
        recent_returns = returns.tail(self.config.volatility_window)
        volatility = recent_returns.std() * np.sqrt(252)  # Annualized

        avg_volatility = volatility.mean()
        return avg_volatility > self.config.volatility_threshold

    def _threshold_rebalance(
        self,
        positions: Dict[str, Position],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        symbols: List[str],
        total_value: float
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Threshold-based rebalancing

        Only rebalance assets that deviate beyond threshold
        """
        trades = {}
        rebalanced_assets = []

        for i, symbol in enumerate(symbols):
            deviation = abs(current_weights[i] - target_weights[i])

            if deviation > self.config.threshold_pct:
                # Calculate target position value
                target_value = target_weights[i] * total_value

                # Calculate current position value
                current_value = positions[symbol].market_value if symbol in positions else 0.0

                # Calculate trade value
                trade_value = target_value - current_value

                # Convert to quantity
                current_price = positions[symbol].current_price if symbol in positions else 1.0
                trade_quantity = trade_value / current_price

                # Check minimum trade size
                if abs(trade_value) >= self.config.min_trade_size:
                    trades[symbol] = trade_quantity
                    rebalanced_assets.append(symbol)

        metadata = {
            'rebalanced_assets': rebalanced_assets,
            'threshold_pct': self.config.threshold_pct
        }

        return trades, metadata

    def _periodic_rebalance(
        self,
        positions: Dict[str, Position],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        symbols: List[str],
        total_value: float
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Periodic rebalancing

        Fully rebalance to target weights
        """
        trades = {}

        for i, symbol in enumerate(symbols):
            # Calculate target position value
            target_value = target_weights[i] * total_value

            # Calculate current position value
            current_value = positions[symbol].market_value if symbol in positions else 0.0

            # Calculate trade value
            trade_value = target_value - current_value

            # Convert to quantity
            current_price = positions[symbol].current_price if symbol in positions else 1.0
            trade_quantity = trade_value / current_price

            # Check minimum trade size
            if abs(trade_value) >= self.config.min_trade_size:
                trades[symbol] = trade_quantity

        metadata = {
            'frequency': self.config.rebalance_frequency,
            'full_rebalance': True
        }

        return trades, metadata

    def _volatility_rebalance(
        self,
        positions: Dict[str, Position],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        symbols: List[str],
        total_value: float,
        **kwargs
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Volatility-based rebalancing

        Adjust rebalancing intensity based on market volatility
        """
        returns = kwargs.get('returns')

        if returns is not None and not returns.empty:
            # Calculate recent volatility
            recent_returns = returns.tail(self.config.volatility_window)
            volatility = recent_returns.std() * np.sqrt(252)
            avg_volatility = volatility.mean()

            # Scale rebalancing based on volatility
            # Higher volatility -> more aggressive rebalancing
            volatility_multiplier = min(avg_volatility / self.config.volatility_threshold, 2.0)
        else:
            volatility_multiplier = 1.0

        # Adjust target weights based on volatility
        adjusted_threshold = self.config.threshold_pct / volatility_multiplier

        trades = {}
        for i, symbol in enumerate(symbols):
            deviation = abs(current_weights[i] - target_weights[i])

            if deviation > adjusted_threshold:
                target_value = target_weights[i] * total_value
                current_value = positions[symbol].market_value if symbol in positions else 0.0
                trade_value = target_value - current_value

                current_price = positions[symbol].current_price if symbol in positions else 1.0
                trade_quantity = trade_value / current_price

                if abs(trade_value) >= self.config.min_trade_size:
                    trades[symbol] = trade_quantity

        metadata = {
            'volatility_multiplier': volatility_multiplier,
            'adjusted_threshold': adjusted_threshold,
            'avg_volatility': avg_volatility if returns is not None else None
        }

        return trades, metadata

    def _tax_aware_rebalance(
        self,
        positions: Dict[str, Position],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        symbols: List[str],
        total_value: float,
        **kwargs
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Tax-aware rebalancing

        Minimize tax impact while rebalancing
        """
        current_date = kwargs.get('current_date', datetime.now())

        # Categorize positions by tax status
        short_term_positions = []
        long_term_positions = []
        loss_positions = []

        for symbol, position in positions.items():
            days_held = (current_date - position.acquisition_date).days

            if position.unrealized_pnl < 0:
                loss_positions.append(symbol)
            elif days_held < self.config.holding_period_days:
                short_term_positions.append(symbol)
            else:
                long_term_positions.append(symbol)

        trades = {}
        tax_savings = 0.0

        # Priority 1: Tax loss harvesting
        if self.config.tax_loss_harvesting:
            for symbol in loss_positions:
                idx = symbols.index(symbol)
                position = positions[symbol]

                # Sell losing positions to realize losses
                if current_weights[idx] > target_weights[idx]:
                    target_value = target_weights[idx] * total_value
                    current_value = position.market_value
                    trade_value = target_value - current_value
                    trade_quantity = trade_value / position.current_price

                    if abs(trade_value) >= self.config.min_trade_size:
                        trades[symbol] = trade_quantity
                        # Tax savings from realized losses
                        loss_amount = abs(trade_quantity * position.unrealized_pnl / position.quantity)
                        tax_savings += loss_amount * self.config.tax_rate_short_term

        # Priority 2: Reduce short-term positions (defer sales when possible)
        for symbol in short_term_positions:
            if symbol in trades:
                continue

            idx = symbols.index(symbol)
            position = positions[symbol]

            # Only reduce if significantly over target
            if current_weights[idx] > target_weights[idx] + 2 * self.config.threshold_pct:
                target_value = target_weights[idx] * total_value
                current_value = position.market_value
                trade_value = target_value - current_value
                trade_quantity = trade_value / position.current_price

                if abs(trade_value) >= self.config.min_trade_size:
                    trades[symbol] = trade_quantity

        # Priority 3: Sell long-term positions if needed
        for symbol in long_term_positions:
            if symbol in trades:
                continue

            idx = symbols.index(symbol)
            position = positions[symbol]

            if current_weights[idx] > target_weights[idx]:
                target_value = target_weights[idx] * total_value
                current_value = position.market_value
                trade_value = target_value - current_value
                trade_quantity = trade_value / position.current_price

                if abs(trade_value) >= self.config.min_trade_size:
                    trades[symbol] = trade_quantity

        # Add buys to reach target weights
        for i, symbol in enumerate(symbols):
            if symbol in trades and trades[symbol] > 0:
                continue

            if current_weights[i] < target_weights[i]:
                target_value = target_weights[i] * total_value
                current_value = positions[symbol].market_value if symbol in positions else 0.0
                trade_value = target_value - current_value

                current_price = positions[symbol].current_price if symbol in positions else 1.0
                trade_quantity = trade_value / current_price

                if abs(trade_value) >= self.config.min_trade_size:
                    trades[symbol] = trade_quantity

        metadata = {
            'tax_loss_harvesting': self.config.tax_loss_harvesting,
            'loss_positions': loss_positions,
            'short_term_positions': short_term_positions,
            'long_term_positions': long_term_positions,
            'estimated_tax_savings': tax_savings
        }

        return trades, metadata

    def _cost_optimized_rebalance(
        self,
        positions: Dict[str, Position],
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        symbols: List[str],
        total_value: float,
        **kwargs
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Transaction cost optimized rebalancing

        Minimize transaction costs while achieving target weights
        """
        n_assets = len(symbols)

        # Decision variables: trade quantities
        trades_var = cp.Variable(n_assets)

        # Calculate prices
        prices = np.array([
            positions[symbol].current_price if symbol in positions else 1.0
            for symbol in symbols
        ])

        # Current values
        current_values = np.array([
            positions[symbol].market_value if symbol in positions else 0.0
            for symbol in symbols
        ])

        # Objective: Minimize tracking error + transaction costs
        new_values = current_values + trades_var * prices
        new_weights = new_values / total_value

        tracking_error = cp.sum_squares(new_weights - target_weights)

        # Transaction costs
        trade_costs = (
            self.config.commission_pct * cp.sum(cp.abs(trades_var) * prices) +
            self.config.slippage_pct * cp.sum(cp.abs(trades_var) * prices)
        )

        # Weight tracking error vs costs
        cost_weight = kwargs.get('cost_weight', 1000.0)
        objective = cp.Minimize(tracking_error + trade_costs / cost_weight)

        # Constraints
        constraints = [
            cp.sum(new_values) <= total_value * (1 + self.config.target_cash_pct),
            new_weights >= self.config.min_weight,
            new_weights <= self.config.max_weight
        ]

        # Turnover constraint
        if self.config.max_turnover is not None:
            turnover = cp.sum(cp.abs(trades_var) * prices) / total_value
            constraints.append(turnover <= self.config.max_turnover)

        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            # Fall back to simple rebalancing
            logger.warning(f"Cost optimization failed: {problem.status}, using simple rebalancing")
            return self._periodic_rebalance(
                positions, current_weights, target_weights, symbols, total_value
            )

        # Extract trades
        trades = {}
        trade_quantities = trades_var.value

        for i, symbol in enumerate(symbols):
            if abs(trade_quantities[i] * prices[i]) >= self.config.min_trade_size:
                trades[symbol] = float(trade_quantities[i])

        metadata = {
            'optimization_status': problem.status,
            'tracking_error': float(tracking_error.value),
            'transaction_cost': float(trade_costs.value)
        }

        return trades, metadata

    def _calculate_transaction_costs(
        self,
        trades: Dict[str, float],
        positions: Dict[str, Position]
    ) -> float:
        """Calculate total transaction costs"""
        total_cost = 0.0

        for symbol, quantity in trades.items():
            price = positions[symbol].current_price if symbol in positions else 1.0
            trade_value = abs(quantity * price)

            # Commission
            total_cost += self.config.commission_per_trade
            total_cost += trade_value * self.config.commission_pct

            # Slippage
            total_cost += trade_value * self.config.slippage_pct

        return total_cost

    def _calculate_tax_impact(
        self,
        trades: Dict[str, float],
        positions: Dict[str, Position],
        current_date: Optional[datetime] = None
    ) -> float:
        """Calculate tax impact of trades"""
        current_date = current_date or datetime.now()
        total_tax = 0.0

        for symbol, quantity in trades.items():
            # Only calculate tax for sells (negative quantity)
            if quantity >= 0 or symbol not in positions:
                continue

            position = positions[symbol]

            # Calculate gain/loss
            gain = (position.current_price - position.cost_basis) * abs(quantity)

            # Only tax gains
            if gain <= 0:
                continue

            # Determine tax rate based on holding period
            days_held = (current_date - position.acquisition_date).days

            if days_held < self.config.holding_period_days:
                tax_rate = self.config.tax_rate_short_term
            else:
                tax_rate = self.config.tax_rate_long_term

            total_tax += gain * tax_rate

        return total_tax

    def _calculate_turnover(
        self,
        trades: Dict[str, float],
        total_value: float
    ) -> float:
        """Calculate portfolio turnover"""
        total_trade_value = sum(
            abs(quantity) for quantity in trades.values()
        )

        return total_trade_value / total_value if total_value > 0 else 0.0

    def simulate_rebalance(
        self,
        positions: Dict[str, Position],
        target_weights: np.ndarray,
        symbols: List[str],
        total_value: float,
        methods: Optional[List[RebalanceMethod]] = None
    ) -> Dict[str, RebalanceResult]:
        """
        Simulate different rebalancing methods

        Args:
            positions: Current portfolio positions
            target_weights: Target portfolio weights
            symbols: List of symbols
            total_value: Total portfolio value
            methods: List of methods to simulate (None = all methods)

        Returns:
            Dictionary mapping method name to RebalanceResult
        """
        if methods is None:
            methods = list(RebalanceMethod)

        results = {}

        for method in methods:
            try:
                result = self.rebalance(
                    positions, target_weights, symbols, method, total_value
                )
                results[method.value] = result
            except Exception as e:
                logger.error(f"Failed to simulate {method.value}: {str(e)}")

        return results

    async def simulate_rebalance_async(
        self,
        positions: Dict[str, Position],
        target_weights: np.ndarray,
        symbols: List[str],
        total_value: float,
        methods: Optional[List[RebalanceMethod]] = None
    ) -> Dict[str, RebalanceResult]:
        """Async version of simulate_rebalance"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.simulate_rebalance,
            positions,
            target_weights,
            symbols,
            total_value,
            methods
        )
