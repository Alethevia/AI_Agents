"""
Portfolio optimizer module for optimizing portfolio allocations.

Provides multiple optimization methods including mean-variance optimization,
Black-Litterman model, risk parity, hierarchical risk parity (HRP),
maximum Sharpe ratio, and minimum variance.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram

from src.core.logging.logger import get_logger, log_async_execution_time


logger = get_logger(__name__)


class OptimizationObjective(str, Enum):
    """Portfolio optimization objectives."""

    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    HIERARCHICAL_RISK_PARITY = "hrp"
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    MAX_DIVERSIFICATION = "max_diversification"


class PortfolioConstraints(BaseModel):
    """Portfolio optimization constraints."""

    min_weight: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Minimum asset weight"
    )
    max_weight: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Maximum asset weight"
    )
    long_only: bool = Field(
        default=True,
        description="Long-only constraint"
    )
    sum_to_one: bool = Field(
        default=True,
        description="Weights must sum to 1"
    )
    max_turnover: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum portfolio turnover"
    )
    sector_constraints: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Sector weight constraints (min, max)"
    )


class PortfolioAllocation(BaseModel):
    """Portfolio allocation result."""

    weights: Dict[str, float] = Field(..., description="Asset weights")
    expected_return: float = Field(..., description="Expected portfolio return")
    volatility: float = Field(..., description="Portfolio volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    optimization_method: OptimizationObjective = Field(
        ...,
        description="Optimization method used"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Allocation timestamp"
    )


class PortfolioOptimizerConfig(BaseModel):
    """Configuration for portfolio optimizer."""

    # General settings
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.MAX_SHARPE,
        description="Default optimization objective"
    )
    risk_free_rate: float = Field(
        default=0.02,
        ge=0,
        description="Risk-free rate for Sharpe ratio"
    )

    # Black-Litterman settings
    bl_tau: float = Field(
        default=0.05,
        ge=0,
        description="Black-Litterman uncertainty parameter"
    )
    bl_risk_aversion: float = Field(
        default=2.5,
        ge=0,
        description="Risk aversion parameter"
    )

    # Risk parity settings
    rp_max_iterations: int = Field(
        default=1000,
        ge=1,
        description="Max iterations for risk parity"
    )
    rp_tolerance: float = Field(
        default=1e-8,
        ge=0,
        description="Convergence tolerance"
    )

    # HRP settings
    hrp_linkage_method: str = Field(
        default="single",
        description="Linkage method for HRP (single/average/complete)"
    )

    # Optimization settings
    max_iterations: int = Field(
        default=1000,
        ge=1,
        description="Maximum optimization iterations"
    )
    tolerance: float = Field(
        default=1e-8,
        ge=0,
        description="Convergence tolerance"
    )


class PortfolioOptimizer:
    """
    Optimize portfolio allocations using various methods.

    Supports mean-variance optimization, Black-Litterman, risk parity,
    hierarchical risk parity (HRP), maximum Sharpe ratio, and minimum variance.
    """

    def __init__(
        self,
        config: Optional[PortfolioOptimizerConfig] = None,
        constraints: Optional[PortfolioConstraints] = None
    ):
        """
        Initialize portfolio optimizer.

        Args:
            config: Optimizer configuration
            constraints: Portfolio constraints
        """
        self.config = config or PortfolioOptimizerConfig()
        self.constraints = constraints or PortfolioConstraints()
        self.logger = get_logger(f"{__name__}.PortfolioOptimizer")

        # Optimization history
        self._history: List[PortfolioAllocation] = []

        # Statistics
        self._stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'errors': 0,
        }

    @log_async_execution_time()
    async def optimize(
        self,
        returns: pd.DataFrame,
        method: Optional[OptimizationObjective] = None,
        views: Optional[Dict[str, float]] = None,
        view_confidences: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> PortfolioAllocation:
        """
        Optimize portfolio allocation.

        Args:
            returns: Historical returns DataFrame (assets as columns)
            method: Optimization method (uses config default if None)
            views: Optional views for Black-Litterman (asset -> expected return)
            view_confidences: Optional view confidences for Black-Litterman
            **kwargs: Method-specific kwargs

        Returns:
            Portfolio allocation
        """
        method = method or self.config.objective

        self.logger.info(
            "Starting portfolio optimization",
            method=method.value,
            n_assets=len(returns.columns)
        )

        try:
            self._stats['total_optimizations'] += 1

            if method == OptimizationObjective.MEAN_VARIANCE:
                allocation = await self._mean_variance_optimization(returns, **kwargs)
            elif method == OptimizationObjective.BLACK_LITTERMAN:
                allocation = await self._black_litterman_optimization(
                    returns, views, view_confidences, **kwargs
                )
            elif method == OptimizationObjective.RISK_PARITY:
                allocation = await self._risk_parity_optimization(returns, **kwargs)
            elif method == OptimizationObjective.HIERARCHICAL_RISK_PARITY:
                allocation = await self._hierarchical_risk_parity(returns, **kwargs)
            elif method == OptimizationObjective.MAX_SHARPE:
                allocation = await self._max_sharpe_optimization(returns, **kwargs)
            elif method == OptimizationObjective.MIN_VARIANCE:
                allocation = await self._min_variance_optimization(returns, **kwargs)
            elif method == OptimizationObjective.MAX_RETURN:
                allocation = await self._max_return_optimization(returns, **kwargs)
            elif method == OptimizationObjective.MAX_DIVERSIFICATION:
                allocation = await self._max_diversification_optimization(returns, **kwargs)
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            # Store in history
            self._history.append(allocation)
            self._stats['successful_optimizations'] += 1

            self.logger.info(
                "Portfolio optimization completed",
                method=method.value,
                expected_return=allocation.expected_return,
                volatility=allocation.volatility,
                sharpe_ratio=allocation.sharpe_ratio
            )

            return allocation

        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(
                "Error during portfolio optimization",
                method=method.value if method else "unknown",
                error=str(e),
                exc_info=True
            )
            raise

    async def _mean_variance_optimization(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
        **kwargs
    ) -> PortfolioAllocation:
        """Perform mean-variance optimization."""
        # Calculate expected returns and covariance
        expected_returns = returns.mean()
        cov_matrix = returns.cov()

        n_assets = len(returns.columns)
        assets = returns.columns.tolist()

        # Objective: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
        ]

        if target_return is not None:
            # Add return constraint
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - target_return
            })

        # Bounds
        bounds = tuple(
            (self.constraints.min_weight, self.constraints.max_weight)
            for _ in range(n_assets)
        )

        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )

        if not result.success:
            self.logger.warning(
                "Optimization did not converge",
                message=result.message
            )

        weights = dict(zip(assets, result.x))

        # Calculate portfolio metrics
        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_vol = np.sqrt(objective(result.x))
        sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        return PortfolioAllocation(
            weights=weights,
            expected_return=float(portfolio_return),
            volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe),
            optimization_method=OptimizationObjective.MEAN_VARIANCE,
            metadata={'target_return': target_return}
        )

    async def _black_litterman_optimization(
        self,
        returns: pd.DataFrame,
        views: Optional[Dict[str, float]] = None,
        view_confidences: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> PortfolioAllocation:
        """Perform Black-Litterman optimization."""
        # Calculate market parameters
        expected_returns = returns.mean()
        cov_matrix = returns.cov()

        n_assets = len(returns.columns)
        assets = returns.columns.tolist()

        # Market equilibrium returns (reverse optimization)
        market_weights = kwargs.get('market_weights')
        if market_weights is None:
            # Equal weights as market portfolio
            market_weights = np.array([1.0 / n_assets] * n_assets)

        # Implied equilibrium returns
        pi = self.config.bl_risk_aversion * np.dot(cov_matrix, market_weights)

        # If no views provided, use equilibrium returns
        if views is None or len(views) == 0:
            bl_returns = pi
        else:
            # Construct view matrix P and view vector Q
            P = []
            Q = []
            omega_diag = []

            for asset, view_return in views.items():
                if asset in assets:
                    # Absolute view on asset
                    p_row = [0] * n_assets
                    p_row[assets.index(asset)] = 1
                    P.append(p_row)
                    Q.append(view_return)

                    # View uncertainty
                    confidence = view_confidences.get(asset, 1.0) if view_confidences else 1.0
                    omega_diag.append(
                        self.config.bl_tau * cov_matrix.iloc[assets.index(asset), assets.index(asset)] / confidence
                    )

            if len(P) > 0:
                P = np.array(P)
                Q = np.array(Q)
                Omega = np.diag(omega_diag)

                # Black-Litterman formula
                tau_sigma = self.config.bl_tau * cov_matrix

                # Posterior returns
                term1 = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P)
                term2 = np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q
                bl_returns = term1 @ term2
            else:
                bl_returns = pi

        # Optimize using BL returns
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = tuple(
            (self.constraints.min_weight, self.constraints.max_weight)
            for _ in range(n_assets)
        )

        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )

        weights = dict(zip(assets, result.x))

        portfolio_return = np.dot(result.x, bl_returns)
        portfolio_vol = np.sqrt(objective(result.x))
        sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        return PortfolioAllocation(
            weights=weights,
            expected_return=float(portfolio_return),
            volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe),
            optimization_method=OptimizationObjective.BLACK_LITTERMAN,
            metadata={
                'views': views,
                'n_views': len(views) if views else 0,
            }
        )

    async def _risk_parity_optimization(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> PortfolioAllocation:
        """Perform risk parity optimization."""
        cov_matrix = returns.cov()
        n_assets = len(returns.columns)
        assets = returns.columns.tolist()

        # Risk parity: equalize risk contributions
        def risk_budget_objective(weights):
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib

            # Target equal risk contributions
            target_risk = portfolio_var / n_assets

            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = tuple(
            (self.constraints.min_weight, self.constraints.max_weight)
            for _ in range(n_assets)
        )

        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.config.rp_max_iterations,
                'ftol': self.config.rp_tolerance
            }
        )

        weights = dict(zip(assets, result.x))

        # Calculate metrics
        expected_returns = returns.mean()
        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_vol = np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x)))
        sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        return PortfolioAllocation(
            weights=weights,
            expected_return=float(portfolio_return),
            volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe),
            optimization_method=OptimizationObjective.RISK_PARITY
        )

    async def _hierarchical_risk_parity(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> PortfolioAllocation:
        """Perform hierarchical risk parity (HRP) optimization."""
        cov_matrix = returns.cov()
        corr_matrix = returns.corr()
        assets = returns.columns.tolist()

        # Step 1: Tree clustering
        # Convert correlation to distance
        dist = np.sqrt(0.5 * (1 - corr_matrix))

        # Hierarchical clustering
        link = linkage(dist, method=self.config.hrp_linkage_method)

        # Step 2: Quasi-diagonalization
        def get_quasi_diag(link):
            """Get quasi-diagonal ordering from linkage."""
            n = link.shape[0] + 1
            sort_ix = [n]

            def quasi_diag_rec(link, sort_ix):
                if len(sort_ix) > n:
                    return sort_ix

                idx = len(sort_ix) - 1
                if sort_ix[idx] < n:
                    return sort_ix

                cluster_idx = int(sort_ix[idx] - n)
                left = int(link[cluster_idx, 0])
                right = int(link[cluster_idx, 1])

                sort_ix[idx] = left
                sort_ix.append(right)

                return quasi_diag_rec(link, sort_ix)

            sort_ix = quasi_diag_rec(link, sort_ix)
            return sort_ix

        sort_ix = get_quasi_diag(link)

        # Step 3: Recursive bisection
        def get_cluster_var(cov, cluster_items):
            """Calculate cluster variance."""
            cov_slice = cov.iloc[cluster_items, cluster_items]
            w = 1.0 / np.diag(cov_slice)
            w /= w.sum()
            return np.dot(w, np.dot(cov_slice, w))

        def recursive_bisection(cov, items):
            """Recursive bisection to allocate weights."""
            weights = pd.Series(1.0, index=items)

            cluster_items = [items]

            while len(cluster_items) > 0:
                cluster = cluster_items.pop(0)

                if len(cluster) == 1:
                    continue

                # Split cluster
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Calculate cluster variances
                var_left = get_cluster_var(cov, left)
                var_right = get_cluster_var(cov, right)

                # Allocate weight inversely proportional to variance
                alpha = 1 - var_left / (var_left + var_right)

                weights[left] *= alpha
                weights[right] *= (1 - alpha)

                cluster_items.extend([left, right])

            return weights

        # Get HRP weights
        hrp_weights = recursive_bisection(cov_matrix, sort_ix)
        hrp_weights = hrp_weights / hrp_weights.sum()

        weights = dict(zip(assets, hrp_weights))

        # Calculate metrics
        expected_returns = returns.mean()
        portfolio_return = np.dot(hrp_weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(hrp_weights, np.dot(cov_matrix, hrp_weights)))
        sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        return PortfolioAllocation(
            weights=weights,
            expected_return=float(portfolio_return),
            volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe),
            optimization_method=OptimizationObjective.HIERARCHICAL_RISK_PARITY,
            metadata={'linkage_method': self.config.hrp_linkage_method}
        )

    async def _max_sharpe_optimization(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> PortfolioAllocation:
        """Maximize Sharpe ratio."""
        expected_returns = returns.mean()
        cov_matrix = returns.cov()

        n_assets = len(returns.columns)
        assets = returns.columns.tolist()

        # Objective: maximize Sharpe ratio (minimize negative Sharpe)
        def neg_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.config.risk_free_rate) / portfolio_vol

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = tuple(
            (self.constraints.min_weight, self.constraints.max_weight)
            for _ in range(n_assets)
        )

        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(
            neg_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )

        weights = dict(zip(assets, result.x))

        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_vol = np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x)))
        sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        return PortfolioAllocation(
            weights=weights,
            expected_return=float(portfolio_return),
            volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe),
            optimization_method=OptimizationObjective.MAX_SHARPE
        )

    async def _min_variance_optimization(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> PortfolioAllocation:
        """Minimize portfolio variance."""
        cov_matrix = returns.cov()
        expected_returns = returns.mean()

        n_assets = len(returns.columns)
        assets = returns.columns.tolist()

        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = tuple(
            (self.constraints.min_weight, self.constraints.max_weight)
            for _ in range(n_assets)
        )

        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )

        weights = dict(zip(assets, result.x))

        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_vol = np.sqrt(objective(result.x))
        sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        return PortfolioAllocation(
            weights=weights,
            expected_return=float(portfolio_return),
            volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe),
            optimization_method=OptimizationObjective.MIN_VARIANCE
        )

    async def _max_return_optimization(
        self,
        returns: pd.DataFrame,
        max_volatility: Optional[float] = None,
        **kwargs
    ) -> PortfolioAllocation:
        """Maximize portfolio return."""
        expected_returns = returns.mean()
        cov_matrix = returns.cov()

        n_assets = len(returns.columns)
        assets = returns.columns.tolist()

        # Objective: maximize return (minimize negative return)
        def neg_return(weights):
            return -np.dot(weights, expected_returns)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        if max_volatility is not None:
            # Add volatility constraint
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: max_volatility - np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            })

        bounds = tuple(
            (self.constraints.min_weight, self.constraints.max_weight)
            for _ in range(n_assets)
        )

        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(
            neg_return,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )

        weights = dict(zip(assets, result.x))

        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_vol = np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x)))
        sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        return PortfolioAllocation(
            weights=weights,
            expected_return=float(portfolio_return),
            volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe),
            optimization_method=OptimizationObjective.MAX_RETURN,
            metadata={'max_volatility': max_volatility}
        )

    async def _max_diversification_optimization(
        self,
        returns: pd.DataFrame,
        **kwargs
    ) -> PortfolioAllocation:
        """Maximize diversification ratio."""
        expected_returns = returns.mean()
        cov_matrix = returns.cov()
        volatilities = np.sqrt(np.diag(cov_matrix))

        n_assets = len(returns.columns)
        assets = returns.columns.tolist()

        # Diversification ratio = (w * sigma) / sqrt(w * Cov * w)
        def neg_diversification(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            weighted_vol = np.dot(weights, volatilities)
            return -weighted_vol / portfolio_vol

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = tuple(
            (self.constraints.min_weight, self.constraints.max_weight)
            for _ in range(n_assets)
        )

        x0 = np.array([1.0 / n_assets] * n_assets)

        result = minimize(
            neg_diversification,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )

        weights = dict(zip(assets, result.x))

        portfolio_return = np.dot(result.x, expected_returns)
        portfolio_vol = np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x)))
        sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        return PortfolioAllocation(
            weights=weights,
            expected_return=float(portfolio_return),
            volatility=float(portfolio_vol),
            sharpe_ratio=float(sharpe),
            optimization_method=OptimizationObjective.MAX_DIVERSIFICATION
        )

    def get_allocation_history(
        self,
        method: Optional[OptimizationObjective] = None
    ) -> List[PortfolioAllocation]:
        """Get allocation history."""
        history = self._history

        if method:
            history = [a for a in history if a.optimization_method == method]

        return history

    def get_latest_allocation(
        self,
        method: Optional[OptimizationObjective] = None
    ) -> Optional[PortfolioAllocation]:
        """Get latest allocation."""
        history = self.get_allocation_history(method)
        return history[-1] if history else None

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        if not self._history:
            return {
                **self._stats,
                'avg_sharpe': 0,
                'avg_return': 0,
                'avg_volatility': 0,
            }

        return {
            **self._stats,
            'allocations_by_method': {
                method.value: sum(
                    1 for a in self._history
                    if a.optimization_method == method
                )
                for method in OptimizationObjective
            },
            'avg_sharpe': np.mean([a.sharpe_ratio for a in self._history]),
            'avg_return': np.mean([a.expected_return for a in self._history]),
            'avg_volatility': np.mean([a.volatility for a in self._history]),
            'best_sharpe': max(a.sharpe_ratio for a in self._history),
        }

    def reset_history(self) -> None:
        """Reset allocation history."""
        self._history.clear()
        self.logger.info("Allocation history reset")

    def reset_statistics(self) -> None:
        """Reset optimizer statistics."""
        self._stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'errors': 0,
        }
