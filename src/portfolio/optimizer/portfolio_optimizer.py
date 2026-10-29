"""
Portfolio Optimizer Module

Implements various portfolio optimization strategies using cvxpy:
- Mean-variance optimization (Markowitz)
- Black-Litterman model
- Risk parity
- Hierarchical risk parity (HRP)
- Maximum Sharpe ratio
- Minimum variance
- Maximum diversification
- Equal risk contribution
- Custom objective functions
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Supported optimization methods"""
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    HRP = "hrp"
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_DIVERSIFICATION = "max_diversification"
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"
    CUSTOM = "custom"


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization"""
    # Weight constraints
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    long_only: bool = True  # Allow only long positions

    # Sector constraints
    sector_limits: Optional[Dict[str, Tuple[float, float]]] = None  # {sector: (min, max)}

    # Turnover constraints
    max_turnover: Optional[float] = None  # Maximum portfolio turnover
    current_weights: Optional[np.ndarray] = None  # Current portfolio weights

    # Risk constraints
    max_volatility: Optional[float] = None  # Maximum portfolio volatility
    max_tracking_error: Optional[float] = None  # Maximum tracking error vs benchmark
    benchmark_weights: Optional[np.ndarray] = None  # Benchmark weights

    # Concentration constraints
    max_concentration: Optional[float] = None  # Max sum of top N positions
    concentration_n: int = 5  # Number of positions for concentration limit

    # Group constraints
    group_limits: Optional[Dict[str, Tuple[float, float]]] = None  # {group: (min, max)}
    asset_groups: Optional[Dict[str, List[int]]] = None  # {group: [asset_indices]}

    # Transaction cost constraints
    transaction_costs: Optional[np.ndarray] = None  # Cost per unit of turnover
    max_transaction_cost: Optional[float] = None  # Maximum total transaction cost


@dataclass
class OptimizationResult:
    """Result of portfolio optimization"""
    weights: np.ndarray
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    optimization_method: str
    success: bool
    message: str
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert result to dictionary"""
        return {
            'weights': self.weights.tolist(),
            'expected_return': float(self.expected_return),
            'expected_risk': float(self.expected_risk),
            'sharpe_ratio': float(self.sharpe_ratio),
            'optimization_method': self.optimization_method,
            'success': self.success,
            'message': self.message,
            'metadata': self.metadata
        }


class PortfolioOptimizer:
    """
    Advanced portfolio optimizer supporting multiple optimization methods
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float = 0.0,
        frequency: int = 252,  # Trading days per year
    ):
        """
        Initialize portfolio optimizer

        Args:
            expected_returns: Expected returns for each asset (annualized)
            covariance_matrix: Covariance matrix of asset returns (annualized)
            risk_free_rate: Risk-free rate (annualized)
            frequency: Number of trading periods per year
        """
        self.expected_returns = np.asarray(expected_returns)
        self.covariance_matrix = np.asarray(covariance_matrix)
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.n_assets = len(expected_returns)

        # Validate inputs
        self._validate_inputs()

        logger.info(
            f"Initialized PortfolioOptimizer with {self.n_assets} assets, "
            f"risk_free_rate={risk_free_rate:.4f}"
        )

    def _validate_inputs(self):
        """Validate input parameters"""
        if len(self.expected_returns) != self.covariance_matrix.shape[0]:
            raise ValueError(
                f"Expected returns length ({len(self.expected_returns)}) "
                f"must match covariance matrix size ({self.covariance_matrix.shape[0]})"
            )

        if self.covariance_matrix.shape[0] != self.covariance_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square")

        # Check if covariance matrix is positive semi-definite
        eigenvalues = np.linalg.eigvalsh(self.covariance_matrix)
        if np.min(eigenvalues) < -1e-8:
            logger.warning(
                f"Covariance matrix is not positive semi-definite. "
                f"Min eigenvalue: {np.min(eigenvalues):.2e}"
            )

    def optimize(
        self,
        method: Union[OptimizationMethod, str],
        constraints: Optional[OptimizationConstraints] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize portfolio using specified method

        Args:
            method: Optimization method to use
            constraints: Portfolio constraints
            **kwargs: Additional method-specific parameters

        Returns:
            OptimizationResult with optimal weights and metrics
        """
        try:
            if isinstance(method, str):
                method = OptimizationMethod(method)

            constraints = constraints or OptimizationConstraints()

            # Route to appropriate optimization method
            if method == OptimizationMethod.MEAN_VARIANCE:
                weights = self._mean_variance_optimization(constraints, **kwargs)
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                weights = self._black_litterman_optimization(constraints, **kwargs)
            elif method == OptimizationMethod.RISK_PARITY:
                weights = self._risk_parity_optimization(constraints, **kwargs)
            elif method == OptimizationMethod.HRP:
                weights = self._hierarchical_risk_parity(constraints, **kwargs)
            elif method == OptimizationMethod.MAX_SHARPE:
                weights = self._max_sharpe_optimization(constraints, **kwargs)
            elif method == OptimizationMethod.MIN_VARIANCE:
                weights = self._min_variance_optimization(constraints, **kwargs)
            elif method == OptimizationMethod.MAX_DIVERSIFICATION:
                weights = self._max_diversification_optimization(constraints, **kwargs)
            elif method == OptimizationMethod.EQUAL_RISK_CONTRIBUTION:
                weights = self._equal_risk_contribution(constraints, **kwargs)
            elif method == OptimizationMethod.CUSTOM:
                weights = self._custom_optimization(constraints, **kwargs)
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_risk = np.sqrt(
                np.dot(weights, np.dot(self.covariance_matrix, weights))
            )
            sharpe_ratio = (
                (portfolio_return - self.risk_free_rate) / portfolio_risk
                if portfolio_risk > 0 else 0.0
            )

            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                expected_risk=portfolio_risk,
                sharpe_ratio=sharpe_ratio,
                optimization_method=method.value,
                success=True,
                message="Optimization completed successfully"
            )

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            return OptimizationResult(
                weights=np.zeros(self.n_assets),
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                optimization_method=method.value if isinstance(method, OptimizationMethod) else str(method),
                success=False,
                message=f"Optimization failed: {str(e)}"
            )

    async def optimize_async(
        self,
        method: Union[OptimizationMethod, str],
        constraints: Optional[OptimizationConstraints] = None,
        **kwargs
    ) -> OptimizationResult:
        """Async version of optimize"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.optimize,
            method,
            constraints,
            **kwargs
        )

    def _mean_variance_optimization(
        self,
        constraints: OptimizationConstraints,
        target_return: Optional[float] = None,
        risk_aversion: float = 1.0
    ) -> np.ndarray:
        """
        Mean-variance optimization (Markowitz)

        Args:
            constraints: Portfolio constraints
            target_return: Target portfolio return (if None, maximize return for given risk)
            risk_aversion: Risk aversion parameter (lambda)

        Returns:
            Optimal weights
        """
        w = cp.Variable(self.n_assets)

        # Objective: maximize return - risk_aversion * variance
        portfolio_return = self.expected_returns @ w
        portfolio_variance = cp.quad_form(w, self.covariance_matrix)
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)

        # Constraints
        constraint_list = [cp.sum(w) == 1]
        constraint_list.extend(self._build_constraints(w, constraints))

        # Add target return constraint if specified
        if target_return is not None:
            constraint_list.append(portfolio_return >= target_return)

        # Solve
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed with status: {problem.status}")

        return np.array(w.value).flatten()

    def _black_litterman_optimization(
        self,
        constraints: OptimizationConstraints,
        market_weights: Optional[np.ndarray] = None,
        views: Optional[Dict[str, Tuple[List[int], float]]] = None,
        view_confidences: Optional[np.ndarray] = None,
        tau: float = 0.05
    ) -> np.ndarray:
        """
        Black-Litterman optimization

        Args:
            constraints: Portfolio constraints
            market_weights: Market capitalization weights
            views: Dictionary of views {view_name: (asset_indices, expected_return)}
            view_confidences: Confidence in each view (diagonal of Omega)
            tau: Uncertainty in prior (typically 0.01-0.05)

        Returns:
            Optimal weights
        """
        # Use equal weights if market weights not provided
        if market_weights is None:
            market_weights = np.ones(self.n_assets) / self.n_assets

        # Calculate implied equilibrium returns (reverse optimization)
        risk_aversion = 2.5  # Typical value
        pi = risk_aversion * self.covariance_matrix @ market_weights

        # If no views provided, use equilibrium returns
        if views is None or len(views) == 0:
            bl_returns = pi
        else:
            # Build view matrix P and view vector Q
            n_views = len(views)
            P = np.zeros((n_views, self.n_assets))
            Q = np.zeros(n_views)

            for i, (view_name, (asset_indices, expected_return)) in enumerate(views.items()):
                for idx in asset_indices:
                    P[i, idx] = 1.0 / len(asset_indices)
                Q[i] = expected_return

            # Build view uncertainty matrix (Omega)
            if view_confidences is None:
                # Use proportional to variance
                Omega = np.diag(np.diag(P @ (tau * self.covariance_matrix) @ P.T))
            else:
                Omega = np.diag(view_confidences)

            # Black-Litterman formula
            tau_sigma = tau * self.covariance_matrix
            M_inv = np.linalg.inv(np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P)
            bl_returns = M_inv @ (
                np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q
            )

        # Use mean-variance optimization with BL returns
        original_returns = self.expected_returns.copy()
        self.expected_returns = bl_returns

        try:
            weights = self._mean_variance_optimization(constraints)
        finally:
            self.expected_returns = original_returns

        return weights

    def _risk_parity_optimization(
        self,
        constraints: OptimizationConstraints,
        target_risk_contributions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Risk parity optimization

        Args:
            constraints: Portfolio constraints
            target_risk_contributions: Target risk contribution for each asset (default: equal)

        Returns:
            Optimal weights
        """
        if target_risk_contributions is None:
            target_risk_contributions = np.ones(self.n_assets) / self.n_assets
        else:
            target_risk_contributions = target_risk_contributions / np.sum(target_risk_contributions)

        def risk_contribution(weights):
            """Calculate risk contribution of each asset"""
            portfolio_vol = np.sqrt(weights @ self.covariance_matrix @ weights)
            marginal_contrib = self.covariance_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            return contrib

        def objective(weights):
            """Minimize squared difference from target risk contributions"""
            contrib = risk_contribution(weights)
            contrib_pct = contrib / np.sum(contrib)
            return np.sum((contrib_pct - target_risk_contributions) ** 2)

        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets

        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(self.n_assets)]

        # Constraints for scipy.optimize
        scipy_constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Risk parity optimization warning: {result.message}")

        return result.x

    def _hierarchical_risk_parity(
        self,
        constraints: OptimizationConstraints,
        linkage_method: str = 'single'
    ) -> np.ndarray:
        """
        Hierarchical Risk Parity (HRP) optimization

        Args:
            constraints: Portfolio constraints
            linkage_method: Linkage method for clustering ('single', 'complete', 'average', 'ward')

        Returns:
            Optimal weights
        """
        # Convert covariance to correlation
        std_devs = np.sqrt(np.diag(self.covariance_matrix))
        correlation = self.covariance_matrix / np.outer(std_devs, std_devs)

        # Calculate distance matrix
        distance = np.sqrt((1 - correlation) / 2)
        np.fill_diagonal(distance, 0)

        # Hierarchical clustering
        dist_condensed = squareform(distance, checks=False)
        linkage_matrix = linkage(dist_condensed, method=linkage_method)

        # Get cluster order
        sort_ix = self._get_quasi_diag(linkage_matrix)
        sort_ix = sort_ix.tolist()

        # Recursive bisection
        weights = self._recursive_bisection(sort_ix)

        return weights

    def _get_quasi_diag(self, linkage_matrix):
        """Get quasi-diagonal matrix order from hierarchical clustering"""
        n = linkage_matrix.shape[0] + 1
        sort_ix = pd.Series(range(n))

        for i in range(linkage_matrix.shape[0]):
            left = int(linkage_matrix[i, 0])
            right = int(linkage_matrix[i, 1])

            # Get items in each cluster
            if left < n:
                left_items = [left]
            else:
                left_items = sort_ix[sort_ix == left].index.tolist()

            if right < n:
                right_items = [right]
            else:
                right_items = sort_ix[sort_ix == right].index.tolist()

            # Update sort_ix
            sort_ix[left_items + right_items] = i + n

        # Get final order
        cluster_order = []
        self._get_cluster_order(linkage_matrix, linkage_matrix.shape[0] - 1, cluster_order, n)

        return np.array(cluster_order)

    def _get_cluster_order(self, linkage_matrix, cluster_id, order, n):
        """Recursively get cluster order"""
        if cluster_id < n:
            order.append(cluster_id)
        else:
            idx = int(cluster_id - n)
            left = int(linkage_matrix[idx, 0])
            right = int(linkage_matrix[idx, 1])
            self._get_cluster_order(linkage_matrix, left, order, n)
            self._get_cluster_order(linkage_matrix, right, order, n)

    def _recursive_bisection(self, sort_ix):
        """Recursive bisection for HRP"""
        weights = pd.Series(1.0, index=sort_ix)
        clusters = [sort_ix]

        while len(clusters) > 0:
            clusters = [
                cluster[start:end]
                for cluster in clusters
                for start, end in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                if len(cluster) > 1
            ]

            for i in range(0, len(clusters), 2):
                if i + 1 >= len(clusters):
                    break

                cluster0 = clusters[i]
                cluster1 = clusters[i + 1]

                # Calculate cluster variance
                cov0 = self.covariance_matrix[np.ix_(cluster0, cluster0)]
                cov1 = self.covariance_matrix[np.ix_(cluster1, cluster1)]

                w0 = weights[cluster0].values
                w1 = weights[cluster1].values

                var0 = np.dot(w0, np.dot(cov0, w0))
                var1 = np.dot(w1, np.dot(cov1, w1))

                # Allocate weight
                alpha = 1 - var0 / (var0 + var1)
                weights[cluster0] *= alpha
                weights[cluster1] *= (1 - alpha)

        # Reorder to original indices
        final_weights = np.zeros(self.n_assets)
        for i, idx in enumerate(sort_ix):
            final_weights[idx] = weights.iloc[i]

        return final_weights

    def _max_sharpe_optimization(
        self,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """
        Maximum Sharpe ratio optimization

        Args:
            constraints: Portfolio constraints

        Returns:
            Optimal weights
        """
        # Use reformulation: minimize -Sharpe = minimize -(mu - rf) / sigma
        # This is equivalent to: minimize variance subject to (mu - rf) = 1

        w = cp.Variable(self.n_assets)
        kappa = cp.Variable()  # Auxiliary variable

        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, self.covariance_matrix)
        objective = cp.Minimize(portfolio_variance)

        # Constraints
        excess_returns = self.expected_returns - self.risk_free_rate
        constraint_list = [
            excess_returns @ w == 1,  # Normalize by excess return
            kappa == cp.sum(w),  # Sum of weights
        ]

        # Add portfolio constraints (scaled by kappa)
        if constraints.long_only:
            constraint_list.append(w >= 0)

        if constraints.min_weight > 0 or constraints.max_weight < 1:
            # These become harder with the kappa scaling
            # For simplicity, apply them directly
            constraint_list.append(w >= constraints.min_weight * kappa)
            constraint_list.append(w <= constraints.max_weight * kappa)

        # Solve
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed with status: {problem.status}")

        # Rescale weights to sum to 1
        weights = np.array(w.value).flatten()
        weights = weights / np.sum(weights)

        return weights

    def _min_variance_optimization(
        self,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """
        Minimum variance optimization

        Args:
            constraints: Portfolio constraints

        Returns:
            Optimal weights
        """
        w = cp.Variable(self.n_assets)

        # Objective: minimize variance
        portfolio_variance = cp.quad_form(w, self.covariance_matrix)
        objective = cp.Minimize(portfolio_variance)

        # Constraints
        constraint_list = [cp.sum(w) == 1]
        constraint_list.extend(self._build_constraints(w, constraints))

        # Solve
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed with status: {problem.status}")

        return np.array(w.value).flatten()

    def _max_diversification_optimization(
        self,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """
        Maximum diversification optimization

        Maximizes: (weighted average volatility) / (portfolio volatility)

        Args:
            constraints: Portfolio constraints

        Returns:
            Optimal weights
        """
        std_devs = np.sqrt(np.diag(self.covariance_matrix))

        w = cp.Variable(self.n_assets)

        # Objective: maximize diversification ratio
        # = maximize (w^T * sigma) / sqrt(w^T * Cov * w)
        # Reformulate: minimize portfolio variance subject to w^T * sigma = 1

        portfolio_variance = cp.quad_form(w, self.covariance_matrix)
        objective = cp.Minimize(portfolio_variance)

        # Constraints
        constraint_list = [
            std_devs @ w == 1,  # Normalize by weighted average volatility
            cp.sum(w) >= 0  # Weights should be positive (will scale later)
        ]

        if constraints.long_only:
            constraint_list.append(w >= 0)

        # Solve
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization failed with status: {problem.status}")

        # Rescale weights to sum to 1
        weights = np.array(w.value).flatten()
        weights = weights / np.sum(weights)

        return weights

    def _equal_risk_contribution(
        self,
        constraints: OptimizationConstraints
    ) -> np.ndarray:
        """
        Equal Risk Contribution (ERC) optimization

        Each asset contributes equally to portfolio risk.

        Args:
            constraints: Portfolio constraints

        Returns:
            Optimal weights
        """
        # ERC is a special case of risk parity with equal target contributions
        return self._risk_parity_optimization(constraints, target_risk_contributions=None)

    def _custom_optimization(
        self,
        constraints: OptimizationConstraints,
        objective_function: Optional[Callable] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Custom optimization with user-defined objective

        Args:
            constraints: Portfolio constraints
            objective_function: Custom objective function f(weights) -> float
            **kwargs: Additional parameters

        Returns:
            Optimal weights
        """
        if objective_function is None:
            raise ValueError("objective_function must be provided for custom optimization")

        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets

        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(self.n_assets)]

        # Constraints for scipy.optimize
        scipy_constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # Optimize
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Custom optimization warning: {result.message}")

        return result.x

    def _build_constraints(
        self,
        w: cp.Variable,
        constraints: OptimizationConstraints
    ) -> List:
        """
        Build constraint list for cvxpy optimization

        Args:
            w: Weight variable
            constraints: Portfolio constraints

        Returns:
            List of cvxpy constraints
        """
        constraint_list = []

        # Weight bounds
        if constraints.long_only:
            constraint_list.append(w >= 0)
        else:
            constraint_list.append(w >= -1)  # No more than 100% short

        constraint_list.append(w >= constraints.min_weight)
        constraint_list.append(w <= constraints.max_weight)

        # Turnover constraint
        if constraints.max_turnover is not None and constraints.current_weights is not None:
            turnover = cp.norm(w - constraints.current_weights, 1)
            constraint_list.append(turnover <= constraints.max_turnover)

        # Volatility constraint
        if constraints.max_volatility is not None:
            portfolio_variance = cp.quad_form(w, self.covariance_matrix)
            constraint_list.append(portfolio_variance <= constraints.max_volatility ** 2)

        # Tracking error constraint
        if (constraints.max_tracking_error is not None and
            constraints.benchmark_weights is not None):
            active_weights = w - constraints.benchmark_weights
            tracking_variance = cp.quad_form(active_weights, self.covariance_matrix)
            constraint_list.append(
                tracking_variance <= constraints.max_tracking_error ** 2
            )

        # Sector constraints
        if constraints.sector_limits is not None and constraints.asset_groups is not None:
            for sector, (min_weight, max_weight) in constraints.sector_limits.items():
                if sector in constraints.asset_groups:
                    sector_indices = constraints.asset_groups[sector]
                    sector_weight = cp.sum([w[i] for i in sector_indices])
                    constraint_list.append(sector_weight >= min_weight)
                    constraint_list.append(sector_weight <= max_weight)

        # Group constraints
        if constraints.group_limits is not None and constraints.asset_groups is not None:
            for group, (min_weight, max_weight) in constraints.group_limits.items():
                if group in constraints.asset_groups:
                    group_indices = constraints.asset_groups[group]
                    group_weight = cp.sum([w[i] for i in group_indices])
                    constraint_list.append(group_weight >= min_weight)
                    constraint_list.append(group_weight <= max_weight)

        return constraint_list

    def efficient_frontier(
        self,
        n_points: int = 100,
        constraints: Optional[OptimizationConstraints] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate efficient frontier

        Args:
            n_points: Number of points on the frontier
            constraints: Portfolio constraints

        Returns:
            Tuple of (returns, risks, weights_matrix)
        """
        constraints = constraints or OptimizationConstraints()

        # Find minimum and maximum return portfolios
        min_var_result = self.optimize(OptimizationMethod.MIN_VARIANCE, constraints)
        min_return = min_var_result.expected_return

        # Maximum return (may need to relax constraints)
        max_return = np.max(self.expected_returns)

        # Generate target returns
        target_returns = np.linspace(min_return, max_return * 0.95, n_points)

        returns = []
        risks = []
        weights_list = []

        for target_return in target_returns:
            try:
                result = self.optimize(
                    OptimizationMethod.MEAN_VARIANCE,
                    constraints,
                    target_return=target_return,
                    risk_aversion=0  # Not used when target_return is specified
                )

                if result.success:
                    returns.append(result.expected_return)
                    risks.append(result.expected_risk)
                    weights_list.append(result.weights)

            except Exception as e:
                logger.debug(f"Failed to optimize for target return {target_return:.4f}: {str(e)}")
                continue

        return (
            np.array(returns),
            np.array(risks),
            np.array(weights_list)
        )

    async def efficient_frontier_async(
        self,
        n_points: int = 100,
        constraints: Optional[OptimizationConstraints] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Async version of efficient_frontier"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.efficient_frontier,
            n_points,
            constraints
        )


class MultiPeriodOptimizer:
    """
    Multi-period portfolio optimization with rebalancing
    """

    def __init__(
        self,
        expected_returns_sequence: List[np.ndarray],
        covariance_sequence: List[np.ndarray],
        risk_free_rate: float = 0.0,
        transaction_costs: Optional[np.ndarray] = None
    ):
        """
        Initialize multi-period optimizer

        Args:
            expected_returns_sequence: List of expected returns for each period
            covariance_sequence: List of covariance matrices for each period
            risk_free_rate: Risk-free rate
            transaction_costs: Transaction costs per unit of turnover
        """
        self.expected_returns_sequence = expected_returns_sequence
        self.covariance_sequence = covariance_sequence
        self.risk_free_rate = risk_free_rate
        self.transaction_costs = transaction_costs
        self.n_periods = len(expected_returns_sequence)
        self.n_assets = len(expected_returns_sequence[0])

    def optimize(
        self,
        initial_weights: np.ndarray,
        constraints: Optional[OptimizationConstraints] = None,
        discount_factor: float = 0.95
    ) -> List[np.ndarray]:
        """
        Optimize portfolio over multiple periods

        Args:
            initial_weights: Initial portfolio weights
            constraints: Portfolio constraints
            discount_factor: Discount factor for future periods

        Returns:
            List of optimal weights for each period
        """
        constraints = constraints or OptimizationConstraints()

        # Decision variables: weights for each period
        weights = [cp.Variable(self.n_assets) for _ in range(self.n_periods)]

        # Objective: maximize discounted sum of utilities
        objective_terms = []
        for t in range(self.n_periods):
            # Period return
            period_return = self.expected_returns_sequence[t] @ weights[t]

            # Period variance
            period_variance = cp.quad_form(weights[t], self.covariance_sequence[t])

            # Utility (mean-variance)
            utility = period_return - 0.5 * period_variance

            # Transaction costs
            if t == 0:
                turnover = cp.norm(weights[t] - initial_weights, 1)
            else:
                turnover = cp.norm(weights[t] - weights[t-1], 1)

            if self.transaction_costs is not None:
                tc = self.transaction_costs @ cp.abs(
                    weights[t] - (initial_weights if t == 0 else weights[t-1])
                )
            else:
                tc = 0.01 * turnover  # Default 1% transaction cost

            # Discounted utility
            discounted_utility = (discount_factor ** t) * (utility - tc)
            objective_terms.append(discounted_utility)

        objective = cp.Maximize(cp.sum(objective_terms))

        # Constraints for each period
        constraint_list = []
        for t in range(self.n_periods):
            constraint_list.append(cp.sum(weights[t]) == 1)
            constraint_list.append(weights[t] >= constraints.min_weight)
            constraint_list.append(weights[t] <= constraints.max_weight)

            if constraints.long_only:
                constraint_list.append(weights[t] >= 0)

        # Solve
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Multi-period optimization failed with status: {problem.status}")

        return [np.array(w.value).flatten() for w in weights]
