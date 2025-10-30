"""
Attribution Analysis Module

Implements various attribution analysis methods:
- Brinson attribution (allocation, selection, interaction)
- Factor attribution
- Sector attribution
- Asset class attribution
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class AttributionMethod(Enum):
    """Supported attribution methods"""
    BRINSON = "brinson"
    BRINSON_FACHLER = "brinson_fachler"
    FACTOR = "factor"
    SECTOR = "sector"
    ASSET_CLASS = "asset_class"


@dataclass
class BrinsonAttribution:
    """Brinson attribution analysis result"""
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float
    by_segment: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = {
            'allocation_effect': float(self.allocation_effect),
            'selection_effect': float(self.selection_effect),
            'interaction_effect': float(self.interaction_effect),
            'total_effect': float(self.total_effect)
        }
        if self.by_segment is not None:
            result['by_segment'] = self.by_segment.to_dict()
        return result


@dataclass
class FactorAttribution:
    """Factor attribution analysis result"""
    factor_returns: Dict[str, float]
    factor_exposures: Dict[str, float]
    factor_contributions: Dict[str, float]
    specific_return: float
    total_return: float
    r_squared: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'factor_returns': {k: float(v) for k, v in self.factor_returns.items()},
            'factor_exposures': {k: float(v) for k, v in self.factor_exposures.items()},
            'factor_contributions': {k: float(v) for k, v in self.factor_contributions.items()},
            'specific_return': float(self.specific_return),
            'total_return': float(self.total_return),
            'r_squared': float(self.r_squared)
        }


@dataclass
class SectorAttribution:
    """Sector attribution analysis result"""
    sector_allocation: pd.DataFrame
    sector_selection: pd.DataFrame
    sector_interaction: pd.DataFrame
    total_by_sector: pd.DataFrame
    summary: Dict[str, float]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'sector_allocation': self.sector_allocation.to_dict(),
            'sector_selection': self.sector_selection.to_dict(),
            'sector_interaction': self.sector_interaction.to_dict(),
            'total_by_sector': self.total_by_sector.to_dict(),
            'summary': {k: float(v) for k, v in self.summary.items()}
        }


class AttributionAnalyzer:
    """
    Advanced attribution analyzer for portfolios
    """

    def __init__(
        self,
        portfolio_weights: Union[pd.DataFrame, np.ndarray],
        portfolio_returns: Union[pd.DataFrame, np.ndarray],
        benchmark_weights: Union[pd.DataFrame, np.ndarray],
        benchmark_returns: Union[pd.DataFrame, np.ndarray],
        asset_names: Optional[List[str]] = None
    ):
        """
        Initialize attribution analyzer

        Args:
            portfolio_weights: Portfolio weights over time
            portfolio_returns: Portfolio returns by asset
            benchmark_weights: Benchmark weights over time
            benchmark_returns: Benchmark returns by asset
            asset_names: List of asset names
        """
        # Convert to DataFrames if necessary
        if isinstance(portfolio_weights, np.ndarray):
            portfolio_weights = pd.DataFrame(portfolio_weights)
        if isinstance(portfolio_returns, np.ndarray):
            portfolio_returns = pd.DataFrame(portfolio_returns)
        if isinstance(benchmark_weights, np.ndarray):
            benchmark_weights = pd.DataFrame(benchmark_weights)
        if isinstance(benchmark_returns, np.ndarray):
            benchmark_returns = pd.DataFrame(benchmark_returns)

        self.portfolio_weights = portfolio_weights
        self.portfolio_returns = portfolio_returns
        self.benchmark_weights = benchmark_weights
        self.benchmark_returns = benchmark_returns
        self.asset_names = asset_names or list(portfolio_weights.columns)

        # Validate inputs
        self._validate_inputs()

        logger.info(
            f"Initialized AttributionAnalyzer with {len(self.asset_names)} assets, "
            f"{len(portfolio_weights)} periods"
        )

    def _validate_inputs(self):
        """Validate input data"""
        if self.portfolio_weights.shape != self.benchmark_weights.shape:
            raise ValueError("Portfolio and benchmark weights must have same shape")

        if self.portfolio_returns.shape != self.benchmark_returns.shape:
            raise ValueError("Portfolio and benchmark returns must have same shape")

    def brinson_attribution(
        self,
        method: str = "brinson"
    ) -> BrinsonAttribution:
        """
        Brinson attribution analysis

        Args:
            method: 'brinson' or 'brinson_fachler'

        Returns:
            BrinsonAttribution with allocation and selection effects
        """
        # Use average weights over the period
        wp = self.portfolio_weights.mean()  # Portfolio weights
        wb = self.benchmark_weights.mean()  # Benchmark weights
        rp = self.portfolio_returns.mean()  # Portfolio returns
        rb = self.benchmark_returns.mean()  # Benchmark returns

        if method == "brinson":
            # Standard Brinson attribution
            # Allocation effect: (wp - wb) * (rb - R_b)
            R_b = (wb * rb).sum()  # Benchmark total return
            allocation = ((wp - wb) * (rb - R_b)).sum()

            # Selection effect: wb * (rp - rb)
            selection = (wb * (rp - rb)).sum()

            # Interaction effect: (wp - wb) * (rp - rb)
            interaction = ((wp - wb) * (rp - rb)).sum()

        elif method == "brinson_fachler":
            # Brinson-Fachler (no interaction term)
            R_b = (wb * rb).sum()
            allocation = ((wp - wb) * rb).sum()
            selection = (wb * (rp - rb)).sum()
            interaction = 0.0

        else:
            raise ValueError(f"Unknown Brinson method: {method}")

        total = allocation + selection + interaction

        # Calculate by segment
        by_segment = pd.DataFrame({
            'Portfolio Weight': wp,
            'Benchmark Weight': wb,
            'Portfolio Return': rp,
            'Benchmark Return': rb,
            'Allocation': (wp - wb) * (rb - (wb * rb).sum()),
            'Selection': wb * (rp - rb),
            'Interaction': (wp - wb) * (rp - rb)
        }, index=self.asset_names)

        return BrinsonAttribution(
            allocation_effect=allocation,
            selection_effect=selection,
            interaction_effect=interaction,
            total_effect=total,
            by_segment=by_segment
        )

    async def brinson_attribution_async(
        self,
        method: str = "brinson"
    ) -> BrinsonAttribution:
        """Async version of brinson_attribution"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.brinson_attribution,
            method
        )

    def factor_attribution(
        self,
        factor_returns: pd.DataFrame,
        factor_exposures: Optional[pd.DataFrame] = None
    ) -> FactorAttribution:
        """
        Factor-based attribution analysis

        Args:
            factor_returns: Returns of risk factors
            factor_exposures: Factor exposures (if None, estimated via regression)

        Returns:
            FactorAttribution with factor contributions
        """
        # Calculate portfolio return
        portfolio_return = (self.portfolio_weights * self.portfolio_returns).sum(axis=1)

        # Estimate factor exposures if not provided
        if factor_exposures is None:
            # Run regression to estimate exposures
            factor_exposures = self._estimate_factor_exposures(
                portfolio_return,
                factor_returns
            )

        # Calculate factor contributions
        factor_contributions = {}
        factor_return_dict = {}
        factor_exposure_dict = {}

        for factor in factor_returns.columns:
            avg_exposure = factor_exposures[factor].mean()
            avg_return = factor_returns[factor].mean()

            factor_contributions[factor] = avg_exposure * avg_return
            factor_return_dict[factor] = avg_return
            factor_exposure_dict[factor] = avg_exposure

        # Calculate specific (unexplained) return
        explained_return = sum(factor_contributions.values())
        total_return = portfolio_return.mean()
        specific_return = total_return - explained_return

        # Calculate R-squared
        predicted_returns = pd.Series(0.0, index=portfolio_return.index)
        for factor in factor_returns.columns:
            predicted_returns += factor_exposures[factor] * factor_returns[factor]

        residuals = portfolio_return - predicted_returns
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((portfolio_return - portfolio_return.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return FactorAttribution(
            factor_returns=factor_return_dict,
            factor_exposures=factor_exposure_dict,
            factor_contributions=factor_contributions,
            specific_return=specific_return,
            total_return=total_return,
            r_squared=r_squared
        )

    def _estimate_factor_exposures(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Estimate factor exposures via rolling regression

        Args:
            portfolio_returns: Portfolio returns
            factor_returns: Factor returns

        Returns:
            DataFrame of factor exposures over time
        """
        # Align dates
        aligned_data = pd.concat([portfolio_returns, factor_returns], axis=1, join='inner')
        portfolio_col = aligned_data.columns[0]
        factor_cols = aligned_data.columns[1:]

        # Rolling regression
        window = min(60, len(aligned_data) // 2)  # 60 days or half the data
        exposures = pd.DataFrame(index=aligned_data.index, columns=factor_cols)

        for i in range(window, len(aligned_data)):
            y = aligned_data[portfolio_col].iloc[i-window:i].values
            X = aligned_data[factor_cols].iloc[i-window:i].values

            # Add constant
            X = np.column_stack([np.ones(len(X)), X])

            # OLS regression
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                exposures.iloc[i] = beta[1:]  # Exclude intercept
            except:
                exposures.iloc[i] = 0.0

        return exposures.astype(float)

    def sector_attribution(
        self,
        sector_mapping: Dict[str, str]
    ) -> SectorAttribution:
        """
        Sector-level attribution analysis

        Args:
            sector_mapping: Mapping from asset name to sector

        Returns:
            SectorAttribution with sector-level analysis
        """
        # Create sector DataFrames
        sectors = sorted(set(sector_mapping.values()))

        # Aggregate weights and returns by sector
        portfolio_sector_weights = pd.DataFrame(index=self.portfolio_weights.index, columns=sectors)
        benchmark_sector_weights = pd.DataFrame(index=self.benchmark_weights.index, columns=sectors)
        portfolio_sector_returns = pd.DataFrame(index=self.portfolio_returns.index, columns=sectors)
        benchmark_sector_returns = pd.DataFrame(index=self.benchmark_returns.index, columns=sectors)

        for sector in sectors:
            # Find assets in this sector
            sector_assets = [asset for asset, s in sector_mapping.items() if s == sector]
            sector_indices = [i for i, asset in enumerate(self.asset_names) if asset in sector_assets]

            if len(sector_indices) > 0:
                # Sum weights
                portfolio_sector_weights[sector] = self.portfolio_weights.iloc[:, sector_indices].sum(axis=1)
                benchmark_sector_weights[sector] = self.benchmark_weights.iloc[:, sector_indices].sum(axis=1)

                # Weighted average returns
                portfolio_sector_returns[sector] = (
                    (self.portfolio_weights.iloc[:, sector_indices] *
                     self.portfolio_returns.iloc[:, sector_indices]).sum(axis=1) /
                    portfolio_sector_weights[sector].replace(0, np.nan)
                ).fillna(0)

                benchmark_sector_returns[sector] = (
                    (self.benchmark_weights.iloc[:, sector_indices] *
                     self.benchmark_returns.iloc[:, sector_indices]).sum(axis=1) /
                    benchmark_sector_weights[sector].replace(0, np.nan)
                ).fillna(0)
            else:
                portfolio_sector_weights[sector] = 0.0
                benchmark_sector_weights[sector] = 0.0
                portfolio_sector_returns[sector] = 0.0
                benchmark_sector_returns[sector] = 0.0

        # Calculate attribution effects
        wp = portfolio_sector_weights.mean()
        wb = benchmark_sector_weights.mean()
        rp = portfolio_sector_returns.mean()
        rb = benchmark_sector_returns.mean()

        R_b = (wb * rb).sum()  # Benchmark total return

        allocation = pd.Series(
            (wp - wb) * (rb - R_b),
            index=sectors,
            name='Allocation'
        )

        selection = pd.Series(
            wb * (rp - rb),
            index=sectors,
            name='Selection'
        )

        interaction = pd.Series(
            (wp - wb) * (rp - rb),
            index=sectors,
            name='Interaction'
        )

        total = pd.Series(
            allocation + selection + interaction,
            index=sectors,
            name='Total'
        )

        # Summary
        summary = {
            'Total Allocation': allocation.sum(),
            'Total Selection': selection.sum(),
            'Total Interaction': interaction.sum(),
            'Total Active Return': total.sum()
        }

        return SectorAttribution(
            sector_allocation=allocation.to_frame(),
            sector_selection=selection.to_frame(),
            sector_interaction=interaction.to_frame(),
            total_by_sector=total.to_frame(),
            summary=summary
        )

    async def sector_attribution_async(
        self,
        sector_mapping: Dict[str, str]
    ) -> SectorAttribution:
        """Async version of sector_attribution"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.sector_attribution,
            sector_mapping
        )

    def asset_class_attribution(
        self,
        asset_class_mapping: Dict[str, str]
    ) -> SectorAttribution:
        """
        Asset class attribution analysis

        Args:
            asset_class_mapping: Mapping from asset name to asset class

        Returns:
            SectorAttribution (reused structure) with asset class analysis
        """
        # Reuse sector attribution logic with asset classes
        return self.sector_attribution(asset_class_mapping)

    def time_series_attribution(
        self,
        frequency: str = 'M'
    ) -> pd.DataFrame:
        """
        Time series of attribution effects

        Args:
            frequency: Resampling frequency ('D', 'W', 'M', 'Q', 'Y')

        Returns:
            DataFrame with attribution effects over time
        """
        results = []

        # Group by frequency
        if hasattr(self.portfolio_weights.index, 'to_period'):
            periods = self.portfolio_weights.index.to_period(frequency).unique()

            for period in periods:
                # Filter data for this period
                mask = self.portfolio_weights.index.to_period(frequency) == period

                period_pw = self.portfolio_weights[mask]
                period_bw = self.benchmark_weights[mask]
                period_pr = self.portfolio_returns[mask]
                period_br = self.benchmark_returns[mask]

                if len(period_pw) == 0:
                    continue

                # Create temporary analyzer
                temp_analyzer = AttributionAnalyzer(
                    portfolio_weights=period_pw,
                    portfolio_returns=period_pr,
                    benchmark_weights=period_bw,
                    benchmark_returns=period_br,
                    asset_names=self.asset_names
                )

                # Calculate Brinson attribution
                attribution = temp_analyzer.brinson_attribution()

                results.append({
                    'Period': period.to_timestamp(),
                    'Allocation': attribution.allocation_effect,
                    'Selection': attribution.selection_effect,
                    'Interaction': attribution.interaction_effect,
                    'Total': attribution.total_effect
                })

        return pd.DataFrame(results)

    def contribution_to_active_return(self) -> pd.DataFrame:
        """
        Calculate contribution of each asset to active return

        Returns:
            DataFrame with active return contributions
        """
        # Active weights
        active_weights = self.portfolio_weights - self.benchmark_weights

        # Active returns
        active_returns = self.portfolio_returns - self.benchmark_returns

        # Contribution to active return
        contributions = active_weights * self.portfolio_returns

        # Summary statistics
        summary = pd.DataFrame({
            'Avg Active Weight': active_weights.mean(),
            'Avg Active Return': active_returns.mean(),
            'Contribution to Active Return': contributions.mean(),
            'Volatility of Contribution': contributions.std()
        }, index=self.asset_names)

        return summary

    def risk_attribution(
        self,
        covariance_matrix: np.ndarray
    ) -> pd.DataFrame:
        """
        Risk attribution analysis

        Args:
            covariance_matrix: Covariance matrix of asset returns

        Returns:
            DataFrame with risk attribution
        """
        # Average portfolio weights
        avg_weights = self.portfolio_weights.mean().values

        # Portfolio variance
        portfolio_var = avg_weights @ covariance_matrix @ avg_weights
        portfolio_vol = np.sqrt(portfolio_var)

        # Marginal contribution to risk
        marginal_contrib = covariance_matrix @ avg_weights / portfolio_vol

        # Total contribution to risk
        risk_contrib = avg_weights * marginal_contrib

        # Percentage contribution
        risk_contrib_pct = risk_contrib / portfolio_vol * 100

        # Create DataFrame
        risk_attr = pd.DataFrame({
            'Weight': avg_weights,
            'Marginal Risk': marginal_contrib,
            'Risk Contribution': risk_contrib,
            'Risk Contribution %': risk_contrib_pct
        }, index=self.asset_names)

        return risk_attr

    def performance_decomposition(
        self,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Comprehensive performance decomposition

        Args:
            sector_mapping: Optional sector mapping for sector attribution

        Returns:
            Dictionary with various attribution analyses
        """
        results = {}

        # Brinson attribution
        results['brinson'] = self.brinson_attribution().by_segment

        # Contribution to active return
        results['active_return'] = self.contribution_to_active_return()

        # Time series
        results['time_series'] = self.time_series_attribution()

        # Sector attribution if mapping provided
        if sector_mapping is not None:
            sector_attr = self.sector_attribution(sector_mapping)
            results['sector'] = sector_attr.total_by_sector

        return results


class MultiPeriodAttribution:
    """
    Multi-period attribution analysis with linking
    """

    def __init__(
        self,
        period_attributions: List[BrinsonAttribution]
    ):
        """
        Initialize multi-period attribution

        Args:
            period_attributions: List of period-level attributions
        """
        self.period_attributions = period_attributions

    def geometric_linking(self) -> BrinsonAttribution:
        """
        Link attribution effects geometrically across periods

        Returns:
            Linked BrinsonAttribution
        """
        # Geometric linking formula
        cumulative_allocation = 1.0
        cumulative_selection = 1.0
        cumulative_interaction = 1.0

        for attr in self.period_attributions:
            cumulative_allocation *= (1 + attr.allocation_effect)
            cumulative_selection *= (1 + attr.selection_effect)
            cumulative_interaction *= (1 + attr.interaction_effect)

        cumulative_allocation -= 1
        cumulative_selection -= 1
        cumulative_interaction -= 1

        total = cumulative_allocation + cumulative_selection + cumulative_interaction

        return BrinsonAttribution(
            allocation_effect=cumulative_allocation,
            selection_effect=cumulative_selection,
            interaction_effect=cumulative_interaction,
            total_effect=total
        )

    def arithmetic_linking(self) -> BrinsonAttribution:
        """
        Link attribution effects arithmetically across periods

        Returns:
            Linked BrinsonAttribution
        """
        total_allocation = sum(attr.allocation_effect for attr in self.period_attributions)
        total_selection = sum(attr.selection_effect for attr in self.period_attributions)
        total_interaction = sum(attr.interaction_effect for attr in self.period_attributions)
        total = total_allocation + total_selection + total_interaction

        return BrinsonAttribution(
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction,
            total_effect=total
        )


class StyleAttribution:
    """
    Style-based attribution analysis
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        style_indices: pd.DataFrame,
        rolling_window: int = 36
    ):
        """
        Initialize style attribution

        Args:
            portfolio_returns: Portfolio returns
            style_indices: Style index returns (e.g., value, growth, size)
            rolling_window: Window for rolling style analysis
        """
        self.portfolio_returns = portfolio_returns
        self.style_indices = style_indices
        self.rolling_window = rolling_window

    def style_exposures(self) -> pd.DataFrame:
        """
        Calculate rolling style exposures

        Returns:
            DataFrame with style exposures over time
        """
        exposures = pd.DataFrame(index=self.portfolio_returns.index, columns=self.style_indices.columns)

        for i in range(self.rolling_window, len(self.portfolio_returns)):
            y = self.portfolio_returns.iloc[i-self.rolling_window:i].values
            X = self.style_indices.iloc[i-self.rolling_window:i].values

            # Add constant
            X = np.column_stack([np.ones(len(X)), X])

            # Constrained regression (weights sum to 1, all positive)
            try:
                # Use simple OLS for now
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                exposures.iloc[i] = beta[1:]  # Exclude intercept
            except:
                exposures.iloc[i] = 0.0

        return exposures.astype(float)

    def style_contribution(self) -> pd.DataFrame:
        """
        Calculate contribution of each style to returns

        Returns:
            DataFrame with style contributions
        """
        exposures = self.style_exposures()

        # Calculate contribution
        contributions = pd.DataFrame(index=self.portfolio_returns.index, columns=self.style_indices.columns)

        for col in self.style_indices.columns:
            contributions[col] = exposures[col] * self.style_indices[col]

        # Add specific return
        explained_return = contributions.sum(axis=1)
        contributions['Specific'] = self.portfolio_returns - explained_return

        return contributions
