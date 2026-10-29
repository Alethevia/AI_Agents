"""
Fundamental Features Extractor

This module provides comprehensive fundamental analysis features for securities.
Handles valuation, profitability, growth, and financial health metrics.

Author: AI Trading System
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List, Tuple
import warnings


class FundamentalFeatures:
    """
    Production-ready fundamental features calculator.

    Supports:
    - Batch computation across multiple symbols
    - Time series of fundamentals
    - Missing data handling
    - Sector/industry relative metrics
    """

    def __init__(self, symbol: Optional[str] = None):
        """
        Initialize fundamental features calculator.

        Args:
            symbol: Optional symbol identifier
        """
        self.symbol = symbol

    # ==================== Valuation Ratios ====================

    @staticmethod
    def price_to_earnings(price: Union[float, pd.Series],
                         earnings_per_share: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Price-to-Earnings (P/E) ratio.

        Args:
            price: Current stock price
            earnings_per_share: Earnings per share (TTM)

        Returns:
            P/E ratio
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pe = price / earnings_per_share
            if isinstance(pe, pd.Series):
                pe = pe.replace([np.inf, -np.inf], np.nan)
            return pe

    @staticmethod
    def price_to_book(price: Union[float, pd.Series],
                     book_value_per_share: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Price-to-Book (P/B) ratio.

        Args:
            price: Current stock price
            book_value_per_share: Book value per share

        Returns:
            P/B ratio
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pb = price / book_value_per_share
            if isinstance(pb, pd.Series):
                pb = pb.replace([np.inf, -np.inf], np.nan)
            return pb

    @staticmethod
    def price_to_sales(price: Union[float, pd.Series],
                      sales_per_share: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Price-to-Sales (P/S) ratio.

        Args:
            price: Current stock price
            sales_per_share: Revenue per share (TTM)

        Returns:
            P/S ratio
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ps = price / sales_per_share
            if isinstance(ps, pd.Series):
                ps = ps.replace([np.inf, -np.inf], np.nan)
            return ps

    @staticmethod
    def enterprise_value_to_ebitda(enterprise_value: Union[float, pd.Series],
                                  ebitda: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Enterprise Value to EBITDA ratio.

        Args:
            enterprise_value: Market cap + debt - cash
            ebitda: Earnings before interest, taxes, depreciation, amortization

        Returns:
            EV/EBITDA ratio
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev_ebitda = enterprise_value / ebitda
            if isinstance(ev_ebitda, pd.Series):
                ev_ebitda = ev_ebitda.replace([np.inf, -np.inf], np.nan)
            return ev_ebitda

    @staticmethod
    def price_to_cashflow(price: Union[float, pd.Series],
                         cashflow_per_share: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Price-to-Cash Flow (P/CF) ratio.

        Args:
            price: Current stock price
            cashflow_per_share: Operating cash flow per share

        Returns:
            P/CF ratio
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pcf = price / cashflow_per_share
            if isinstance(pcf, pd.Series):
                pcf = pcf.replace([np.inf, -np.inf], np.nan)
            return pcf

    @staticmethod
    def peg_ratio(pe_ratio: Union[float, pd.Series],
                 earnings_growth_rate: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Price/Earnings to Growth (PEG) ratio.

        Args:
            pe_ratio: P/E ratio
            earnings_growth_rate: Expected earnings growth rate (%)

        Returns:
            PEG ratio
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peg = pe_ratio / earnings_growth_rate
            if isinstance(peg, pd.Series):
                peg = peg.replace([np.inf, -np.inf], np.nan)
            return peg

    @staticmethod
    def dividend_yield(dividend_per_share: Union[float, pd.Series],
                      price: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate dividend yield.

        Args:
            dividend_per_share: Annual dividend per share
            price: Current stock price

        Returns:
            Dividend yield (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dy = (dividend_per_share / price) * 100
            if isinstance(dy, pd.Series):
                dy = dy.replace([np.inf, -np.inf], np.nan)
            return dy

    # ==================== Profitability Metrics ====================

    @staticmethod
    def return_on_equity(net_income: Union[float, pd.Series],
                        shareholders_equity: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Return on Equity (ROE).

        Args:
            net_income: Net income
            shareholders_equity: Shareholders' equity

        Returns:
            ROE (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roe = (net_income / shareholders_equity) * 100
            if isinstance(roe, pd.Series):
                roe = roe.replace([np.inf, -np.inf], np.nan)
            return roe

    @staticmethod
    def return_on_assets(net_income: Union[float, pd.Series],
                        total_assets: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Return on Assets (ROA).

        Args:
            net_income: Net income
            total_assets: Total assets

        Returns:
            ROA (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roa = (net_income / total_assets) * 100
            if isinstance(roa, pd.Series):
                roa = roa.replace([np.inf, -np.inf], np.nan)
            return roa

    @staticmethod
    def return_on_invested_capital(nopat: Union[float, pd.Series],
                                  invested_capital: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Return on Invested Capital (ROIC).

        Args:
            nopat: Net operating profit after tax
            invested_capital: Total debt + equity - cash

        Returns:
            ROIC (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roic = (nopat / invested_capital) * 100
            if isinstance(roic, pd.Series):
                roic = roic.replace([np.inf, -np.inf], np.nan)
            return roic

    @staticmethod
    def gross_profit_margin(gross_profit: Union[float, pd.Series],
                          revenue: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Gross Profit Margin.

        Args:
            gross_profit: Revenue - COGS
            revenue: Total revenue

        Returns:
            Gross margin (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            margin = (gross_profit / revenue) * 100
            if isinstance(margin, pd.Series):
                margin = margin.replace([np.inf, -np.inf], np.nan)
            return margin

    @staticmethod
    def operating_profit_margin(operating_income: Union[float, pd.Series],
                               revenue: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Operating Profit Margin.

        Args:
            operating_income: EBIT
            revenue: Total revenue

        Returns:
            Operating margin (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            margin = (operating_income / revenue) * 100
            if isinstance(margin, pd.Series):
                margin = margin.replace([np.inf, -np.inf], np.nan)
            return margin

    @staticmethod
    def net_profit_margin(net_income: Union[float, pd.Series],
                         revenue: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Net Profit Margin.

        Args:
            net_income: Net income
            revenue: Total revenue

        Returns:
            Net margin (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            margin = (net_income / revenue) * 100
            if isinstance(margin, pd.Series):
                margin = margin.replace([np.inf, -np.inf], np.nan)
            return margin

    @staticmethod
    def ebitda_margin(ebitda: Union[float, pd.Series],
                     revenue: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate EBITDA Margin.

        Args:
            ebitda: Earnings before interest, taxes, depreciation, amortization
            revenue: Total revenue

        Returns:
            EBITDA margin (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            margin = (ebitda / revenue) * 100
            if isinstance(margin, pd.Series):
                margin = margin.replace([np.inf, -np.inf], np.nan)
            return margin

    # ==================== Growth Metrics ====================

    @staticmethod
    def revenue_growth(current_revenue: Union[float, pd.Series],
                      previous_revenue: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate revenue growth rate.

        Args:
            current_revenue: Current period revenue
            previous_revenue: Previous period revenue

        Returns:
            Revenue growth rate (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            growth = ((current_revenue - previous_revenue) / previous_revenue) * 100
            if isinstance(growth, pd.Series):
                growth = growth.replace([np.inf, -np.inf], np.nan)
            return growth

    @staticmethod
    def earnings_growth(current_earnings: Union[float, pd.Series],
                       previous_earnings: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate earnings growth rate.

        Args:
            current_earnings: Current period earnings
            previous_earnings: Previous period earnings

        Returns:
            Earnings growth rate (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            growth = ((current_earnings - previous_earnings) / abs(previous_earnings)) * 100
            if isinstance(growth, pd.Series):
                growth = growth.replace([np.inf, -np.inf], np.nan)
            return growth

    @staticmethod
    def eps_growth(current_eps: Union[float, pd.Series],
                  previous_eps: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate EPS growth rate.

        Args:
            current_eps: Current period EPS
            previous_eps: Previous period EPS

        Returns:
            EPS growth rate (%)
        """
        return FundamentalFeatures.earnings_growth(current_eps, previous_eps)

    @staticmethod
    def cagr(ending_value: Union[float, pd.Series],
            beginning_value: Union[float, pd.Series],
            num_periods: Union[int, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Compound Annual Growth Rate (CAGR).

        Args:
            ending_value: Ending value
            beginning_value: Beginning value
            num_periods: Number of periods (years)

        Returns:
            CAGR (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(ending_value, pd.Series):
                cagr = ((ending_value / beginning_value) ** (1 / num_periods) - 1) * 100
                cagr = cagr.replace([np.inf, -np.inf], np.nan)
            else:
                cagr = ((ending_value / beginning_value) ** (1 / num_periods) - 1) * 100
            return cagr

    # ==================== Financial Health ====================

    @staticmethod
    def debt_to_equity(total_debt: Union[float, pd.Series],
                      shareholders_equity: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Debt-to-Equity ratio.

        Args:
            total_debt: Total debt
            shareholders_equity: Shareholders' equity

        Returns:
            D/E ratio
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            de = total_debt / shareholders_equity
            if isinstance(de, pd.Series):
                de = de.replace([np.inf, -np.inf], np.nan)
            return de

    @staticmethod
    def debt_to_assets(total_debt: Union[float, pd.Series],
                      total_assets: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Debt-to-Assets ratio.

        Args:
            total_debt: Total debt
            total_assets: Total assets

        Returns:
            Debt-to-Assets ratio (%)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            da = (total_debt / total_assets) * 100
            if isinstance(da, pd.Series):
                da = da.replace([np.inf, -np.inf], np.nan)
            return da

    @staticmethod
    def current_ratio(current_assets: Union[float, pd.Series],
                     current_liabilities: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Current Ratio.

        Args:
            current_assets: Current assets
            current_liabilities: Current liabilities

        Returns:
            Current ratio
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cr = current_assets / current_liabilities
            if isinstance(cr, pd.Series):
                cr = cr.replace([np.inf, -np.inf], np.nan)
            return cr

    @staticmethod
    def quick_ratio(current_assets: Union[float, pd.Series],
                   inventory: Union[float, pd.Series],
                   current_liabilities: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Quick Ratio (Acid Test).

        Args:
            current_assets: Current assets
            inventory: Inventory
            current_liabilities: Current liabilities

        Returns:
            Quick ratio
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qr = (current_assets - inventory) / current_liabilities
            if isinstance(qr, pd.Series):
                qr = qr.replace([np.inf, -np.inf], np.nan)
            return qr

    @staticmethod
    def cash_ratio(cash_and_equivalents: Union[float, pd.Series],
                  current_liabilities: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Cash Ratio.

        Args:
            cash_and_equivalents: Cash and cash equivalents
            current_liabilities: Current liabilities

        Returns:
            Cash ratio
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cr = cash_and_equivalents / current_liabilities
            if isinstance(cr, pd.Series):
                cr = cr.replace([np.inf, -np.inf], np.nan)
            return cr

    @staticmethod
    def interest_coverage(ebit: Union[float, pd.Series],
                         interest_expense: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate Interest Coverage Ratio.

        Args:
            ebit: Earnings before interest and taxes
            interest_expense: Interest expense

        Returns:
            Interest coverage ratio
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic = ebit / interest_expense
            if isinstance(ic, pd.Series):
                ic = ic.replace([np.inf, -np.inf], np.nan)
            return ic

    # ==================== Quality Scores ====================

    @staticmethod
    def piotroski_f_score(financial_data: Dict[str, float]) -> int:
        """
        Calculate Piotroski F-Score (0-9).

        Args:
            financial_data: Dictionary with required financial metrics

        Returns:
            F-Score (0-9)
        """
        score = 0

        # Profitability (4 points)
        if financial_data.get('net_income', 0) > 0:
            score += 1
        if financial_data.get('operating_cashflow', 0) > 0:
            score += 1
        if financial_data.get('roa', 0) > financial_data.get('roa_previous', -999):
            score += 1
        if financial_data.get('operating_cashflow', 0) > financial_data.get('net_income', 0):
            score += 1

        # Leverage, Liquidity, and Source of Funds (3 points)
        if financial_data.get('debt_ratio', 1) < financial_data.get('debt_ratio_previous', 1):
            score += 1
        if financial_data.get('current_ratio', 0) > financial_data.get('current_ratio_previous', 0):
            score += 1
        if financial_data.get('shares_outstanding', 1) <= financial_data.get('shares_outstanding_previous', 1):
            score += 1

        # Operating Efficiency (2 points)
        if financial_data.get('gross_margin', 0) > financial_data.get('gross_margin_previous', 0):
            score += 1
        if financial_data.get('asset_turnover', 0) > financial_data.get('asset_turnover_previous', 0):
            score += 1

        return score

    @staticmethod
    def altman_z_score(working_capital: float, total_assets: float, retained_earnings: float,
                      ebit: float, market_value_equity: float, total_liabilities: float,
                      sales: float, is_public: bool = True) -> float:
        """
        Calculate Altman Z-Score for bankruptcy prediction.

        Args:
            working_capital: Current assets - current liabilities
            total_assets: Total assets
            retained_earnings: Retained earnings
            ebit: Earnings before interest and taxes
            market_value_equity: Market value of equity
            total_liabilities: Total liabilities
            sales: Total sales/revenue
            is_public: Whether company is publicly traded

        Returns:
            Z-Score (>2.99: safe, 1.81-2.99: gray, <1.81: distress)
        """
        if total_assets == 0:
            return np.nan

        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_value_equity / total_liabilities if total_liabilities > 0 else 0
        x5 = sales / total_assets

        if is_public:
            # Original Z-Score for public companies
            z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        else:
            # Z'-Score for private companies
            z_score = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5

        return z_score

    @staticmethod
    def quality_score(roe: float, roa: float, gross_margin: float,
                     operating_margin: float, debt_to_equity: float,
                     current_ratio: float) -> float:
        """
        Calculate composite quality score (0-100).

        Args:
            roe: Return on equity (%)
            roa: Return on assets (%)
            gross_margin: Gross profit margin (%)
            operating_margin: Operating margin (%)
            debt_to_equity: Debt-to-equity ratio
            current_ratio: Current ratio

        Returns:
            Quality score (0-100)
        """
        # Normalize and weight components
        score = 0

        # Profitability (40%)
        roe_score = min(max(roe / 20 * 100, 0), 100)  # 20% ROE = full points
        roa_score = min(max(roa / 10 * 100, 0), 100)  # 10% ROA = full points
        score += (roe_score * 0.2 + roa_score * 0.2)

        # Margins (30%)
        gm_score = min(max(gross_margin / 40 * 100, 0), 100)  # 40% = full points
        om_score = min(max(operating_margin / 20 * 100, 0), 100)  # 20% = full points
        score += (gm_score * 0.15 + om_score * 0.15)

        # Financial Health (30%)
        # Lower debt is better
        de_score = max(100 - (debt_to_equity * 20), 0)  # 0-5 D/E scale
        # Current ratio: 1.5-3.0 is ideal
        cr_score = 100 if 1.5 <= current_ratio <= 3.0 else max(100 - abs(current_ratio - 2.0) * 30, 0)
        score += (de_score * 0.15 + cr_score * 0.15)

        return min(max(score, 0), 100)

    # ==================== Batch Processing ====================

    def calculate_all_fundamentals(self, data: Dict[str, Union[float, pd.Series]]) -> Dict[str, Union[float, pd.Series]]:
        """
        Calculate all fundamental features from financial data.

        Args:
            data: Dictionary containing financial statement data

        Returns:
            Dictionary with all calculated fundamental features
        """
        results = {}

        try:
            # Valuation Ratios
            if 'price' in data and 'eps' in data:
                results['pe_ratio'] = self.price_to_earnings(data['price'], data['eps'])

            if 'price' in data and 'book_value_per_share' in data:
                results['pb_ratio'] = self.price_to_book(data['price'], data['book_value_per_share'])

            if 'price' in data and 'sales_per_share' in data:
                results['ps_ratio'] = self.price_to_sales(data['price'], data['sales_per_share'])

            if 'enterprise_value' in data and 'ebitda' in data:
                results['ev_ebitda'] = self.enterprise_value_to_ebitda(data['enterprise_value'], data['ebitda'])

            if 'price' in data and 'cashflow_per_share' in data:
                results['pcf_ratio'] = self.price_to_cashflow(data['price'], data['cashflow_per_share'])

            if 'dividend_per_share' in data and 'price' in data:
                results['dividend_yield'] = self.dividend_yield(data['dividend_per_share'], data['price'])

            # Profitability
            if 'net_income' in data and 'shareholders_equity' in data:
                results['roe'] = self.return_on_equity(data['net_income'], data['shareholders_equity'])

            if 'net_income' in data and 'total_assets' in data:
                results['roa'] = self.return_on_assets(data['net_income'], data['total_assets'])

            if 'gross_profit' in data and 'revenue' in data:
                results['gross_margin'] = self.gross_profit_margin(data['gross_profit'], data['revenue'])

            if 'operating_income' in data and 'revenue' in data:
                results['operating_margin'] = self.operating_profit_margin(data['operating_income'], data['revenue'])

            if 'net_income' in data and 'revenue' in data:
                results['net_margin'] = self.net_profit_margin(data['net_income'], data['revenue'])

            if 'ebitda' in data and 'revenue' in data:
                results['ebitda_margin'] = self.ebitda_margin(data['ebitda'], data['revenue'])

            # Growth (requires historical data)
            if 'revenue' in data and 'revenue_previous' in data:
                results['revenue_growth'] = self.revenue_growth(data['revenue'], data['revenue_previous'])

            if 'net_income' in data and 'net_income_previous' in data:
                results['earnings_growth'] = self.earnings_growth(data['net_income'], data['net_income_previous'])

            if 'eps' in data and 'eps_previous' in data:
                results['eps_growth'] = self.eps_growth(data['eps'], data['eps_previous'])

            # Financial Health
            if 'total_debt' in data and 'shareholders_equity' in data:
                results['debt_to_equity'] = self.debt_to_equity(data['total_debt'], data['shareholders_equity'])

            if 'total_debt' in data and 'total_assets' in data:
                results['debt_to_assets'] = self.debt_to_assets(data['total_debt'], data['total_assets'])

            if 'current_assets' in data and 'current_liabilities' in data:
                results['current_ratio'] = self.current_ratio(data['current_assets'], data['current_liabilities'])

            if 'current_assets' in data and 'inventory' in data and 'current_liabilities' in data:
                results['quick_ratio'] = self.quick_ratio(
                    data['current_assets'], data['inventory'], data['current_liabilities']
                )

            if 'cash' in data and 'current_liabilities' in data:
                results['cash_ratio'] = self.cash_ratio(data['cash'], data['current_liabilities'])

            if 'ebit' in data and 'interest_expense' in data:
                results['interest_coverage'] = self.interest_coverage(data['ebit'], data['interest_expense'])

        except Exception as e:
            warnings.warn(f"Error calculating fundamentals: {str(e)}")

        return results

    @staticmethod
    def calculate_relative_metrics(data: pd.DataFrame, group_by: str = 'sector') -> pd.DataFrame:
        """
        Calculate sector/industry relative metrics.

        Args:
            data: DataFrame with fundamentals and grouping column
            group_by: Column to group by (e.g., 'sector', 'industry')

        Returns:
            DataFrame with relative metrics added
        """
        df = data.copy()

        # Metrics to calculate relative values for
        metrics = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'roa',
                  'gross_margin', 'operating_margin', 'debt_to_equity']

        for metric in metrics:
            if metric in df.columns:
                # Calculate percentile rank within group
                df[f'{metric}_percentile'] = df.groupby(group_by)[metric].rank(pct=True) * 100

                # Calculate z-score within group
                group_mean = df.groupby(group_by)[metric].transform('mean')
                group_std = df.groupby(group_by)[metric].transform('std')
                df[f'{metric}_zscore'] = (df[metric] - group_mean) / group_std.replace(0, 1)

                # Calculate relative to group median
                group_median = df.groupby(group_by)[metric].transform('median')
                df[f'{metric}_relative'] = df[metric] / group_median.replace(0, 1)

        return df
