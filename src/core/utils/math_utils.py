"""
Mathematical and financial calculation utilities.
Includes risk metrics, returns calculations, and statistical functions.
"""

import logging
from typing import Union, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def calculate_returns(
    prices: Union[pd.Series, np.ndarray],
    method: str = "simple"
) -> Union[pd.Series, np.ndarray]:
    """
    Calculate returns from price series.

    Args:
        prices: Price series
        method: Return calculation method ("simple" or "log")

    Returns:
        Returns series
    """
    if isinstance(prices, pd.Series):
        if method == "simple":
            return prices.pct_change()
        elif method == "log":
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        if method == "simple":
            return np.diff(prices) / prices[:-1]
        elif method == "log":
            return np.log(prices[1:] / prices[:-1])
        else:
            raise ValueError(f"Unknown method: {method}")


def calculate_cumulative_returns(
    returns: Union[pd.Series, np.ndarray]
) -> Union[pd.Series, np.ndarray]:
    """
    Calculate cumulative returns.

    Args:
        returns: Returns series

    Returns:
        Cumulative returns
    """
    if isinstance(returns, pd.Series):
        return (1 + returns).cumprod() - 1
    else:
        return np.cumprod(1 + returns) - 1


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.045,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    # Calculate excess returns
    period_rf_rate = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf_rate

    # Calculate Sharpe ratio
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess == 0:
        return 0.0

    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.045,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio.

    Args:
        returns: Returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    # Calculate excess returns
    period_rf_rate = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf_rate

    # Calculate downside deviation
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0

    if downside_std == 0:
        return 0.0

    sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_max_drawdown(
    returns: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Returns series

    Returns:
        Maximum drawdown (negative value)
    """
    if len(returns) == 0:
        return 0.0

    # Calculate cumulative returns
    cum_returns = calculate_cumulative_returns(returns)

    if isinstance(cum_returns, pd.Series):
        cum_returns = cum_returns.values

    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)

    # Calculate drawdown
    drawdown = (cum_returns - running_max) / (1 + running_max)

    # Return maximum drawdown
    return float(np.min(drawdown))


def calculate_calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        returns: Returns series
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    # Annualized return
    total_return = np.prod(1 + returns) - 1
    years = len(returns) / periods_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1

    # Maximum drawdown
    max_dd = abs(calculate_max_drawdown(returns))

    if max_dd == 0:
        return 0.0

    return float(annual_return / max_dd)


def calculate_volatility(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Returns series
        periods_per_year: Number of periods per year

    Returns:
        Annualized volatility
    """
    if len(returns) == 0:
        return 0.0

    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    return float(vol)


def calculate_var(
    returns: Union[pd.Series, np.ndarray],
    confidence: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Returns series
        confidence: Confidence level

    Returns:
        VaR (negative value)
    """
    if len(returns) == 0:
        return 0.0

    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    var = np.percentile(returns, (1 - confidence) * 100)
    return float(var)


def calculate_cvar(
    returns: Union[pd.Series, np.ndarray],
    confidence: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

    Args:
        returns: Returns series
        confidence: Confidence level

    Returns:
        CVaR (negative value)
    """
    if len(returns) == 0:
        return 0.0

    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    var = calculate_var(returns, confidence)
    cvar = np.mean(returns[returns <= var])
    return float(cvar)


def calculate_beta(
    returns: Union[pd.Series, np.ndarray],
    market_returns: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate beta relative to market.

    Args:
        returns: Asset returns
        market_returns: Market returns

    Returns:
        Beta
    """
    if len(returns) == 0 or len(market_returns) == 0:
        return 1.0

    # Convert to numpy arrays
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(market_returns, pd.Series):
        market_returns = market_returns.values

    # Ensure same length
    min_len = min(len(returns), len(market_returns))
    returns = returns[:min_len]
    market_returns = market_returns[:min_len]

    # Remove NaN values
    mask = ~(np.isnan(returns) | np.isnan(market_returns))
    returns = returns[mask]
    market_returns = market_returns[mask]

    if len(returns) < 2:
        return 1.0

    # Calculate beta
    covariance = np.cov(returns, market_returns)[0, 1]
    market_variance = np.var(market_returns, ddof=1)

    if market_variance == 0:
        return 1.0

    beta = covariance / market_variance
    return float(beta)


def calculate_alpha(
    returns: Union[pd.Series, np.ndarray],
    market_returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.045,
    periods_per_year: int = 252
) -> float:
    """
    Calculate alpha (Jensen's alpha).

    Args:
        returns: Asset returns
        market_returns: Market returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Alpha
    """
    if len(returns) == 0 or len(market_returns) == 0:
        return 0.0

    # Convert to numpy arrays
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(market_returns, pd.Series):
        market_returns = market_returns.values

    # Ensure same length
    min_len = min(len(returns), len(market_returns))
    returns = returns[:min_len]
    market_returns = market_returns[:min_len]

    # Remove NaN values
    mask = ~(np.isnan(returns) | np.isnan(market_returns))
    returns = returns[mask]
    market_returns = market_returns[mask]

    if len(returns) < 2:
        return 0.0

    # Calculate beta
    beta = calculate_beta(returns, market_returns)

    # Period risk-free rate
    period_rf_rate = risk_free_rate / periods_per_year

    # Calculate alpha
    mean_return = np.mean(returns)
    mean_market_return = np.mean(market_returns)

    alpha = mean_return - (period_rf_rate + beta * (mean_market_return - period_rf_rate))
    return float(alpha) * periods_per_year


def calculate_information_ratio(
    returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate information ratio.

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Information ratio
    """
    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0

    # Convert to numpy arrays
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values

    # Ensure same length
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]

    # Remove NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns = returns[mask]
    benchmark_returns = benchmark_returns[mask]

    if len(returns) < 2:
        return 0.0

    # Tracking error (active returns)
    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns, ddof=1)

    if tracking_error == 0:
        return 0.0

    ir = np.mean(active_returns) / tracking_error
    return float(ir)


def calculate_omega_ratio(
    returns: Union[pd.Series, np.ndarray],
    threshold: float = 0.0
) -> float:
    """
    Calculate Omega ratio.

    Args:
        returns: Returns series
        threshold: Threshold return

    Returns:
        Omega ratio
    """
    if len(returns) == 0:
        return 1.0

    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 1.0

    # Calculate gains and losses relative to threshold
    excess_returns = returns - threshold
    gains = np.sum(excess_returns[excess_returns > 0])
    losses = np.abs(np.sum(excess_returns[excess_returns < 0]))

    if losses == 0:
        return np.inf if gains > 0 else 1.0

    omega = gains / losses
    return float(omega)


def calculate_tail_ratio(
    returns: Union[pd.Series, np.ndarray],
    percentile: int = 5
) -> float:
    """
    Calculate tail ratio (right tail / left tail).

    Args:
        returns: Returns series
        percentile: Percentile for tails

    Returns:
        Tail ratio
    """
    if len(returns) == 0:
        return 1.0

    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 1.0

    right_tail = np.percentile(returns, 100 - percentile)
    left_tail = np.abs(np.percentile(returns, percentile))

    if left_tail == 0:
        return np.inf if right_tail > 0 else 1.0

    return float(right_tail / left_tail)


def calculate_kelly_criterion(
    win_rate: float,
    win_loss_ratio: float
) -> float:
    """
    Calculate Kelly criterion for position sizing.

    Args:
        win_rate: Probability of winning
        win_loss_ratio: Average win / average loss

    Returns:
        Optimal fraction to risk
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0

    if win_loss_ratio <= 0:
        return 0.0

    kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    return max(0.0, min(1.0, float(kelly)))


def calculate_correlation(
    series1: Union[pd.Series, np.ndarray],
    series2: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate correlation between two series.

    Args:
        series1: First series
        series2: Second series

    Returns:
        Correlation coefficient
    """
    if len(series1) == 0 or len(series2) == 0:
        return 0.0

    # Convert to numpy arrays
    if isinstance(series1, pd.Series):
        series1 = series1.values
    if isinstance(series2, pd.Series):
        series2 = series2.values

    # Ensure same length
    min_len = min(len(series1), len(series2))
    series1 = series1[:min_len]
    series2 = series2[:min_len]

    # Remove NaN values
    mask = ~(np.isnan(series1) | np.isnan(series2))
    series1 = series1[mask]
    series2 = series2[mask]

    if len(series1) < 2:
        return 0.0

    corr = np.corrcoef(series1, series2)[0, 1]
    return float(corr)


def calculate_rolling_stat(
    series: pd.Series,
    window: int,
    stat: str = "mean"
) -> pd.Series:
    """
    Calculate rolling statistic.

    Args:
        series: Input series
        window: Window size
        stat: Statistic to calculate ("mean", "std", "max", "min")

    Returns:
        Rolling statistic series
    """
    if stat == "mean":
        return series.rolling(window=window).mean()
    elif stat == "std":
        return series.rolling(window=window).std()
    elif stat == "max":
        return series.rolling(window=window).max()
    elif stat == "min":
        return series.rolling(window=window).min()
    else:
        raise ValueError(f"Unknown statistic: {stat}")


def normalize_series(
    series: Union[pd.Series, np.ndarray],
    method: str = "zscore"
) -> Union[pd.Series, np.ndarray]:
    """
    Normalize series.

    Args:
        series: Input series
        method: Normalization method ("zscore", "minmax", "robust")

    Returns:
        Normalized series
    """
    if isinstance(series, pd.Series):
        values = series.values
        is_series = True
    else:
        values = series
        is_series = False

    # Remove NaN values for calculation
    mask = ~np.isnan(values)
    valid_values = values[mask]

    if len(valid_values) == 0:
        return series

    if method == "zscore":
        mean = np.mean(valid_values)
        std = np.std(valid_values, ddof=1)
        if std == 0:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - mean) / std

    elif method == "minmax":
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
        if max_val == min_val:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - min_val) / (max_val - min_val)

    elif method == "robust":
        median = np.median(valid_values)
        q75, q25 = np.percentile(valid_values, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - median) / iqr

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if is_series:
        return pd.Series(normalized, index=series.index)
    else:
        return normalized


def compound_returns(returns: List[float]) -> float:
    """
    Calculate compounded return from list of returns.

    Args:
        returns: List of returns

    Returns:
        Compounded return
    """
    if not returns:
        return 0.0

    compound = 1.0
    for r in returns:
        compound *= (1 + r)

    return compound - 1.0


def annualize_return(
    total_return: float,
    periods: int,
    periods_per_year: int = 252
) -> float:
    """
    Annualize a total return.

    Args:
        total_return: Total return
        periods: Number of periods
        periods_per_year: Periods per year

    Returns:
        Annualized return
    """
    if periods == 0:
        return 0.0

    years = periods / periods_per_year
    if years == 0:
        return 0.0

    annualized = (1 + total_return) ** (1 / years) - 1
    return float(annualized)
