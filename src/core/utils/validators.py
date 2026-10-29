"""
Data validation utilities for trading system.
Validates orders, prices, positions, and other trading data.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from decimal import Decimal
import re

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class OrderValidator:
    """Validates trading orders."""

    VALID_SIDES = ["buy", "sell", "long", "short"]
    VALID_ORDER_TYPES = ["market", "limit", "stop", "stop_limit"]
    VALID_TIME_IN_FORCE = ["day", "gtc", "ioc", "fok", "opg", "cls"]

    @staticmethod
    def validate_order(order: Dict[str, Any]) -> bool:
        """
        Validate order data.

        Args:
            order: Order dictionary

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        # Required fields
        required_fields = ["symbol", "side", "quantity", "order_type"]
        for field in required_fields:
            if field not in order:
                raise ValidationError(f"Missing required field: {field}")

        # Validate symbol
        if not OrderValidator.validate_symbol(order["symbol"]):
            raise ValidationError(f"Invalid symbol: {order['symbol']}")

        # Validate side
        if order["side"].lower() not in OrderValidator.VALID_SIDES:
            raise ValidationError(
                f"Invalid side: {order['side']}. "
                f"Must be one of {OrderValidator.VALID_SIDES}"
            )

        # Validate quantity
        if not OrderValidator.validate_quantity(order["quantity"]):
            raise ValidationError(f"Invalid quantity: {order['quantity']}")

        # Validate order type
        if order["order_type"].lower() not in OrderValidator.VALID_ORDER_TYPES:
            raise ValidationError(
                f"Invalid order type: {order['order_type']}. "
                f"Must be one of {OrderValidator.VALID_ORDER_TYPES}"
            )

        # Validate price for limit/stop orders
        if order["order_type"].lower() in ["limit", "stop", "stop_limit"]:
            if "price" not in order:
                raise ValidationError(
                    f"Price required for {order['order_type']} order"
                )
            if not OrderValidator.validate_price(order["price"]):
                raise ValidationError(f"Invalid price: {order['price']}")

        # Validate time in force
        if "time_in_force" in order:
            if order["time_in_force"].lower() not in OrderValidator.VALID_TIME_IN_FORCE:
                raise ValidationError(
                    f"Invalid time_in_force: {order['time_in_force']}. "
                    f"Must be one of {OrderValidator.VALID_TIME_IN_FORCE}"
                )

        return True

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """
        Validate trading symbol.

        Args:
            symbol: Symbol string

        Returns:
            True if valid
        """
        if not symbol or not isinstance(symbol, str):
            return False

        # Basic symbol format: 1-10 uppercase letters
        pattern = r"^[A-Z]{1,10}$"
        return bool(re.match(pattern, symbol.upper()))

    @staticmethod
    def validate_quantity(quantity: Union[int, float]) -> bool:
        """
        Validate order quantity.

        Args:
            quantity: Order quantity

        Returns:
            True if valid
        """
        try:
            qty = float(quantity)
            return qty > 0 and qty < 1e10
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_price(price: Union[int, float]) -> bool:
        """
        Validate price.

        Args:
            price: Price value

        Returns:
            True if valid
        """
        try:
            p = float(price)
            return p > 0 and p < 1e10
        except (ValueError, TypeError):
            return False


class PositionValidator:
    """Validates position data."""

    @staticmethod
    def validate_position(position: Dict[str, Any]) -> bool:
        """
        Validate position data.

        Args:
            position: Position dictionary

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        # Required fields
        required_fields = ["symbol", "quantity", "avg_entry_price"]
        for field in required_fields:
            if field not in position:
                raise ValidationError(f"Missing required field: {field}")

        # Validate symbol
        if not OrderValidator.validate_symbol(position["symbol"]):
            raise ValidationError(f"Invalid symbol: {position['symbol']}")

        # Validate quantity (can be negative for short positions)
        try:
            qty = float(position["quantity"])
            if abs(qty) > 1e10:
                raise ValidationError(f"Quantity too large: {qty}")
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid quantity: {position['quantity']}")

        # Validate avg entry price
        if not OrderValidator.validate_price(position["avg_entry_price"]):
            raise ValidationError(
                f"Invalid avg_entry_price: {position['avg_entry_price']}"
            )

        # Validate unrealized PnL if present
        if "unrealized_pnl" in position:
            try:
                float(position["unrealized_pnl"])
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Invalid unrealized_pnl: {position['unrealized_pnl']}"
                )

        return True


class PriceValidator:
    """Validates price data."""

    @staticmethod
    def validate_price_data(
        data: Union[pd.DataFrame, Dict[str, Any]],
        required_fields: Optional[List[str]] = None
    ) -> bool:
        """
        Validate price data.

        Args:
            data: Price data (DataFrame or dict)
            required_fields: Required field names

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        if required_fields is None:
            required_fields = ["open", "high", "low", "close", "volume"]

        if isinstance(data, pd.DataFrame):
            # Check required columns
            for field in required_fields:
                if field not in data.columns:
                    raise ValidationError(f"Missing required column: {field}")

            # Validate OHLC relationship
            if all(f in data.columns for f in ["open", "high", "low", "close"]):
                invalid_bars = (
                    (data["high"] < data["low"]) |
                    (data["high"] < data["open"]) |
                    (data["high"] < data["close"]) |
                    (data["low"] > data["open"]) |
                    (data["low"] > data["close"])
                )

                if invalid_bars.any():
                    count = invalid_bars.sum()
                    raise ValidationError(
                        f"Found {count} invalid OHLC bars where "
                        "high < low or high < open/close or low > open/close"
                    )

            # Validate non-negative prices
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if col in data.columns:
                    if (data[col] < 0).any():
                        raise ValidationError(f"Negative prices found in {col}")

            # Validate non-negative volume
            if "volume" in data.columns:
                if (data["volume"] < 0).any():
                    raise ValidationError("Negative volume found")

        elif isinstance(data, dict):
            # Check required fields
            for field in required_fields:
                if field not in data:
                    raise ValidationError(f"Missing required field: {field}")

            # Validate OHLC relationship
            if all(f in data for f in ["open", "high", "low", "close"]):
                o, h, l, c = data["open"], data["high"], data["low"], data["close"]
                if h < l or h < o or h < c or l > o or l > c:
                    raise ValidationError(
                        "Invalid OHLC: high < low or high < open/close or low > open/close"
                    )

            # Validate non-negative prices
            price_fields = ["open", "high", "low", "close"]
            for field in price_fields:
                if field in data and data[field] < 0:
                    raise ValidationError(f"Negative price in {field}")

            # Validate non-negative volume
            if "volume" in data and data["volume"] < 0:
                raise ValidationError("Negative volume")

        else:
            raise ValidationError("Data must be DataFrame or dict")

        return True

    @staticmethod
    def validate_tick_data(tick: Dict[str, Any]) -> bool:
        """
        Validate tick data.

        Args:
            tick: Tick dictionary

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        # Required fields
        required_fields = ["symbol", "price", "size", "timestamp"]
        for field in required_fields:
            if field not in tick:
                raise ValidationError(f"Missing required field: {field}")

        # Validate symbol
        if not OrderValidator.validate_symbol(tick["symbol"]):
            raise ValidationError(f"Invalid symbol: {tick['symbol']}")

        # Validate price
        if not OrderValidator.validate_price(tick["price"]):
            raise ValidationError(f"Invalid price: {tick['price']}")

        # Validate size
        try:
            size = float(tick["size"])
            if size <= 0:
                raise ValidationError(f"Invalid size: {size}")
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid size: {tick['size']}")

        # Validate timestamp
        if not isinstance(tick["timestamp"], (datetime, pd.Timestamp, int, float)):
            raise ValidationError(f"Invalid timestamp type: {type(tick['timestamp'])}")

        return True


class RiskValidator:
    """Validates risk parameters."""

    @staticmethod
    def validate_risk_limits(limits: Dict[str, Any]) -> bool:
        """
        Validate risk limits.

        Args:
            limits: Risk limits dictionary

        Returns:
            True if valid

        Raises:
            ValidationError: If validation fails
        """
        # Validate max position size
        if "max_position_size" in limits:
            try:
                size = float(limits["max_position_size"])
                if size <= 0:
                    raise ValidationError(
                        f"max_position_size must be positive: {size}"
                    )
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Invalid max_position_size: {limits['max_position_size']}"
                )

        # Validate max drawdown
        if "max_drawdown" in limits:
            try:
                dd = float(limits["max_drawdown"])
                if dd < 0 or dd > 1:
                    raise ValidationError(
                        f"max_drawdown must be between 0 and 1: {dd}"
                    )
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Invalid max_drawdown: {limits['max_drawdown']}"
                )

        # Validate max leverage
        if "max_leverage" in limits:
            try:
                lev = float(limits["max_leverage"])
                if lev < 1:
                    raise ValidationError(f"max_leverage must be >= 1: {lev}")
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Invalid max_leverage: {limits['max_leverage']}"
                )

        # Validate stop loss percentage
        if "stop_loss_pct" in limits:
            try:
                sl = float(limits["stop_loss_pct"])
                if sl < 0 or sl > 1:
                    raise ValidationError(
                        f"stop_loss_pct must be between 0 and 1: {sl}"
                    )
            except (ValueError, TypeError):
                raise ValidationError(
                    f"Invalid stop_loss_pct: {limits['stop_loss_pct']}"
                )

        return True


class DataQualityValidator:
    """Validates data quality."""

    @staticmethod
    def check_missing_values(
        data: pd.DataFrame,
        threshold: float = 0.1
    ) -> bool:
        """
        Check for missing values in DataFrame.

        Args:
            data: Input DataFrame
            threshold: Maximum allowed missing value ratio

        Returns:
            True if within threshold

        Raises:
            ValidationError: If too many missing values
        """
        missing_ratio = data.isnull().sum() / len(data)
        problematic = missing_ratio[missing_ratio > threshold]

        if len(problematic) > 0:
            raise ValidationError(
                f"Columns with too many missing values: "
                f"{problematic.to_dict()}"
            )

        return True

    @staticmethod
    def check_outliers(
        series: pd.Series,
        n_std: float = 5.0
    ) -> bool:
        """
        Check for outliers in series.

        Args:
            series: Input series
            n_std: Number of standard deviations for outlier detection

        Returns:
            True if no extreme outliers

        Raises:
            ValidationError: If extreme outliers found
        """
        mean = series.mean()
        std = series.std()

        outliers = np.abs(series - mean) > n_std * std
        outlier_count = outliers.sum()

        if outlier_count > 0:
            raise ValidationError(
                f"Found {outlier_count} outliers beyond {n_std} std deviations"
            )

        return True

    @staticmethod
    def check_duplicates(
        data: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> bool:
        """
        Check for duplicate rows.

        Args:
            data: Input DataFrame
            subset: Column subset to check

        Returns:
            True if no duplicates

        Raises:
            ValidationError: If duplicates found
        """
        duplicates = data.duplicated(subset=subset)
        duplicate_count = duplicates.sum()

        if duplicate_count > 0:
            raise ValidationError(f"Found {duplicate_count} duplicate rows")

        return True

    @staticmethod
    def check_data_types(
        data: pd.DataFrame,
        expected_types: Dict[str, type]
    ) -> bool:
        """
        Check DataFrame column data types.

        Args:
            data: Input DataFrame
            expected_types: Expected types for columns

        Returns:
            True if types match

        Raises:
            ValidationError: If types don't match
        """
        mismatches = []

        for col, expected_type in expected_types.items():
            if col not in data.columns:
                mismatches.append(f"Missing column: {col}")
                continue

            actual_type = data[col].dtype
            if not np.issubdtype(actual_type, expected_type):
                mismatches.append(
                    f"{col}: expected {expected_type}, got {actual_type}"
                )

        if mismatches:
            raise ValidationError(f"Type mismatches: {', '.join(mismatches)}")

        return True

    @staticmethod
    def check_time_series_gaps(
        data: pd.DataFrame,
        time_column: str = "timestamp",
        expected_freq: str = "1min",
        max_gap_multiple: int = 5
    ) -> bool:
        """
        Check for gaps in time series data.

        Args:
            data: Input DataFrame
            time_column: Name of time column
            expected_freq: Expected frequency
            max_gap_multiple: Maximum allowed gap as multiple of frequency

        Returns:
            True if no large gaps

        Raises:
            ValidationError: If large gaps found
        """
        if time_column not in data.columns:
            raise ValidationError(f"Time column not found: {time_column}")

        # Sort by time
        data_sorted = data.sort_values(time_column)

        # Calculate time differences
        time_diffs = data_sorted[time_column].diff()

        # Convert expected frequency to timedelta
        freq_delta = pd.Timedelta(expected_freq)

        # Find large gaps
        large_gaps = time_diffs > freq_delta * max_gap_multiple
        gap_count = large_gaps.sum()

        if gap_count > 0:
            raise ValidationError(
                f"Found {gap_count} large gaps in time series "
                f"(>{max_gap_multiple}x expected frequency)"
            )

        return True


def validate_percentage(value: float, name: str = "value") -> bool:
    """
    Validate percentage value.

    Args:
        value: Value to validate
        name: Value name for error message

    Returns:
        True if valid

    Raises:
        ValidationError: If invalid
    """
    if not 0 <= value <= 1:
        raise ValidationError(f"{name} must be between 0 and 1: {value}")
    return True


def validate_positive(value: float, name: str = "value") -> bool:
    """
    Validate positive value.

    Args:
        value: Value to validate
        name: Value name for error message

    Returns:
        True if valid

    Raises:
        ValidationError: If invalid
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive: {value}")
    return True


def validate_non_negative(value: float, name: str = "value") -> bool:
    """
    Validate non-negative value.

    Args:
        value: Value to validate
        name: Value name for error message

    Returns:
        True if valid

    Raises:
        ValidationError: If invalid
    """
    if value < 0:
        raise ValidationError(f"{name} must be non-negative: {value}")
    return True


def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value"
) -> bool:
    """
    Validate value is within range.

    Args:
        value: Value to validate
        min_val: Minimum value
        max_val: Maximum value
        name: Value name for error message

    Returns:
        True if valid

    Raises:
        ValidationError: If invalid
    """
    if not min_val <= value <= max_val:
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}: {value}"
        )
    return True
