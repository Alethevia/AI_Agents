"""
Market calendar and date/time utilities.
Handles trading hours, market holidays, and timezone conversions.
"""

import logging
from datetime import datetime, time, timedelta, date
from typing import Optional, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# US Market Holidays (NYSE/NASDAQ)
US_MARKET_HOLIDAYS = {
    # New Year's Day
    "2024-01-01": "New Year's Day",
    "2025-01-01": "New Year's Day",
    "2026-01-01": "New Year's Day",

    # Martin Luther King Jr. Day (3rd Monday of January)
    "2024-01-15": "Martin Luther King Jr. Day",
    "2025-01-20": "Martin Luther King Jr. Day",
    "2026-01-19": "Martin Luther King Jr. Day",

    # Presidents' Day (3rd Monday of February)
    "2024-02-19": "Presidents' Day",
    "2025-02-17": "Presidents' Day",
    "2026-02-16": "Presidents' Day",

    # Good Friday
    "2024-03-29": "Good Friday",
    "2025-04-18": "Good Friday",
    "2026-04-03": "Good Friday",

    # Memorial Day (Last Monday of May)
    "2024-05-27": "Memorial Day",
    "2025-05-26": "Memorial Day",
    "2026-05-25": "Memorial Day",

    # Juneteenth
    "2024-06-19": "Juneteenth",
    "2025-06-19": "Juneteenth",
    "2026-06-19": "Juneteenth",

    # Independence Day
    "2024-07-04": "Independence Day",
    "2025-07-04": "Independence Day",
    "2026-07-03": "Independence Day (Observed)",  # Falls on Saturday

    # Labor Day (1st Monday of September)
    "2024-09-02": "Labor Day",
    "2025-09-01": "Labor Day",
    "2026-09-07": "Labor Day",

    # Thanksgiving Day (4th Thursday of November)
    "2024-11-28": "Thanksgiving Day",
    "2025-11-27": "Thanksgiving Day",
    "2026-11-26": "Thanksgiving Day",

    # Christmas Day
    "2024-12-25": "Christmas Day",
    "2025-12-25": "Christmas Day",
    "2026-12-25": "Christmas Day",
}

# Early close days (1:00 PM ET close)
EARLY_CLOSE_DAYS = {
    "2024-07-03": "Day before Independence Day",
    "2024-11-29": "Day after Thanksgiving",
    "2024-12-24": "Christmas Eve",
    "2025-07-03": "Day before Independence Day",
    "2025-11-28": "Day after Thanksgiving",
    "2025-12-24": "Christmas Eve",
    "2026-11-27": "Day after Thanksgiving",
    "2026-12-24": "Christmas Eve",
}


class MarketCalendar:
    """Market calendar for trading hours and holidays."""

    def __init__(
        self,
        market_timezone: str = "America/New_York",
        regular_open: time = time(9, 30),
        regular_close: time = time(16, 0),
        early_close: time = time(13, 0)
    ):
        """
        Initialize market calendar.

        Args:
            market_timezone: Market timezone
            regular_open: Regular market open time
            regular_close: Regular market close time
            early_close: Early close time
        """
        self.market_tz = ZoneInfo(market_timezone)
        self.regular_open = regular_open
        self.regular_close = regular_close
        self.early_close = early_close
        self.holidays = US_MARKET_HOLIDAYS
        self.early_close_days = EARLY_CLOSE_DAYS

    def is_market_holiday(self, dt: datetime) -> bool:
        """
        Check if date is a market holiday.

        Args:
            dt: Date to check

        Returns:
            True if market holiday
        """
        date_str = dt.strftime("%Y-%m-%d")
        return date_str in self.holidays

    def is_early_close_day(self, dt: datetime) -> bool:
        """
        Check if date is an early close day.

        Args:
            dt: Date to check

        Returns:
            True if early close day
        """
        date_str = dt.strftime("%Y-%m-%d")
        return date_str in self.early_close_days

    def is_trading_day(self, dt: datetime) -> bool:
        """
        Check if date is a trading day.

        Args:
            dt: Date to check

        Returns:
            True if trading day
        """
        # Weekend
        if dt.weekday() >= 5:
            return False

        # Holiday
        if self.is_market_holiday(dt):
            return False

        return True

    def get_market_open(self, dt: datetime) -> Optional[datetime]:
        """
        Get market open datetime for a given date.

        Args:
            dt: Date to check

        Returns:
            Market open datetime or None if not a trading day
        """
        if not self.is_trading_day(dt):
            return None

        return datetime.combine(
            dt.date(),
            self.regular_open,
            tzinfo=self.market_tz
        )

    def get_market_close(self, dt: datetime) -> Optional[datetime]:
        """
        Get market close datetime for a given date.

        Args:
            dt: Date to check

        Returns:
            Market close datetime or None if not a trading day
        """
        if not self.is_trading_day(dt):
            return None

        close_time = (
            self.early_close if self.is_early_close_day(dt)
            else self.regular_close
        )

        return datetime.combine(
            dt.date(),
            close_time,
            tzinfo=self.market_tz
        )

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open.

        Args:
            dt: Datetime to check (default: now)

        Returns:
            True if market is open
        """
        if dt is None:
            dt = datetime.now(self.market_tz)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.market_tz)
        else:
            dt = dt.astimezone(self.market_tz)

        if not self.is_trading_day(dt):
            return False

        market_open = self.get_market_open(dt)
        market_close = self.get_market_close(dt)

        return market_open <= dt < market_close

    def get_next_trading_day(self, dt: datetime) -> datetime:
        """
        Get next trading day after given date.

        Args:
            dt: Starting date

        Returns:
            Next trading day
        """
        next_day = dt + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

    def get_previous_trading_day(self, dt: datetime) -> datetime:
        """
        Get previous trading day before given date.

        Args:
            dt: Starting date

        Returns:
            Previous trading day
        """
        prev_day = dt - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day

    def get_trading_days(
        self,
        start: datetime,
        end: datetime
    ) -> List[datetime]:
        """
        Get list of trading days between start and end dates.

        Args:
            start: Start date
            end: End date

        Returns:
            List of trading days
        """
        trading_days = []
        current = start

        while current <= end:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days

    def get_trading_hours(
        self,
        dt: Optional[datetime] = None
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Get trading hours for a given date.

        Args:
            dt: Date to check (default: today)

        Returns:
            Tuple of (open, close) or None if not trading day
        """
        if dt is None:
            dt = datetime.now(self.market_tz)

        market_open = self.get_market_open(dt)
        market_close = self.get_market_close(dt)

        if market_open is None or market_close is None:
            return None

        return (market_open, market_close)

    def time_until_open(self, dt: Optional[datetime] = None) -> Optional[timedelta]:
        """
        Get time until market opens.

        Args:
            dt: Reference datetime (default: now)

        Returns:
            Time until open or None if currently open
        """
        if dt is None:
            dt = datetime.now(self.market_tz)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.market_tz)
        else:
            dt = dt.astimezone(self.market_tz)

        if self.is_market_open(dt):
            return None

        # Check today
        market_open = self.get_market_open(dt)
        if market_open and dt < market_open:
            return market_open - dt

        # Check next trading day
        next_day = self.get_next_trading_day(dt)
        market_open = self.get_market_open(next_day)
        return market_open - dt

    def time_until_close(self, dt: Optional[datetime] = None) -> Optional[timedelta]:
        """
        Get time until market closes.

        Args:
            dt: Reference datetime (default: now)

        Returns:
            Time until close or None if currently closed
        """
        if dt is None:
            dt = datetime.now(self.market_tz)
        elif dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.market_tz)
        else:
            dt = dt.astimezone(self.market_tz)

        if not self.is_market_open(dt):
            return None

        market_close = self.get_market_close(dt)
        return market_close - dt


def get_market_calendar() -> MarketCalendar:
    """
    Get market calendar instance.

    Returns:
        MarketCalendar instance
    """
    return MarketCalendar()


def is_trading_hours(dt: Optional[datetime] = None) -> bool:
    """
    Check if currently in trading hours.

    Args:
        dt: Datetime to check (default: now)

    Returns:
        True if in trading hours
    """
    calendar = get_market_calendar()
    return calendar.is_market_open(dt)


def get_next_market_open() -> datetime:
    """
    Get next market open datetime.

    Returns:
        Next market open datetime
    """
    calendar = get_market_calendar()
    now = datetime.now(calendar.market_tz)

    if calendar.is_market_open(now):
        # Already open, get next day
        next_day = calendar.get_next_trading_day(now)
        return calendar.get_market_open(next_day)

    # Check today
    market_open = calendar.get_market_open(now)
    if market_open and now < market_open:
        return market_open

    # Get next trading day
    next_day = calendar.get_next_trading_day(now)
    return calendar.get_market_open(next_day)


def get_next_market_close() -> datetime:
    """
    Get next market close datetime.

    Returns:
        Next market close datetime
    """
    calendar = get_market_calendar()
    now = datetime.now(calendar.market_tz)

    if calendar.is_market_open(now):
        # Currently open, return today's close
        return calendar.get_market_close(now)

    # Get next trading day
    next_day = calendar.get_next_trading_day(now)
    return calendar.get_market_close(next_day)


def convert_timezone(
    dt: datetime,
    from_tz: str,
    to_tz: str
) -> datetime:
    """
    Convert datetime between timezones.

    Args:
        dt: Datetime to convert
        from_tz: Source timezone
        to_tz: Target timezone

    Returns:
        Converted datetime
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(from_tz))
    return dt.astimezone(ZoneInfo(to_tz))


def align_to_trading_day(
    dt: datetime,
    direction: str = "forward"
) -> datetime:
    """
    Align datetime to nearest trading day.

    Args:
        dt: Datetime to align
        direction: Direction to align ("forward" or "backward")

    Returns:
        Aligned datetime
    """
    calendar = get_market_calendar()

    if calendar.is_trading_day(dt):
        return dt

    if direction == "forward":
        return calendar.get_next_trading_day(dt)
    else:
        return calendar.get_previous_trading_day(dt)


def get_trading_sessions(
    start: datetime,
    end: datetime,
    session_type: str = "regular"
) -> List[Tuple[datetime, datetime]]:
    """
    Get list of trading sessions between dates.

    Args:
        start: Start date
        end: End date
        session_type: Session type ("regular" or "extended")

    Returns:
        List of (open, close) tuples
    """
    calendar = get_market_calendar()
    trading_days = calendar.get_trading_days(start, end)

    sessions = []
    for day in trading_days:
        market_open = calendar.get_market_open(day)
        market_close = calendar.get_market_close(day)

        if market_open and market_close:
            sessions.append((market_open, market_close))

    return sessions


def round_to_nearest_minute(dt: datetime) -> datetime:
    """
    Round datetime to nearest minute.

    Args:
        dt: Datetime to round

    Returns:
        Rounded datetime
    """
    return dt.replace(second=0, microsecond=0)


def get_bar_timestamps(
    start: datetime,
    end: datetime,
    interval: str = "1min"
) -> pd.DatetimeIndex:
    """
    Generate bar timestamps for given interval.

    Args:
        start: Start datetime
        end: End datetime
        interval: Bar interval (e.g., "1min", "5min", "1h")

    Returns:
        DatetimeIndex of bar timestamps
    """
    return pd.date_range(start=start, end=end, freq=interval)


def parse_time_string(time_str: str) -> time:
    """
    Parse time string to time object.

    Args:
        time_str: Time string (HH:MM or HH:MM:SS)

    Returns:
        time object
    """
    formats = ["%H:%M:%S", "%H:%M"]

    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt).time()
        except ValueError:
            continue

    raise ValueError(f"Invalid time string: {time_str}")


def format_duration(td: timedelta) -> str:
    """
    Format timedelta as human-readable string.

    Args:
        td: Timedelta to format

    Returns:
        Formatted string
    """
    total_seconds = int(td.total_seconds())

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def get_market_age(dt: datetime) -> str:
    """
    Get market age description (e.g., "early", "mid", "late").

    Args:
        dt: Datetime to check

    Returns:
        Market age description
    """
    calendar = get_market_calendar()

    if not calendar.is_market_open(dt):
        return "closed"

    market_open = calendar.get_market_open(dt)
    market_close = calendar.get_market_close(dt)

    total_duration = (market_close - market_open).total_seconds()
    elapsed = (dt - market_open).total_seconds()

    progress = elapsed / total_duration

    if progress < 0.25:
        return "early"
    elif progress < 0.75:
        return "mid"
    else:
        return "late"
