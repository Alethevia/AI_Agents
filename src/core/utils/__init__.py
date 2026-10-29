"""
Core utility modules for trading system.
"""

from src.core.utils.date_utils import (
    MarketCalendar,
    get_market_calendar,
    is_trading_hours,
    get_next_market_open,
    get_next_market_close,
    convert_timezone,
    align_to_trading_day,
)

from src.core.utils.math_utils import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_volatility,
    calculate_var,
    calculate_cvar,
    calculate_beta,
    calculate_alpha,
)

from src.core.utils.validators import (
    ValidationError,
    OrderValidator,
    PositionValidator,
    PriceValidator,
    RiskValidator,
    DataQualityValidator,
)

from src.core.utils.serializers import (
    serialize_json,
    deserialize_json,
    serialize_pickle,
    deserialize_pickle,
    OrderSerializer,
    PositionSerializer,
    MarketDataSerializer,
)

from src.core.utils.cache_utils import (
    MemoryCache,
    RedisCache,
    CacheManager,
    get_cache,
    init_cache,
    cached,
)

__all__ = [
    # Date utils
    "MarketCalendar",
    "get_market_calendar",
    "is_trading_hours",
    "get_next_market_open",
    "get_next_market_close",
    "convert_timezone",
    "align_to_trading_day",
    # Math utils
    "calculate_returns",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_volatility",
    "calculate_var",
    "calculate_cvar",
    "calculate_beta",
    "calculate_alpha",
    # Validators
    "ValidationError",
    "OrderValidator",
    "PositionValidator",
    "PriceValidator",
    "RiskValidator",
    "DataQualityValidator",
    # Serializers
    "serialize_json",
    "deserialize_json",
    "serialize_pickle",
    "deserialize_pickle",
    "OrderSerializer",
    "PositionSerializer",
    "MarketDataSerializer",
    # Cache
    "MemoryCache",
    "RedisCache",
    "CacheManager",
    "get_cache",
    "init_cache",
    "cached",
]
