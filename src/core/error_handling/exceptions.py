"""
Custom exception classes for the trading system.
"""

from typing import Optional, Dict, Any


class TradingSystemError(Exception):
    """Base exception for trading system errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize trading system error.

        Args:
            message: Error message
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ConfigurationError(TradingSystemError):
    """Raised when there's a configuration error."""
    pass


class ValidationError(TradingSystemError):
    """Raised when validation fails."""
    pass


class DataError(TradingSystemError):
    """Base class for data-related errors."""
    pass


class DataFetchError(DataError):
    """Raised when data fetching fails."""
    pass


class DataProcessingError(DataError):
    """Raised when data processing fails."""
    pass


class DataQualityError(DataError):
    """Raised when data quality issues are detected."""
    pass


class ModelError(TradingSystemError):
    """Base class for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    pass


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class ExecutionError(TradingSystemError):
    """Base class for execution-related errors."""
    pass


class OrderError(ExecutionError):
    """Raised when order execution fails."""
    pass


class BrokerError(ExecutionError):
    """Raised when broker API calls fail."""
    pass


class SlippageError(ExecutionError):
    """Raised when slippage exceeds acceptable limits."""
    pass


class RiskError(TradingSystemError):
    """Base class for risk-related errors."""
    pass


class RiskLimitError(RiskError):
    """Raised when risk limits are breached."""
    pass


class PositionSizeError(RiskError):
    """Raised when position size limits are breached."""
    pass


class DrawdownError(RiskError):
    """Raised when drawdown limits are breached."""
    pass


class MarginError(RiskError):
    """Raised when margin requirements are not met."""
    pass


class PortfolioError(TradingSystemError):
    """Base class for portfolio-related errors."""
    pass


class InsufficientFundsError(PortfolioError):
    """Raised when insufficient funds for trade."""
    pass


class PositionNotFoundError(PortfolioError):
    """Raised when position is not found."""
    pass


class AgentError(TradingSystemError):
    """Base class for agent-related errors."""
    pass


class AgentCommunicationError(AgentError):
    """Raised when agent communication fails."""
    pass


class AgentDecisionError(AgentError):
    """Raised when agent decision making fails."""
    pass


class BacktestError(TradingSystemError):
    """Base class for backtesting errors."""
    pass


class BacktestDataError(BacktestError):
    """Raised when backtest data is invalid."""
    pass


class APIError(TradingSystemError):
    """Base class for API errors."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass


class TimeoutError(TradingSystemError):
    """Raised when an operation times out."""
    pass


class CircuitBreakerError(TradingSystemError):
    """Raised when circuit breaker is triggered."""
    pass
