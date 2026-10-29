"""
Global settings and configuration management.
Uses pydantic-settings for type-safe configuration from environment variables.
"""

from functools import lru_cache
from typing import Literal, Optional
from pathlib import Path

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Runtime environment"
    )
    debug: bool = Field(default=False, description="Debug mode")

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    alpha_vantage_api_key: Optional[str] = Field(default=None, description="Alpha Vantage API key")

    # Database - PostgreSQL
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(default="trading_user", description="PostgreSQL user")
    postgres_password: str = Field(default="", description="PostgreSQL password")
    postgres_db: str = Field(default="trading_db", description="PostgreSQL database")

    # Database - MongoDB
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )
    mongodb_db: str = Field(default="trading_db", description="MongoDB database")

    # Redis
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")

    # Vector Database - Qdrant
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")

    # Trading Brokers - Alpaca
    alpaca_api_key: Optional[str] = Field(default=None, description="Alpaca API key")
    alpaca_secret_key: Optional[str] = Field(default=None, description="Alpaca secret key")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL"
    )

    # Trading Brokers - Interactive Brokers
    interactive_brokers_host: str = Field(default="127.0.0.1", description="IB Gateway host")
    interactive_brokers_port: int = Field(default=7497, description="IB Gateway port")
    interactive_brokers_client_id: int = Field(default=1, description="IB client ID")

    # Monitoring
    prometheus_port: int = Field(default=9090, description="Prometheus metrics port")
    grafana_port: int = Field(default=3000, description="Grafana dashboard port")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: Literal["json", "text"] = Field(default="json", description="Log format")
    log_file_path: Path = Field(
        default=Path("/var/log/trading_system"),
        description="Log file directory"
    )

    # Trading Configuration
    max_position_size: float = Field(
        default=100000.0,
        description="Maximum position size in USD"
    )
    max_portfolio_risk: float = Field(
        default=0.02,
        description="Maximum portfolio risk per trade",
        ge=0.0,
        le=1.0
    )
    max_drawdown: float = Field(
        default=0.15,
        description="Maximum allowed drawdown",
        ge=0.0,
        le=1.0
    )
    trading_hours_start: str = Field(
        default="09:30",
        description="Trading hours start (HH:MM)"
    )
    trading_hours_end: str = Field(
        default="16:00",
        description="Trading hours end (HH:MM)"
    )

    # Model Configuration
    model_update_interval: int = Field(
        default=3600,
        description="Model update interval in seconds"
    )
    feature_update_interval: int = Field(
        default=60,
        description="Feature update interval in seconds"
    )
    prediction_horizon: int = Field(
        default=60,
        description="Prediction horizon in minutes"
    )

    # Risk Management
    stop_loss_pct: float = Field(
        default=0.02,
        description="Default stop loss percentage",
        ge=0.0,
        le=1.0
    )
    take_profit_pct: float = Field(
        default=0.05,
        description="Default take profit percentage",
        ge=0.0
    )
    max_leverage: float = Field(
        default=2.0,
        description="Maximum allowed leverage",
        ge=1.0
    )
    risk_free_rate: float = Field(
        default=0.045,
        description="Risk-free rate for calculations"
    )

    # Performance
    num_workers: int = Field(
        default=8,
        description="Number of worker processes",
        ge=1
    )
    batch_size: int = Field(
        default=256,
        description="Batch size for model training",
        ge=1
    )
    prefetch_factor: int = Field(
        default=2,
        description="DataLoader prefetch factor",
        ge=1
    )

    # Paths
    data_dir: Path = Field(
        default=Path("data"),
        description="Data directory path"
    )
    models_dir: Path = Field(
        default=Path("models"),
        description="Models directory path"
    )
    logs_dir: Path = Field(
        default=Path("logs"),
        description="Logs directory path"
    )
    cache_dir: Path = Field(
        default=Path("cache"),
        description="Cache directory path"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_sync_url(self) -> str:
        """Get PostgreSQL synchronous connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}@{self.redis_host}:"
                f"{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.env == "development"

    def model_post_init(self, __context) -> None:
        """Create required directories after initialization."""
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Global settings instance
    """
    return Settings()
