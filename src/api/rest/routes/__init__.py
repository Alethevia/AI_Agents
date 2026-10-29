"""
API routes package.

This package contains all REST API route definitions organized by resource type.
"""

from . import (
    auth_routes,
    portfolio_routes,
    order_routes,
    strategy_routes,
    market_data_routes,
    risk_routes,
)

__all__ = [
    "auth_routes",
    "portfolio_routes",
    "order_routes",
    "strategy_routes",
    "market_data_routes",
    "risk_routes",
]
