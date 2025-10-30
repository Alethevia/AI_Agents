"""
API module for the trading robot.

This module provides a comprehensive REST API and WebSocket server
for managing trading operations, portfolio monitoring, and real-time data.
"""

from .rest.main import app

__all__ = ["app"]
