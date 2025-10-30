"""
Multi-agent trading system.

This module provides a sophisticated multi-agent trading system with specialized
agents for different trading strategies, coordinated through a central coordinator.
"""

from .base_agent import BaseAgent, AgentState, AgentMode, AgentMetrics
from .market_maker import MarketMakerAgent
from .trend_follower import TrendFollowerAgent
from .mean_reversion import MeanReversionAgent
from .arbitrage import ArbitrageAgent
from .momentum import MomentumAgent
from .coordinator import AgentCoordinator, CoordinatorMetrics

__all__ = [
    # Base
    "BaseAgent",
    "AgentState",
    "AgentMode",
    "AgentMetrics",

    # Agents
    "MarketMakerAgent",
    "TrendFollowerAgent",
    "MeanReversionAgent",
    "ArbitrageAgent",
    "MomentumAgent",

    # Coordinator
    "AgentCoordinator",
    "CoordinatorMetrics",
]
