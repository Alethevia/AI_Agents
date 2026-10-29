"""
Abstract base agent for multi-agent trading system.

Provides lifecycle management, state management, communication protocol,
and performance tracking for all trading agents.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.logging.logger import get_logger
from src.core.types.signals import Signal, SignalType, SignalStrength
from src.config.trading_config import TradingMode


class AgentState(str, Enum):
    """Agent lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class AgentMode(str, Enum):
    """Agent operating modes."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class AgentMetrics(BaseModel):
    """Agent performance metrics."""

    # Trade statistics
    total_signals: int = Field(default=0, description="Total signals generated")
    buy_signals: int = Field(default=0, description="Buy signals generated")
    sell_signals: int = Field(default=0, description="Sell signals generated")
    signals_accepted: int = Field(default=0, description="Signals accepted by coordinator")
    signals_rejected: int = Field(default=0, description="Signals rejected")

    # Performance
    total_pnl: Decimal = Field(default=Decimal("0"), description="Total P&L")
    win_rate: Decimal = Field(default=Decimal("0"), description="Win rate (0-1)")
    sharpe_ratio: Optional[Decimal] = Field(None, description="Sharpe ratio")
    max_drawdown: Decimal = Field(default=Decimal("0"), description="Maximum drawdown")

    # Timing
    avg_signal_time_ms: Decimal = Field(
        default=Decimal("0"),
        description="Average signal generation time"
    )
    update_count: int = Field(default=0, description="Number of updates")

    # Resource usage
    memory_usage_mb: Decimal = Field(default=Decimal("0"), description="Memory usage")
    cpu_time_seconds: Decimal = Field(default=Decimal("0"), description="CPU time used")

    # Errors
    error_count: int = Field(default=0, description="Number of errors")
    last_error: Optional[str] = Field(None, description="Last error message")
    last_error_time: Optional[datetime] = Field(None, description="Last error timestamp")

    def update_pnl(self, pnl: Decimal) -> None:
        """Update P&L and related metrics."""
        self.total_pnl += pnl

    def record_signal(self, signal_type: SignalType) -> None:
        """Record a signal."""
        self.total_signals += 1
        if signal_type in [SignalType.BUY, SignalType.INCREASE]:
            self.buy_signals += 1
        elif signal_type in [SignalType.SELL, SignalType.REDUCE]:
            self.sell_signals += 1

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.error_count += 1
        self.last_error = error
        self.last_error_time = datetime.utcnow()


class BaseAgent(ABC):
    """
    Abstract base agent for trading strategies.

    All trading agents must inherit from this class and implement the abstract methods.
    Provides common functionality for lifecycle management, state tracking, performance
    monitoring, and communication.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        mode: AgentMode = AgentMode.PAPER,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize base agent.

        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            mode: Operating mode (backtest, paper, live)
            config: Agent-specific configuration
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.mode = mode
        self.config = config or {}

        # State management
        self._state = AgentState.UNINITIALIZED
        self._state_history: List[tuple] = []
        self._lock = asyncio.Lock()

        # Performance tracking
        self.metrics = AgentMetrics()
        self._pnl_history: List[tuple] = []  # (timestamp, pnl)
        self._signal_history: List[Signal] = []

        # Communication
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._outgoing_signals: asyncio.Queue = asyncio.Queue()

        # Lifecycle
        self._initialized_at: Optional[datetime] = None
        self._started_at: Optional[datetime] = None
        self._stopped_at: Optional[datetime] = None

        # Logging
        self.logger = get_logger(f"{__name__}.{self.name}")

        # Agent-specific state
        self._agent_state: Dict[str, Any] = {}

    # ==================== Lifecycle Methods ====================

    async def initialize(self) -> None:
        """
        Initialize agent resources.

        Override this method to perform agent-specific initialization like
        loading models, connecting to data sources, etc.
        """
        async with self._lock:
            self._set_state(AgentState.INITIALIZING)

            try:
                self.logger.info(
                    "Initializing agent",
                    agent_id=self.agent_id,
                    mode=self.mode.value
                )

                # Call agent-specific initialization
                await self._initialize()

                self._initialized_at = datetime.utcnow()
                self._set_state(AgentState.READY)

                self.logger.info(
                    "Agent initialized successfully",
                    agent_id=self.agent_id
                )

            except Exception as e:
                self.logger.error(
                    "Failed to initialize agent",
                    agent_id=self.agent_id,
                    error=str(e),
                    exc_info=True
                )
                self._set_state(AgentState.ERROR)
                self.metrics.record_error(str(e))
                raise

    @abstractmethod
    async def _initialize(self) -> None:
        """Agent-specific initialization. Must be implemented by subclasses."""
        pass

    async def start(self) -> None:
        """Start agent operation."""
        async with self._lock:
            if self._state != AgentState.READY:
                raise RuntimeError(f"Cannot start agent in state {self._state}")

            self._set_state(AgentState.RUNNING)
            self._started_at = datetime.utcnow()

            self.logger.info("Agent started", agent_id=self.agent_id)

    async def pause(self) -> None:
        """Pause agent operation."""
        async with self._lock:
            if self._state != AgentState.RUNNING:
                raise RuntimeError(f"Cannot pause agent in state {self._state}")

            self._set_state(AgentState.PAUSED)
            self.logger.info("Agent paused", agent_id=self.agent_id)

    async def resume(self) -> None:
        """Resume agent operation."""
        async with self._lock:
            if self._state != AgentState.PAUSED:
                raise RuntimeError(f"Cannot resume agent in state {self._state}")

            self._set_state(AgentState.RUNNING)
            self.logger.info("Agent resumed", agent_id=self.agent_id)

    async def stop(self) -> None:
        """Stop agent operation."""
        async with self._lock:
            if self._state == AgentState.STOPPED:
                return

            self.logger.info("Stopping agent", agent_id=self.agent_id)

            try:
                await self._cleanup()
                self._stopped_at = datetime.utcnow()
                self._set_state(AgentState.STOPPED)

                self.logger.info("Agent stopped", agent_id=self.agent_id)

            except Exception as e:
                self.logger.error(
                    "Error during agent cleanup",
                    agent_id=self.agent_id,
                    error=str(e),
                    exc_info=True
                )
                self._set_state(AgentState.ERROR)
                raise

    async def cleanup(self) -> None:
        """Cleanup agent resources. Alias for stop()."""
        await self.stop()

    @abstractmethod
    async def _cleanup(self) -> None:
        """Agent-specific cleanup. Must be implemented by subclasses."""
        pass

    # ==================== Update and Decision Methods ====================

    async def update(self, market_data: Dict[str, Any]) -> None:
        """
        Update agent with new market data.

        Args:
            market_data: Market data dictionary
        """
        if self._state != AgentState.RUNNING:
            return

        try:
            start_time = datetime.utcnow()

            # Call agent-specific update
            await self._update(market_data)

            # Track timing
            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics.update_count += 1

            # Update average timing
            n = self.metrics.update_count
            current_avg = float(self.metrics.avg_signal_time_ms)
            self.metrics.avg_signal_time_ms = Decimal(
                str((current_avg * (n - 1) + elapsed) / n)
            )

        except Exception as e:
            self.logger.error(
                "Error during agent update",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True
            )
            self.metrics.record_error(str(e))

    @abstractmethod
    async def _update(self, market_data: Dict[str, Any]) -> None:
        """Agent-specific update logic. Must be implemented by subclasses."""
        pass

    async def decide(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Generate trading decision for a symbol.

        Args:
            symbol: Trading symbol
            market_data: Market data for the symbol

        Returns:
            Trading signal or None
        """
        if self._state != AgentState.RUNNING:
            return None

        try:
            start_time = datetime.utcnow()

            # Call agent-specific decision logic
            signal = await self._decide(symbol, market_data)

            if signal:
                # Record signal
                self.metrics.record_signal(signal.signal_type)
                self._signal_history.append(signal)

                # Queue for coordinator
                await self._outgoing_signals.put(signal)

                self.logger.info(
                    "Signal generated",
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=signal.signal_type.value,
                    confidence=float(signal.confidence)
                )

            return signal

        except Exception as e:
            self.logger.error(
                "Error during decision making",
                agent_id=self.agent_id,
                symbol=symbol,
                error=str(e),
                exc_info=True
            )
            self.metrics.record_error(str(e))
            return None

    @abstractmethod
    async def _decide(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """Agent-specific decision logic. Must be implemented by subclasses."""
        pass

    # ==================== Communication Methods ====================

    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send message to other agents via coordinator."""
        message['from_agent'] = self.agent_id
        message['timestamp'] = datetime.utcnow()
        await self._message_queue.put(message)

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from coordinator."""
        try:
            return await asyncio.wait_for(
                self._message_queue.get(),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            return None

    async def get_next_signal(self) -> Optional[Signal]:
        """Get next outgoing signal."""
        try:
            return await asyncio.wait_for(
                self._outgoing_signals.get(),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            return None

    def broadcast_state(self) -> Dict[str, Any]:
        """Broadcast agent state to coordinator."""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'state': self._state.value,
            'mode': self.mode.value,
            'metrics': self.metrics.model_dump(),
            'timestamp': datetime.utcnow()
        }

    # ==================== State Management ====================

    def _set_state(self, state: AgentState) -> None:
        """Set agent state and record history."""
        old_state = self._state
        self._state = state
        self._state_history.append((datetime.utcnow(), old_state, state))

        self.logger.debug(
            "Agent state changed",
            agent_id=self.agent_id,
            old_state=old_state.value,
            new_state=state.value
        )

    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._state == AgentState.RUNNING

    @property
    def is_ready(self) -> bool:
        """Check if agent is ready."""
        return self._state in [AgentState.READY, AgentState.RUNNING]

    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Get agent-specific state value."""
        return self._agent_state.get(key, default)

    def set_state_value(self, key: str, value: Any) -> None:
        """Set agent-specific state value."""
        self._agent_state[key] = value

    # ==================== Performance Tracking ====================

    def record_pnl(self, pnl: Decimal, timestamp: Optional[datetime] = None) -> None:
        """Record P&L."""
        timestamp = timestamp or datetime.utcnow()
        self.metrics.update_pnl(pnl)
        self._pnl_history.append((timestamp, pnl))

    def calculate_sharpe_ratio(self, risk_free_rate: Decimal = Decimal("0.02")) -> Decimal:
        """Calculate Sharpe ratio from P&L history."""
        if len(self._pnl_history) < 2:
            return Decimal("0")

        returns = [float(pnl) for _, pnl in self._pnl_history]

        import numpy as np
        returns_array = np.array(returns)

        if returns_array.std() == 0:
            return Decimal("0")

        excess_return = returns_array.mean() - float(risk_free_rate) / 252
        sharpe = excess_return / returns_array.std() * np.sqrt(252)

        return Decimal(str(sharpe))

    def calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown."""
        if not self._pnl_history:
            return Decimal("0")

        cumulative_pnl = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")

        for _, pnl in self._pnl_history:
            cumulative_pnl += pnl
            peak = max(peak, cumulative_pnl)
            drawdown = peak - cumulative_pnl
            max_dd = max(max_dd, drawdown)

        return max_dd

    def update_performance_metrics(self) -> None:
        """Update performance metrics."""
        self.metrics.sharpe_ratio = self.calculate_sharpe_ratio()
        self.metrics.max_drawdown = self.calculate_max_drawdown()

        # Calculate win rate
        if self._signal_history:
            # This is simplified - in practice, you'd track actual trade outcomes
            total = len(self._signal_history)
            if total > 0:
                wins = sum(1 for s in self._signal_history if s.confidence > Decimal("0.6"))
                self.metrics.win_rate = Decimal(str(wins / total))

    # ==================== Utility Methods ====================

    def get_runtime(self) -> Optional[timedelta]:
        """Get agent runtime."""
        if not self._started_at:
            return None

        end_time = self._stopped_at or datetime.utcnow()
        return end_time - self._started_at

    def get_signal_history(
        self,
        limit: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[Signal]:
        """Get signal history."""
        signals = self._signal_history

        if since:
            signals = [s for s in signals if s.timestamp >= since]

        if limit:
            signals = signals[-limit:]

        return signals

    def get_status(self) -> Dict[str, Any]:
        """Get agent status summary."""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'state': self._state.value,
            'mode': self.mode.value,
            'initialized_at': self._initialized_at.isoformat() if self._initialized_at else None,
            'started_at': self._started_at.isoformat() if self._started_at else None,
            'runtime': str(self.get_runtime()) if self.get_runtime() else None,
            'metrics': self.metrics.model_dump(),
            'config': self.config,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(id={self.agent_id}, "
            f"state={self._state.value}, mode={self.mode.value})"
        )
