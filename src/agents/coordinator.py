"""
Agent coordinator for multi-agent system management.

Coordinates multiple trading agents, aggregates signals, resolves conflicts,
manages portfolio allocation, monitors performance, and dynamically weights agents.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentState, AgentMode, AgentMetrics
from src.core.types.signals import Signal, SignalType, SignalStrength, SignalCollection
from src.core.logging.logger import get_logger


class CoordinatorMetrics(BaseModel):
    """Coordinator performance metrics."""

    # Signal statistics
    total_signals_received: int = Field(default=0, description="Total signals received")
    total_signals_accepted: int = Field(default=0, description="Signals accepted")
    total_signals_rejected: int = Field(default=0, description="Signals rejected")
    conflicts_resolved: int = Field(default=0, description="Conflicts resolved")

    # Agent statistics
    active_agents: int = Field(default=0, description="Number of active agents")
    total_agents: int = Field(default=0, description="Total agents managed")

    # Portfolio statistics
    total_positions: int = Field(default=0, description="Total active positions")
    portfolio_value: Decimal = Field(default=Decimal("0"), description="Portfolio value")
    cash_balance: Decimal = Field(default=Decimal("0"), description="Cash balance")

    # Performance
    total_pnl: Decimal = Field(default=Decimal("0"), description="Total P&L")
    sharpe_ratio: Optional[Decimal] = Field(None, description="Sharpe ratio")
    win_rate: Decimal = Field(default=Decimal("0"), description="Win rate")

    # Update timing
    last_update: Optional[datetime] = Field(None, description="Last update time")
    update_count: int = Field(default=0, description="Number of updates")


class AgentCoordinator:
    """
    Multi-agent coordinator.

    Manages multiple trading agents, aggregates their signals, resolves conflicts,
    allocates portfolio across agents, and monitors performance.
    """

    def __init__(
        self,
        coordinator_id: Optional[str] = None,
        initial_capital: Decimal = Decimal("100000"),
        max_positions: int = 50,
        max_agents: int = 20,
        conflict_resolution_method: str = "weighted_vote",
        enable_dynamic_weighting: bool = True,
    ):
        """
        Initialize agent coordinator.

        Args:
            coordinator_id: Unique coordinator identifier
            initial_capital: Initial capital for portfolio
            max_positions: Maximum number of positions
            max_agents: Maximum number of agents
            conflict_resolution_method: Method for conflict resolution
            enable_dynamic_weighting: Enable dynamic agent weighting
        """
        self.coordinator_id = coordinator_id or str(uuid.uuid4())
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.max_agents = max_agents
        self.conflict_resolution_method = conflict_resolution_method
        self.enable_dynamic_weighting = enable_dynamic_weighting

        # Agent management
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_weights: Dict[str, Decimal] = {}
        self._agent_allocations: Dict[str, Decimal] = {}  # Capital allocation per agent

        # Signal management
        self._pending_signals: asyncio.Queue = asyncio.Queue()
        self._accepted_signals: List[Signal] = []
        self._rejected_signals: List[Signal] = []
        self._signal_history: List[Signal] = []

        # Portfolio management
        self._positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        self._cash_balance: Decimal = initial_capital
        self._portfolio_value: Decimal = initial_capital

        # Performance tracking
        self.metrics = CoordinatorMetrics()
        self._pnl_history: List[Tuple[datetime, Decimal]] = []
        self._portfolio_history: List[Tuple[datetime, Decimal]] = []

        # Communication
        self._message_bus: asyncio.Queue = asyncio.Queue()

        # State
        self._running = False
        self._lock = asyncio.Lock()

        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    # ==================== Agent Management ====================

    async def register_agent(
        self,
        agent: BaseAgent,
        initial_weight: Decimal = Decimal("1.0"),
        capital_allocation: Optional[Decimal] = None
    ) -> bool:
        """
        Register an agent with the coordinator.

        Args:
            agent: Agent to register
            initial_weight: Initial weight for the agent
            capital_allocation: Initial capital allocation

        Returns:
            True if registered successfully
        """
        async with self._lock:
            if len(self._agents) >= self.max_agents:
                self.logger.warning(
                    "Cannot register agent, max agents reached",
                    max_agents=self.max_agents
                )
                return False

            if agent.agent_id in self._agents:
                self.logger.warning(
                    "Agent already registered",
                    agent_id=agent.agent_id
                )
                return False

            # Register agent
            self._agents[agent.agent_id] = agent
            self._agent_weights[agent.agent_id] = initial_weight

            # Allocate capital
            if capital_allocation is None:
                # Equal allocation by default
                n_agents = len(self._agents)
                capital_allocation = self.initial_capital / Decimal(str(n_agents))

            self._agent_allocations[agent.agent_id] = capital_allocation

            # Update metrics
            self.metrics.total_agents = len(self._agents)

            self.logger.info(
                "Agent registered",
                agent_id=agent.agent_id,
                agent_name=agent.name,
                weight=float(initial_weight),
                allocation=float(capital_allocation)
            )

            return True

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: Agent ID to unregister

        Returns:
            True if unregistered successfully
        """
        async with self._lock:
            if agent_id not in self._agents:
                return False

            # Stop agent if running
            agent = self._agents[agent_id]
            if agent.is_running:
                await agent.stop()

            # Remove agent
            del self._agents[agent_id]
            del self._agent_weights[agent_id]
            del self._agent_allocations[agent_id]

            # Update metrics
            self.metrics.total_agents = len(self._agents)

            self.logger.info("Agent unregistered", agent_id=agent_id)

            return True

    async def start_all_agents(self) -> None:
        """Start all registered agents."""
        for agent in self._agents.values():
            try:
                if agent.state == AgentState.UNINITIALIZED:
                    await agent.initialize()
                if agent.state == AgentState.READY:
                    await agent.start()
            except Exception as e:
                self.logger.error(
                    "Failed to start agent",
                    agent_id=agent.agent_id,
                    error=str(e),
                    exc_info=True
                )

        # Update metrics
        self.metrics.active_agents = sum(1 for a in self._agents.values() if a.is_running)

    async def stop_all_agents(self) -> None:
        """Stop all registered agents."""
        for agent in self._agents.values():
            try:
                if agent.is_running or agent.state == AgentState.PAUSED:
                    await agent.stop()
            except Exception as e:
                self.logger.error(
                    "Failed to stop agent",
                    agent_id=agent.agent_id,
                    error=str(e),
                    exc_info=True
                )

        self.metrics.active_agents = 0

    # ==================== Signal Processing ====================

    async def start(self) -> None:
        """Start coordinator."""
        self._running = True
        self.logger.info("Coordinator started", coordinator_id=self.coordinator_id)

        # Start signal processing loop
        asyncio.create_task(self._signal_processing_loop())

    async def stop(self) -> None:
        """Stop coordinator."""
        self._running = False
        await self.stop_all_agents()
        self.logger.info("Coordinator stopped")

    async def _signal_processing_loop(self) -> None:
        """Main signal processing loop."""
        while self._running:
            try:
                # Collect signals from all agents
                await self._collect_agent_signals()

                # Process pending signals
                await self._process_signals()

                # Update agent weights if enabled
                if self.enable_dynamic_weighting:
                    await self._update_agent_weights()

                # Update metrics
                await self._update_metrics()

                # Small delay to prevent busy loop
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(
                    "Error in signal processing loop",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(1)

    async def _collect_agent_signals(self) -> None:
        """Collect signals from all agents."""
        for agent in self._agents.values():
            if not agent.is_running:
                continue

            try:
                signal = await agent.get_next_signal()
                if signal:
                    await self._pending_signals.put((agent.agent_id, signal))
                    self.metrics.total_signals_received += 1

            except Exception as e:
                self.logger.error(
                    "Error collecting signal from agent",
                    agent_id=agent.agent_id,
                    error=str(e),
                    exc_info=True
                )

    async def _process_signals(self) -> None:
        """Process pending signals."""
        # Collect all pending signals
        signals_batch: List[Tuple[str, Signal]] = []

        while not self._pending_signals.empty():
            try:
                agent_signal = await asyncio.wait_for(
                    self._pending_signals.get(),
                    timeout=0.01
                )
                signals_batch.append(agent_signal)
            except asyncio.TimeoutError:
                break

        if not signals_batch:
            return

        # Group signals by symbol
        signals_by_symbol: Dict[str, List[Tuple[str, Signal]]] = {}
        for agent_id, signal in signals_batch:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append((agent_id, signal))

        # Process each symbol's signals
        for symbol, agent_signals in signals_by_symbol.items():
            await self._process_symbol_signals(symbol, agent_signals)

    async def _process_symbol_signals(
        self,
        symbol: str,
        agent_signals: List[Tuple[str, Signal]]
    ) -> None:
        """
        Process signals for a specific symbol.

        Args:
            symbol: Trading symbol
            agent_signals: List of (agent_id, signal) tuples
        """
        # Check for conflicts
        signal_types = [s.signal_type for _, s in agent_signals]

        # If all agree, no conflict
        if len(set(signal_types)) == 1:
            # Aggregate agreeing signals
            final_signal = await self._aggregate_signals(symbol, agent_signals)
            if final_signal:
                await self._accept_signal(final_signal, agent_signals)
            return

        # Conflict detected
        self.metrics.conflicts_resolved += 1
        self.logger.info(
            "Signal conflict detected",
            symbol=symbol,
            num_signals=len(agent_signals),
            signal_types=[st.value for st in signal_types]
        )

        # Resolve conflict
        final_signal = await self._resolve_conflict(symbol, agent_signals)

        if final_signal:
            await self._accept_signal(final_signal, agent_signals)
        else:
            # Reject all conflicting signals
            for agent_id, signal in agent_signals:
                await self._reject_signal(signal, "conflict_unresolved")

    async def _aggregate_signals(
        self,
        symbol: str,
        agent_signals: List[Tuple[str, Signal]]
    ) -> Optional[Signal]:
        """
        Aggregate multiple agreeing signals into one.

        Args:
            symbol: Trading symbol
            agent_signals: List of (agent_id, signal) tuples

        Returns:
            Aggregated signal or None
        """
        if not agent_signals:
            return None

        # Calculate weighted average confidence
        total_weight = Decimal("0")
        weighted_confidence = Decimal("0")
        weighted_position_size = Decimal("0")

        signal_type = agent_signals[0][1].signal_type
        strengths = []

        for agent_id, signal in agent_signals:
            weight = self._agent_weights.get(agent_id, Decimal("1.0"))
            total_weight += weight

            weighted_confidence += signal.confidence * weight

            if signal.position_size:
                weighted_position_size += signal.position_size * weight

            strengths.append(signal.strength)

        if total_weight == 0:
            return None

        # Calculate aggregated values
        avg_confidence = weighted_confidence / total_weight
        avg_position_size = weighted_position_size / total_weight if weighted_position_size > 0 else None

        # Determine aggregated strength
        strength_counts = {}
        for strength in strengths:
            strength_counts[strength] = strength_counts.get(strength, 0) + 1

        # Most common strength
        aggregated_strength = max(strength_counts.items(), key=lambda x: x[1])[0]

        # Use first signal as template
        template_signal = agent_signals[0][1]

        # Create aggregated signal
        aggregated_signal = Signal(
            signal_id=f"coord_{self.coordinator_id}_{symbol}_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            signal_type=signal_type,
            strength=aggregated_strength,
            confidence=avg_confidence,
            source="coordinator",
            strategy_id=self.coordinator_id,
            market_price=template_signal.market_price,
            stop_loss=template_signal.stop_loss,
            take_profit=template_signal.take_profit,
            position_size=avg_position_size,
            indicators={
                'num_agents': Decimal(str(len(agent_signals))),
                'avg_confidence': avg_confidence
            },
            metadata={
                'aggregated': True,
                'num_agents': len(agent_signals),
                'contributing_agents': [agent_id for agent_id, _ in agent_signals],
                'original_signals': [s.signal_id for _, s in agent_signals]
            }
        )

        return aggregated_signal

    async def _resolve_conflict(
        self,
        symbol: str,
        agent_signals: List[Tuple[str, Signal]]
    ) -> Optional[Signal]:
        """
        Resolve conflicting signals.

        Args:
            symbol: Trading symbol
            agent_signals: List of (agent_id, signal) tuples

        Returns:
            Resolved signal or None
        """
        if self.conflict_resolution_method == "weighted_vote":
            return await self._resolve_by_weighted_vote(symbol, agent_signals)
        elif self.conflict_resolution_method == "highest_confidence":
            return await self._resolve_by_highest_confidence(symbol, agent_signals)
        elif self.conflict_resolution_method == "best_performer":
            return await self._resolve_by_best_performer(symbol, agent_signals)
        else:
            return await self._resolve_by_weighted_vote(symbol, agent_signals)

    async def _resolve_by_weighted_vote(
        self,
        symbol: str,
        agent_signals: List[Tuple[str, Signal]]
    ) -> Optional[Signal]:
        """Resolve by weighted vote."""
        # Calculate weighted votes
        buy_weight = Decimal("0")
        sell_weight = Decimal("0")
        hold_weight = Decimal("0")

        buy_signals = []
        sell_signals = []

        for agent_id, signal in agent_signals:
            weight = self._agent_weights.get(agent_id, Decimal("1.0"))
            weighted_score = weight * signal.confidence

            if signal.signal_type in [SignalType.BUY, SignalType.INCREASE]:
                buy_weight += weighted_score
                buy_signals.append((agent_id, signal))
            elif signal.signal_type in [SignalType.SELL, SignalType.REDUCE]:
                sell_weight += weighted_score
                sell_signals.append((agent_id, signal))
            else:
                hold_weight += weighted_score

        # Determine winner
        if buy_weight > sell_weight and buy_weight > hold_weight:
            return await self._aggregate_signals(symbol, buy_signals)
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            return await self._aggregate_signals(symbol, sell_signals)
        else:
            # No clear winner
            return None

    async def _resolve_by_highest_confidence(
        self,
        symbol: str,
        agent_signals: List[Tuple[str, Signal]]
    ) -> Optional[Signal]:
        """Resolve by highest confidence signal."""
        # Find signal with highest confidence
        best_signal = max(agent_signals, key=lambda x: x[1].confidence)
        return best_signal[1]

    async def _resolve_by_best_performer(
        self,
        symbol: str,
        agent_signals: List[Tuple[str, Signal]]
    ) -> Optional[Signal]:
        """Resolve by signal from best performing agent."""
        # Find best performing agent
        best_agent_id = None
        best_performance = Decimal("-inf")

        for agent_id, _ in agent_signals:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                if agent.metrics.total_pnl > best_performance:
                    best_performance = agent.metrics.total_pnl
                    best_agent_id = agent_id

        if best_agent_id:
            # Return signal from best performer
            for agent_id, signal in agent_signals:
                if agent_id == best_agent_id:
                    return signal

        return None

    async def _accept_signal(
        self,
        signal: Signal,
        contributing_signals: List[Tuple[str, Signal]]
    ) -> None:
        """
        Accept a signal for execution.

        Args:
            signal: Signal to accept
            contributing_signals: Signals that contributed to this decision
        """
        # Validate portfolio constraints
        if not await self._validate_portfolio_constraints(signal):
            await self._reject_signal(signal, "portfolio_constraints")
            return

        # Accept signal
        self._accepted_signals.append(signal)
        self._signal_history.append(signal)
        self.metrics.total_signals_accepted += 1

        # Notify contributing agents
        for agent_id, original_signal in contributing_signals:
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                agent.metrics.signals_accepted += 1

        self.logger.info(
            "Signal accepted",
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            signal_type=signal.signal_type.value,
            confidence=float(signal.confidence),
            num_contributing_agents=len(contributing_signals)
        )

    async def _reject_signal(self, signal: Signal, reason: str) -> None:
        """
        Reject a signal.

        Args:
            signal: Signal to reject
            reason: Rejection reason
        """
        self._rejected_signals.append(signal)
        self.metrics.total_signals_rejected += 1

        self.logger.debug(
            "Signal rejected",
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            reason=reason
        )

    async def _validate_portfolio_constraints(self, signal: Signal) -> bool:
        """Validate portfolio constraints."""
        # Check max positions
        if len(self._positions) >= self.max_positions:
            # Only allow if closing existing position
            if signal.symbol not in self._positions:
                return False

        # Check cash balance
        if signal.position_size and signal.market_price:
            required_cash = signal.position_size * signal.market_price
            if required_cash > self._cash_balance:
                return False

        return True

    # ==================== Agent Weight Management ====================

    async def _update_agent_weights(self) -> None:
        """Update agent weights based on performance."""
        if not self.enable_dynamic_weighting:
            return

        # Calculate performance scores
        performance_scores: Dict[str, Decimal] = {}

        for agent_id, agent in self._agents.items():
            score = await self._calculate_agent_score(agent)
            performance_scores[agent_id] = score

        # Normalize scores to weights
        total_score = sum(performance_scores.values())

        if total_score > 0:
            for agent_id, score in performance_scores.items():
                # Weighted average: 70% performance, 30% equal weight
                performance_weight = score / total_score
                equal_weight = Decimal("1") / Decimal(str(len(self._agents)))

                new_weight = (
                    performance_weight * Decimal("0.7") +
                    equal_weight * Decimal("0.3")
                )

                self._agent_weights[agent_id] = new_weight

            self.logger.debug(
                "Agent weights updated",
                weights={aid: float(w) for aid, w in self._agent_weights.items()}
            )

    async def _calculate_agent_score(self, agent: BaseAgent) -> Decimal:
        """Calculate performance score for an agent."""
        score = Decimal("1.0")  # Base score

        # P&L contribution
        if agent.metrics.total_pnl > 0:
            score += agent.metrics.total_pnl / Decimal("1000")  # Scale by $1000

        # Win rate
        score += agent.metrics.win_rate * Decimal("2")

        # Sharpe ratio
        if agent.metrics.sharpe_ratio and agent.metrics.sharpe_ratio > 0:
            score += agent.metrics.sharpe_ratio * Decimal("0.5")

        # Signal acceptance rate
        if agent.metrics.total_signals > 0:
            acceptance_rate = Decimal(str(agent.metrics.signals_accepted)) / Decimal(str(agent.metrics.total_signals))
            score += acceptance_rate

        # Penalize errors
        if agent.metrics.error_count > 0:
            score *= Decimal("0.9") ** Decimal(str(agent.metrics.error_count))

        return max(score, Decimal("0.1"))  # Minimum score

    # ==================== Metrics and Monitoring ====================

    async def _update_metrics(self) -> None:
        """Update coordinator metrics."""
        self.metrics.active_agents = sum(1 for a in self._agents.values() if a.is_running)
        self.metrics.total_agents = len(self._agents)
        self.metrics.total_positions = len(self._positions)
        self.metrics.cash_balance = self._cash_balance
        self.metrics.portfolio_value = self._portfolio_value
        self.metrics.last_update = datetime.utcnow()
        self.metrics.update_count += 1

        # Calculate total P&L from agents
        total_pnl = sum(a.metrics.total_pnl for a in self._agents.values())
        self.metrics.total_pnl = total_pnl

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            'coordinator_id': self.coordinator_id,
            'running': self._running,
            'metrics': self.metrics.model_dump(),
            'agents': {
                agent_id: {
                    'name': agent.name,
                    'state': agent.state.value,
                    'weight': float(self._agent_weights.get(agent_id, Decimal("0"))),
                    'allocation': float(self._agent_allocations.get(agent_id, Decimal("0"))),
                    'metrics': agent.metrics.model_dump()
                }
                for agent_id, agent in self._agents.items()
            },
            'portfolio': {
                'cash_balance': float(self._cash_balance),
                'portfolio_value': float(self._portfolio_value),
                'num_positions': len(self._positions),
                'positions': list(self._positions.keys())
            }
        }

    def get_signal_summary(self, limit: int = 100) -> Dict[str, Any]:
        """Get signal summary."""
        recent_signals = self._signal_history[-limit:]

        return {
            'total_signals': len(self._signal_history),
            'accepted': self.metrics.total_signals_accepted,
            'rejected': self.metrics.total_signals_rejected,
            'conflicts_resolved': self.metrics.conflicts_resolved,
            'recent_signals': [
                {
                    'signal_id': s.signal_id,
                    'symbol': s.symbol,
                    'signal_type': s.signal_type.value,
                    'strength': s.strength.value,
                    'confidence': float(s.confidence),
                    'timestamp': s.timestamp.isoformat()
                }
                for s in recent_signals
            ]
        }

    def get_agent_performance(self) -> Dict[str, Any]:
        """Get agent performance summary."""
        return {
            'agents': {
                agent.name: {
                    'agent_id': agent.agent_id,
                    'weight': float(self._agent_weights.get(agent.agent_id, Decimal("0"))),
                    'total_signals': agent.metrics.total_signals,
                    'accepted_signals': agent.metrics.signals_accepted,
                    'acceptance_rate': (
                        float(agent.metrics.signals_accepted / agent.metrics.total_signals)
                        if agent.metrics.total_signals > 0 else 0.0
                    ),
                    'total_pnl': float(agent.metrics.total_pnl),
                    'win_rate': float(agent.metrics.win_rate),
                    'sharpe_ratio': float(agent.metrics.sharpe_ratio) if agent.metrics.sharpe_ratio else None,
                    'error_count': agent.metrics.error_count
                }
                for agent in self._agents.values()
            },
            'best_performer': self._get_best_performer(),
            'total_pnl': float(self.metrics.total_pnl)
        }

    def _get_best_performer(self) -> Optional[str]:
        """Get best performing agent."""
        if not self._agents:
            return None

        best_agent = max(
            self._agents.values(),
            key=lambda a: a.metrics.total_pnl
        )

        return best_agent.name

    async def broadcast_market_data(self, market_data: Dict[str, Any]) -> None:
        """
        Broadcast market data to all agents.

        Args:
            market_data: Market data to broadcast
        """
        update_tasks = []

        for agent in self._agents.values():
            if agent.is_running:
                update_tasks.append(agent.update(market_data))

        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)

    async def get_decisions_for_symbol(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> List[Signal]:
        """
        Get trading decisions from all agents for a symbol.

        Args:
            symbol: Trading symbol
            market_data: Market data for the symbol

        Returns:
            List of signals from agents
        """
        decision_tasks = []

        for agent in self._agents.values():
            if agent.is_running:
                decision_tasks.append(agent.decide(symbol, market_data))

        if not decision_tasks:
            return []

        results = await asyncio.gather(*decision_tasks, return_exceptions=True)

        # Filter out None and exceptions
        signals = [r for r in results if isinstance(r, Signal)]

        return signals
