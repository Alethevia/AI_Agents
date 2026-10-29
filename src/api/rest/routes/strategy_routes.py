"""
Strategy routes.

This module provides endpoints for strategy management including listing,
starting, stopping, and monitoring trading strategies.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...middleware.auth import get_current_active_user, require_permission
from ...models.api_models import (
    StrategyConfig,
    StrategyResponse,
    StrategyPerformanceResponse,
    StartStrategyRequest,
    StopStrategyRequest,
    MessageResponse,
    StrategyStatus,
)


router = APIRouter(prefix="/strategies", tags=["Strategies"])


# Mock strategy storage (in production, use database)
STRATEGIES_DB = {}


# ============================================================================
# Strategy Management
# ============================================================================

@router.get("", response_model=List[StrategyResponse])
async def list_strategies(
    status: Optional[StrategyStatus] = Query(None, description="Filter by status"),
    current_user: dict = Depends(require_permission("read:strategies"))
):
    """
    List all trading strategies.

    Args:
        status: Optional status filter
        current_user: Current authenticated user

    Returns:
        List of strategies
    """
    strategies = list(STRATEGIES_DB.values())

    # Filter by status if provided
    if status:
        strategies = [s for s in strategies if s.status == status]

    return strategies


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str,
    current_user: dict = Depends(require_permission("read:strategies"))
):
    """
    Get strategy details by ID.

    Args:
        strategy_id: Strategy identifier
        current_user: Current authenticated user

    Returns:
        Strategy details
    """
    if strategy_id not in STRATEGIES_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )

    return STRATEGIES_DB[strategy_id]


@router.post("", response_model=StrategyResponse)
async def create_strategy(
    config: StrategyConfig,
    current_user: dict = Depends(require_permission("write:strategies"))
):
    """
    Create a new trading strategy.

    Args:
        config: Strategy configuration
        current_user: Current authenticated user

    Returns:
        Created strategy
    """
    # Generate strategy ID
    strategy_id = str(uuid4())

    # Create strategy
    strategy = StrategyResponse(
        strategy_id=strategy_id,
        name=config.name,
        status=StrategyStatus.STOPPED,
        symbols=config.symbols,
        parameters=config.parameters,
        created_at=datetime.utcnow(),
    )

    # Store strategy
    STRATEGIES_DB[strategy_id] = strategy

    return strategy


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: str,
    config: StrategyConfig,
    current_user: dict = Depends(require_permission("write:strategies"))
):
    """
    Update strategy configuration.

    Args:
        strategy_id: Strategy identifier
        config: Updated strategy configuration
        current_user: Current authenticated user

    Returns:
        Updated strategy
    """
    if strategy_id not in STRATEGIES_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )

    strategy = STRATEGIES_DB[strategy_id]

    # Don't allow updating running strategies
    if strategy.status == StrategyStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update running strategy. Stop it first."
        )

    # Update configuration
    strategy.name = config.name
    strategy.symbols = config.symbols
    strategy.parameters = config.parameters

    return strategy


@router.delete("/{strategy_id}", response_model=MessageResponse)
async def delete_strategy(
    strategy_id: str,
    current_user: dict = Depends(require_permission("write:strategies"))
):
    """
    Delete a strategy.

    Args:
        strategy_id: Strategy identifier
        current_user: Current authenticated user

    Returns:
        Success message
    """
    if strategy_id not in STRATEGIES_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )

    strategy = STRATEGIES_DB[strategy_id]

    # Don't allow deleting running strategies
    if strategy.status == StrategyStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete running strategy. Stop it first."
        )

    # Delete strategy
    del STRATEGIES_DB[strategy_id]

    return MessageResponse(message=f"Strategy {strategy_id} deleted successfully")


# ============================================================================
# Strategy Control
# ============================================================================

@router.post("/start", response_model=StrategyResponse)
async def start_strategy(
    request: StartStrategyRequest,
    current_user: dict = Depends(require_permission("write:strategies"))
):
    """
    Start a trading strategy.

    Args:
        request: Start strategy request
        current_user: Current authenticated user

    Returns:
        Updated strategy
    """
    strategy_id = request.strategy_id

    if strategy_id not in STRATEGIES_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )

    strategy = STRATEGIES_DB[strategy_id]

    # Check if already running
    if strategy.status == StrategyStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy is already running"
        )

    # Update parameters if provided
    if request.config:
        strategy.parameters.update(request.config)

    # Start strategy
    strategy.status = StrategyStatus.RUNNING
    strategy.started_at = datetime.utcnow()
    strategy.stopped_at = None

    # TODO: Integration with actual strategy execution system
    # await strategy_engine.start_strategy(strategy_id)

    return strategy


@router.post("/stop", response_model=StrategyResponse)
async def stop_strategy(
    request: StopStrategyRequest,
    current_user: dict = Depends(require_permission("write:strategies"))
):
    """
    Stop a trading strategy.

    Args:
        request: Stop strategy request
        current_user: Current authenticated user

    Returns:
        Updated strategy
    """
    strategy_id = request.strategy_id

    if strategy_id not in STRATEGIES_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )

    strategy = STRATEGIES_DB[strategy_id]

    # Check if running
    if strategy.status != StrategyStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Strategy is not running"
        )

    # Stop strategy
    strategy.status = StrategyStatus.STOPPED
    strategy.stopped_at = datetime.utcnow()

    # TODO: Integration with actual strategy execution system
    # await strategy_engine.stop_strategy(strategy_id)

    return strategy


@router.post("/{strategy_id}/pause", response_model=StrategyResponse)
async def pause_strategy(
    strategy_id: str,
    current_user: dict = Depends(require_permission("write:strategies"))
):
    """
    Pause a trading strategy.

    Args:
        strategy_id: Strategy identifier
        current_user: Current authenticated user

    Returns:
        Updated strategy
    """
    if strategy_id not in STRATEGIES_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )

    strategy = STRATEGIES_DB[strategy_id]

    # Check if running
    if strategy.status != StrategyStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only pause running strategies"
        )

    # Pause strategy
    strategy.status = StrategyStatus.PAUSED

    return strategy


@router.post("/{strategy_id}/resume", response_model=StrategyResponse)
async def resume_strategy(
    strategy_id: str,
    current_user: dict = Depends(require_permission("write:strategies"))
):
    """
    Resume a paused strategy.

    Args:
        strategy_id: Strategy identifier
        current_user: Current authenticated user

    Returns:
        Updated strategy
    """
    if strategy_id not in STRATEGIES_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )

    strategy = STRATEGIES_DB[strategy_id]

    # Check if paused
    if strategy.status != StrategyStatus.PAUSED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only resume paused strategies"
        )

    # Resume strategy
    strategy.status = StrategyStatus.RUNNING

    return strategy


# ============================================================================
# Strategy Performance
# ============================================================================

@router.get("/{strategy_id}/performance", response_model=StrategyPerformanceResponse)
async def get_strategy_performance(
    strategy_id: str,
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    current_user: dict = Depends(require_permission("read:strategies"))
):
    """
    Get strategy performance metrics.

    Args:
        strategy_id: Strategy identifier
        start_date: Optional start date for performance calculation
        end_date: Optional end date for performance calculation
        current_user: Current authenticated user

    Returns:
        Strategy performance metrics
    """
    if strategy_id not in STRATEGIES_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )

    strategy = STRATEGIES_DB[strategy_id]

    # TODO: Integration with actual strategy execution system
    # This is a mock implementation

    return StrategyPerformanceResponse(
        strategy_id=strategy_id,
        name=strategy.name,
        total_trades=85,
        winning_trades=52,
        losing_trades=33,
        total_pnl=Decimal("5250.00"),
        win_rate=Decimal("61.18"),
        average_pnl_per_trade=Decimal("61.76"),
        sharpe_ratio=Decimal("1.65"),
        max_drawdown=Decimal("-3.2"),
    )


@router.get("/{strategy_id}/signals")
async def get_strategy_signals(
    strategy_id: str,
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of signals"),
    current_user: dict = Depends(require_permission("read:strategies"))
):
    """
    Get strategy trading signals.

    Args:
        strategy_id: Strategy identifier
        start_date: Optional start date filter
        end_date: Optional end date filter
        limit: Maximum number of signals to return
        current_user: Current authenticated user

    Returns:
        List of trading signals
    """
    if strategy_id not in STRATEGIES_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Strategy {strategy_id} not found"
        )

    # Default date range
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=7)

    # TODO: Integration with actual strategy execution system
    # This is a mock implementation

    signals = [
        {
            "signal_id": str(uuid4()),
            "strategy_id": strategy_id,
            "symbol": "AAPL",
            "action": "BUY",
            "price": "155.50",
            "quantity": "100",
            "confidence": "0.85",
            "timestamp": datetime.utcnow().isoformat(),
        }
    ]

    return signals[:limit]


# ============================================================================
# Strategy Statistics
# ============================================================================

@router.get("/statistics/summary")
async def get_strategies_summary(
    current_user: dict = Depends(require_permission("read:strategies"))
):
    """
    Get summary statistics for all strategies.

    Args:
        current_user: Current authenticated user

    Returns:
        Strategy statistics
    """
    total_strategies = len(STRATEGIES_DB)
    running_strategies = sum(1 for s in STRATEGIES_DB.values() if s.status == StrategyStatus.RUNNING)
    paused_strategies = sum(1 for s in STRATEGIES_DB.values() if s.status == StrategyStatus.PAUSED)
    stopped_strategies = sum(1 for s in STRATEGIES_DB.values() if s.status == StrategyStatus.STOPPED)

    return {
        "total_strategies": total_strategies,
        "running_strategies": running_strategies,
        "paused_strategies": paused_strategies,
        "stopped_strategies": stopped_strategies,
        "error_strategies": sum(1 for s in STRATEGIES_DB.values() if s.status == StrategyStatus.ERROR),
        "timestamp": datetime.utcnow().isoformat(),
    }
