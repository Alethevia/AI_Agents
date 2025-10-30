"""
Order routes.

This module provides endpoints for order management including creating,
canceling, and querying orders.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...middleware.auth import get_current_active_user, require_permission
from ...models.api_models import (
    CreateOrderRequest,
    OrderResponse,
    CancelOrderRequest,
    OrderStatusResponse,
    MessageResponse,
    OrderStatus,
    OrderSide,
    OrderType,
)


router = APIRouter(prefix="/orders", tags=["Orders"])


# Mock order storage (in production, use database)
ORDERS_DB = {}


# ============================================================================
# Order Creation
# ============================================================================

@router.post("/create", response_model=OrderResponse)
async def create_order(
    order_request: CreateOrderRequest,
    current_user: dict = Depends(require_permission("write:orders"))
):
    """
    Create a new order.

    Args:
        order_request: Order creation request
        current_user: Current authenticated user

    Returns:
        Created order details
    """
    # TODO: Integration with actual order management system
    # This is a mock implementation

    # Generate order ID
    order_id = str(uuid4())

    # Create order
    order = OrderResponse(
        order_id=order_id,
        client_order_id=order_request.client_order_id,
        symbol=order_request.symbol,
        side=order_request.side,
        order_type=order_request.order_type,
        quantity=order_request.quantity,
        price=order_request.price,
        stop_price=order_request.stop_price,
        filled_quantity=Decimal("0"),
        average_price=None,
        status=OrderStatus.PENDING,
        time_in_force=order_request.time_in_force,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    # Store order
    ORDERS_DB[order_id] = order

    # In production, send order to execution system and WebSocket broadcast
    # await broadcast_order_update(current_user["username"], order.dict())

    return order


@router.post("/batch", response_model=List[OrderResponse])
async def create_batch_orders(
    orders: List[CreateOrderRequest],
    current_user: dict = Depends(require_permission("write:orders"))
):
    """
    Create multiple orders in a batch.

    Args:
        orders: List of order creation requests
        current_user: Current authenticated user

    Returns:
        List of created orders
    """
    created_orders = []

    for order_request in orders:
        try:
            order_id = str(uuid4())

            order = OrderResponse(
                order_id=order_id,
                client_order_id=order_request.client_order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=order_request.price,
                stop_price=order_request.stop_price,
                filled_quantity=Decimal("0"),
                average_price=None,
                status=OrderStatus.PENDING,
                time_in_force=order_request.time_in_force,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            ORDERS_DB[order_id] = order
            created_orders.append(order)

        except Exception as e:
            # In production, handle partial failures better
            continue

    return created_orders


# ============================================================================
# Order Cancellation
# ============================================================================

@router.post("/cancel", response_model=MessageResponse)
async def cancel_order(
    cancel_request: CancelOrderRequest,
    current_user: dict = Depends(require_permission("write:orders"))
):
    """
    Cancel an order.

    Args:
        cancel_request: Order cancellation request
        current_user: Current authenticated user

    Returns:
        Success message
    """
    order_id = cancel_request.order_id

    # Check if order exists
    if order_id not in ORDERS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )

    order = ORDERS_DB[order_id]

    # Check if order can be cancelled
    if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel order in {order.status} status"
        )

    # Update order status
    order.status = OrderStatus.CANCELLED
    order.updated_at = datetime.utcnow()

    # In production, send cancellation to execution system
    # await broadcast_order_update(current_user["username"], order.dict())

    return MessageResponse(message=f"Order {order_id} cancelled successfully")


@router.post("/cancel-all", response_model=MessageResponse)
async def cancel_all_orders(
    symbol: Optional[str] = Query(None, description="Cancel orders for specific symbol"),
    current_user: dict = Depends(require_permission("write:orders"))
):
    """
    Cancel all open orders.

    Args:
        symbol: Optional symbol to filter cancellations
        current_user: Current authenticated user

    Returns:
        Success message
    """
    cancelled_count = 0

    for order_id, order in ORDERS_DB.items():
        # Skip if wrong symbol
        if symbol and order.symbol != symbol.upper():
            continue

        # Skip if not cancellable
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            continue

        # Cancel order
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        cancelled_count += 1

    return MessageResponse(message=f"Cancelled {cancelled_count} orders")


# ============================================================================
# Order Queries
# ============================================================================

@router.get("", response_model=List[OrderResponse])
async def get_orders(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[OrderStatus] = Query(None, description="Filter by status"),
    side: Optional[OrderSide] = Query(None, description="Filter by side"),
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of orders"),
    current_user: dict = Depends(require_permission("read:orders"))
):
    """
    Get orders with optional filters.

    Args:
        symbol: Optional symbol filter
        status: Optional status filter
        side: Optional side filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        limit: Maximum number of orders to return
        current_user: Current authenticated user

    Returns:
        List of orders
    """
    # Default date range
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Filter orders
    orders = []
    for order in ORDERS_DB.values():
        # Apply filters
        if symbol and order.symbol != symbol.upper():
            continue
        if status and order.status != status:
            continue
        if side and order.side != side:
            continue
        if order.created_at < start_date or order.created_at > end_date:
            continue

        orders.append(order)

    # Sort by creation date (newest first)
    orders.sort(key=lambda x: x.created_at, reverse=True)

    # Apply limit
    orders = orders[:limit]

    return orders


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: str,
    current_user: dict = Depends(require_permission("read:orders"))
):
    """
    Get order by ID.

    Args:
        order_id: Order identifier
        current_user: Current authenticated user

    Returns:
        Order details
    """
    if order_id not in ORDERS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )

    return ORDERS_DB[order_id]


@router.get("/{order_id}/status", response_model=OrderStatusResponse)
async def get_order_status(
    order_id: str,
    current_user: dict = Depends(require_permission("read:orders"))
):
    """
    Get order status.

    Args:
        order_id: Order identifier
        current_user: Current authenticated user

    Returns:
        Order status information
    """
    if order_id not in ORDERS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )

    order = ORDERS_DB[order_id]

    return OrderStatusResponse(
        order_id=order.order_id,
        status=order.status,
        filled_quantity=order.filled_quantity,
        remaining_quantity=order.quantity - order.filled_quantity,
        average_price=order.average_price,
        updated_at=order.updated_at,
    )


# ============================================================================
# Order History
# ============================================================================

@router.get("/history/fills")
async def get_order_fills(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of fills"),
    current_user: dict = Depends(require_permission("read:orders"))
):
    """
    Get order fills (executions).

    Args:
        symbol: Optional symbol filter
        start_date: Optional start date filter
        end_date: Optional end date filter
        limit: Maximum number of fills to return
        current_user: Current authenticated user

    Returns:
        List of order fills
    """
    # TODO: Integration with actual order management system
    # This is a mock implementation

    # Default date range
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Get filled orders
    fills = []
    for order in ORDERS_DB.values():
        if order.status not in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
            continue

        if symbol and order.symbol != symbol.upper():
            continue

        if order.filled_at and (order.filled_at < start_date or order.filled_at > end_date):
            continue

        fills.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": str(order.filled_quantity),
            "price": str(order.average_price) if order.average_price else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
        })

    # Sort by fill time (newest first)
    fills.sort(key=lambda x: x["filled_at"] or "", reverse=True)

    return fills[:limit]


# ============================================================================
# Order Statistics
# ============================================================================

@router.get("/statistics/summary")
async def get_order_statistics(
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    current_user: dict = Depends(require_permission("read:orders"))
):
    """
    Get order statistics.

    Args:
        start_date: Optional start date
        end_date: Optional end date
        current_user: Current authenticated user

    Returns:
        Order statistics
    """
    # Default date range
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Calculate statistics
    total_orders = 0
    filled_orders = 0
    cancelled_orders = 0
    rejected_orders = 0
    buy_orders = 0
    sell_orders = 0

    for order in ORDERS_DB.values():
        if order.created_at < start_date or order.created_at > end_date:
            continue

        total_orders += 1

        if order.status == OrderStatus.FILLED:
            filled_orders += 1
        elif order.status == OrderStatus.CANCELLED:
            cancelled_orders += 1
        elif order.status == OrderStatus.REJECTED:
            rejected_orders += 1

        if order.side == OrderSide.BUY:
            buy_orders += 1
        else:
            sell_orders += 1

    return {
        "total_orders": total_orders,
        "filled_orders": filled_orders,
        "cancelled_orders": cancelled_orders,
        "rejected_orders": rejected_orders,
        "pending_orders": total_orders - filled_orders - cancelled_orders - rejected_orders,
        "buy_orders": buy_orders,
        "sell_orders": sell_orders,
        "fill_rate": f"{(filled_orders / total_orders * 100):.2f}%" if total_orders > 0 else "0%",
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        }
    }
