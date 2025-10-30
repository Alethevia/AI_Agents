"""
FastAPI main application.

This module sets up the FastAPI application with all routes, middleware,
and configuration for the trading robot REST API.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..middleware.auth import cleanup_expired_tokens
from ..middleware.rate_limiter import RateLimitMiddleware
from ..models.api_models import HealthCheckResponse, ErrorResponse, ErrorDetail
from ..websocket.ws_server import websocket_endpoint, start_websocket_tasks
from .routes import (
    auth_routes,
    portfolio_routes,
    order_routes,
    strategy_routes,
    market_data_routes,
    risk_routes,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Trading Robot API...")

    # Start background tasks
    await start_websocket_tasks()

    # Initialize components
    # TODO: Initialize database connections, cache, etc.

    logger.info("Trading Robot API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Trading Robot API...")

    # Cleanup
    cleanup_expired_tokens()

    # TODO: Close database connections, cleanup resources

    logger.info("Trading Robot API shut down successfully")


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Trading Robot API",
    description="""
    # Trading Robot REST API

    A comprehensive REST API for algorithmic trading with features including:

    - **Authentication**: JWT-based authentication and API keys
    - **Portfolio Management**: Track positions, performance, and history
    - **Order Management**: Create, cancel, and monitor orders
    - **Strategy Management**: Deploy and manage trading strategies
    - **Market Data**: Real-time and historical market data
    - **Risk Management**: Monitor risk metrics and alerts
    - **WebSocket**: Real-time data streaming

    ## Authentication

    Most endpoints require authentication using either:
    - Bearer token (JWT) in the Authorization header
    - API key in the X-API-Key header

    To get started:
    1. Login via `/api/v1/auth/login` to get an access token
    2. Use the token in the Authorization header: `Bearer <token>`

    ## Rate Limiting

    All endpoints are rate limited to ensure fair usage:
    - Global: 1000 requests per minute
    - Per user: 500 requests per minute
    - Per IP (unauthenticated): 100 requests per minute
    - Endpoint-specific limits apply

    Rate limit headers are included in responses:
    - X-RateLimit-Limit
    - X-RateLimit-Remaining
    - X-RateLimit-Reset

    ## WebSocket

    Connect to `/ws` for real-time data streaming.
    Subscribe to topics to receive updates.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ============================================================================
# Middleware Configuration
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        # Add your frontend origins here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
)

# Rate Limiting Middleware
app.add_middleware(RateLimitMiddleware)


# Request ID Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response


# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()

    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"[{getattr(request.state, 'request_id', 'unknown')}]"
    )

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Log response
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"[{getattr(request.state, 'request_id', 'unknown')}] "
        f"Status: {response.status_code} Duration: {duration:.3f}s"
    )

    # Add timing header
    response.headers["X-Process-Time"] = str(duration)

    return response


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    errors = []
    for error in exc.errors():
        errors.append(
            ErrorDetail(
                field=".".join(str(loc) for loc in error["loc"]),
                message=error["msg"],
                code=error["type"],
            )
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="ValidationError",
            message="Request validation failed",
            details=errors,
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
    )


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "name": "Trading Robot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns system health status and uptime.
    """
    # TODO: Add actual health checks for dependencies
    # - Database connection
    # - Cache connection
    # - External services

    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        uptime=0.0,  # TODO: Calculate actual uptime
        services={
            "database": "healthy",
            "cache": "healthy",
            "market_data": "healthy",
        }
    )


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint.

    Returns whether the service is ready to accept requests.
    """
    # TODO: Add actual readiness checks
    return {"status": "ready"}


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """
    Liveness check endpoint.

    Returns whether the service is alive.
    """
    return {"status": "alive"}


# ============================================================================
# API Routes
# ============================================================================

# API version prefix
API_V1_PREFIX = "/api/v1"

# Include routers
app.include_router(auth_routes.router, prefix=API_V1_PREFIX)
app.include_router(portfolio_routes.router, prefix=API_V1_PREFIX)
app.include_router(order_routes.router, prefix=API_V1_PREFIX)
app.include_router(strategy_routes.router, prefix=API_V1_PREFIX)
app.include_router(market_data_routes.router, prefix=API_V1_PREFIX)
app.include_router(risk_routes.router, prefix=API_V1_PREFIX)


# ============================================================================
# WebSocket Route
# ============================================================================

@app.websocket("/ws")
async def websocket_route(websocket):
    """WebSocket endpoint for real-time data streaming."""
    await websocket_endpoint(websocket)


# ============================================================================
# Admin/Monitoring Endpoints
# ============================================================================

@app.get("/admin/stats", tags=["Admin"])
async def get_stats(request: Request):
    """
    Get API statistics.

    Note: In production, this should be protected with admin authentication.
    """
    from ..middleware.rate_limiter import get_rate_limit_stats
    from ..websocket.ws_server import get_websocket_stats

    return {
        "rate_limiting": get_rate_limit_stats(),
        "websocket": get_websocket_stats(),
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
