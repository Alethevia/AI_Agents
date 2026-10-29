"""
Authentication routes.

This module provides endpoints for user authentication, token management,
and API key operations.
"""

from datetime import timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from ...middleware.auth import (
    authenticate_user,
    create_access_token,
    create_refresh_token,
    create_api_key,
    revoke_api_key,
    decode_token,
    get_current_active_user,
    require_permission,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    API_KEYS_DB,
)
from ...models.api_models import (
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    APIKeyRequest,
    APIKeyResponse,
    MessageResponse,
)


router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============================================================================
# Authentication Endpoints
# ============================================================================

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return access token.

    Args:
        request: Login credentials

    Returns:
        Access token and refresh token
    """
    user = authenticate_user(request.username, request.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create tokens
    access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(user["username"])

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token.

    Args:
        request: Refresh token

    Returns:
        New access token
    """
    try:
        payload = decode_token(request.refresh_token)

        # Verify token type
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        # Create new access token
        access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": username},
            expires_delta=access_token_expires
        )

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate refresh token",
        )


@router.post("/logout", response_model=MessageResponse)
async def logout(current_user: dict = Depends(get_current_active_user)):
    """
    Logout user (invalidate token).

    Args:
        current_user: Current authenticated user

    Returns:
        Success message
    """
    # In a real implementation, you would revoke the token
    # For now, just return success
    return MessageResponse(message="Successfully logged out")


# ============================================================================
# API Key Management
# ============================================================================

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_new_api_key(
    request: APIKeyRequest,
    current_user: dict = Depends(require_permission("admin:api_keys"))
):
    """
    Create a new API key.

    Args:
        request: API key creation request
        current_user: Current authenticated user

    Returns:
        API key details (key is only shown once)
    """
    try:
        key_data = create_api_key(
            username=current_user["username"],
            name=request.name,
            permissions=request.permissions,
            expires_in_days=request.expires_in_days,
        )

        return APIKeyResponse(**key_data)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}"
        )


@router.get("/api-keys", response_model=List[dict])
async def list_api_keys(
    current_user: dict = Depends(require_permission("admin:api_keys"))
):
    """
    List all API keys for current user.

    Args:
        current_user: Current authenticated user

    Returns:
        List of API keys (without actual key values)
    """
    username = current_user["username"]

    keys = []
    for key_data in API_KEYS_DB.values():
        if key_data["username"] == username:
            keys.append({
                "key_id": key_data["key_id"],
                "name": key_data["name"],
                "created_at": key_data["created_at"],
                "expires_at": key_data["expires_at"],
                "last_used": key_data["last_used"],
                "disabled": key_data["disabled"],
                "permissions": key_data["permissions"],
            })

    return keys


@router.delete("/api-keys/{key_id}", response_model=MessageResponse)
async def revoke_api_key_endpoint(
    key_id: str,
    current_user: dict = Depends(require_permission("admin:api_keys"))
):
    """
    Revoke an API key.

    Args:
        key_id: API key ID to revoke
        current_user: Current authenticated user

    Returns:
        Success message
    """
    # Verify key belongs to user
    key_data = API_KEYS_DB.get(key_id)
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    if key_data["username"] != current_user["username"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to revoke this API key"
        )

    # Revoke key
    if revoke_api_key(key_id):
        return MessageResponse(message="API key revoked successfully")
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key"
        )


# ============================================================================
# User Info
# ============================================================================

@router.get("/me")
async def get_current_user_info(
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        User information (without password)
    """
    return {
        "username": current_user["username"],
        "email": current_user.get("email"),
        "roles": current_user.get("roles", []),
        "created_at": current_user.get("created_at"),
    }
