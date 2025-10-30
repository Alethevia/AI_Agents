"""
Authentication and authorization middleware.

This module provides JWT-based authentication, API key authentication,
and role-based access control for the trading robot API.
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import jwt
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
    APIKeyHeader,
)
from passlib.context import CryptContext


# ============================================================================
# Configuration
# ============================================================================

# JWT Configuration
JWT_SECRET_KEY = secrets.token_urlsafe(32)  # In production, load from config
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============================================================================
# Data Stores (In production, use database)
# ============================================================================

# User store (username -> user data)
USERS_DB: Dict[str, Dict] = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),  # Change in production!
        "email": "admin@example.com",
        "roles": ["admin", "trader"],
        "disabled": False,
        "created_at": datetime.utcnow(),
    }
}

# API key store (key_id -> key data)
API_KEYS_DB: Dict[str, Dict] = {}

# Active tokens (token_id -> token data) - for revocation
ACTIVE_TOKENS: Dict[str, Dict] = {}


# ============================================================================
# Role Permissions
# ============================================================================

ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    "admin": {
        "read:portfolio",
        "write:portfolio",
        "read:orders",
        "write:orders",
        "read:strategies",
        "write:strategies",
        "read:market_data",
        "read:risk",
        "write:risk",
        "admin:users",
        "admin:api_keys",
    },
    "trader": {
        "read:portfolio",
        "read:orders",
        "write:orders",
        "read:strategies",
        "write:strategies",
        "read:market_data",
        "read:risk",
    },
    "viewer": {
        "read:portfolio",
        "read:orders",
        "read:strategies",
        "read:market_data",
        "read:risk",
    },
}


# ============================================================================
# Password Management
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


# ============================================================================
# User Management
# ============================================================================

def get_user(username: str) -> Optional[Dict]:
    """Get user by username."""
    return USERS_DB.get(username)


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate a user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    if user.get("disabled", False):
        return None
    return user


def create_user(
    username: str,
    password: str,
    email: str,
    roles: List[str] = None
) -> Dict:
    """Create a new user."""
    if username in USERS_DB:
        raise ValueError(f"User {username} already exists")

    user = {
        "username": username,
        "hashed_password": get_password_hash(password),
        "email": email,
        "roles": roles or ["viewer"],
        "disabled": False,
        "created_at": datetime.utcnow(),
    }
    USERS_DB[username] = user
    return user


# ============================================================================
# JWT Token Management
# ============================================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    token_id = str(uuid.uuid4())
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": token_id,
        "type": "access"
    })

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    # Store token for revocation tracking
    ACTIVE_TOKENS[token_id] = {
        "username": data.get("sub"),
        "type": "access",
        "created_at": datetime.utcnow(),
        "expires_at": expire,
    }

    return encoded_jwt


def create_refresh_token(username: str) -> str:
    """Create a JWT refresh token."""
    expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    token_id = str(uuid.uuid4())

    to_encode = {
        "sub": username,
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": token_id,
        "type": "refresh"
    }

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    # Store token for revocation tracking
    ACTIVE_TOKENS[token_id] = {
        "username": username,
        "type": "refresh",
        "created_at": datetime.utcnow(),
        "expires_at": expire,
    }

    return encoded_jwt


def decode_token(token: str) -> Dict:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        # Check if token is revoked
        token_id = payload.get("jti")
        if token_id and token_id not in ACTIVE_TOKENS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
            )

        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


def revoke_token(token: str) -> bool:
    """Revoke a token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        token_id = payload.get("jti")
        if token_id and token_id in ACTIVE_TOKENS:
            del ACTIVE_TOKENS[token_id]
            return True
    except jwt.JWTError:
        pass
    return False


def cleanup_expired_tokens():
    """Remove expired tokens from the active tokens store."""
    now = datetime.utcnow()
    expired = [
        token_id for token_id, data in ACTIVE_TOKENS.items()
        if data["expires_at"] < now
    ]
    for token_id in expired:
        del ACTIVE_TOKENS[token_id]


# ============================================================================
# API Key Management
# ============================================================================

def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def create_api_key(
    username: str,
    name: str,
    permissions: List[str] = None,
    expires_in_days: Optional[int] = None
) -> Dict:
    """Create a new API key."""
    key_id = str(uuid.uuid4())
    api_key = f"ak_{secrets.token_urlsafe(32)}"

    expires_at = None
    if expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

    key_data = {
        "key_id": key_id,
        "hashed_key": hash_api_key(api_key),
        "name": name,
        "username": username,
        "permissions": permissions or [],
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
        "last_used": None,
        "disabled": False,
    }

    API_KEYS_DB[key_id] = key_data

    return {
        "key_id": key_id,
        "api_key": api_key,  # Only returned once
        "name": name,
        "created_at": key_data["created_at"],
        "expires_at": expires_at,
        "permissions": permissions or [],
    }


def validate_api_key(api_key: str) -> Optional[Dict]:
    """Validate an API key and return associated data."""
    hashed = hash_api_key(api_key)

    for key_data in API_KEYS_DB.values():
        if key_data["hashed_key"] == hashed:
            # Check if disabled
            if key_data.get("disabled", False):
                return None

            # Check if expired
            if key_data["expires_at"] and key_data["expires_at"] < datetime.utcnow():
                return None

            # Update last used
            key_data["last_used"] = datetime.utcnow()

            return key_data

    return None


def revoke_api_key(key_id: str) -> bool:
    """Revoke an API key."""
    if key_id in API_KEYS_DB:
        API_KEYS_DB[key_id]["disabled"] = True
        return True
    return False


# ============================================================================
# Authentication Dependencies
# ============================================================================

async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> Dict:
    """Get current user from JWT token."""
    token = credentials.credentials
    payload = decode_token(token)

    username = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    # Verify token type
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    user = get_user(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if user.get("disabled", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    return user


async def get_current_user_from_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[Dict]:
    """Get current user from API key."""
    if not api_key:
        return None

    key_data = validate_api_key(api_key)
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    username = key_data["username"]
    user = get_user(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if user.get("disabled", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    # Attach API key permissions to user
    user = user.copy()
    user["api_key_permissions"] = key_data.get("permissions", [])

    return user


async def get_current_user(
    request: Request,
    token_user: Optional[Dict] = Depends(get_current_user_from_token),
    api_key_user: Optional[Dict] = Depends(get_current_user_from_api_key)
) -> Dict:
    """Get current user from either JWT token or API key."""
    # Try token first, then API key
    user = token_user or api_key_user

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Attach request info
    user["request_id"] = getattr(request.state, "request_id", None)

    return user


async def get_current_active_user(
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    """Get current active user (not disabled)."""
    if current_user.get("disabled", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    return current_user


# ============================================================================
# Permission Checking
# ============================================================================

def check_permission(user: Dict, required_permission: str) -> bool:
    """Check if user has a specific permission."""
    # Get permissions from roles
    user_permissions = set()
    for role in user.get("roles", []):
        user_permissions.update(ROLE_PERMISSIONS.get(role, set()))

    # Add API key specific permissions if present
    if "api_key_permissions" in user:
        user_permissions.update(user["api_key_permissions"])

    return required_permission in user_permissions


def require_permission(permission: str):
    """Dependency to require a specific permission."""
    async def permission_checker(
        current_user: Dict = Depends(get_current_active_user)
    ):
        if not check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission} required",
            )
        return current_user

    return permission_checker


def require_any_permission(*permissions: str):
    """Dependency to require any of the specified permissions."""
    async def permission_checker(
        current_user: Dict = Depends(get_current_active_user)
    ):
        for permission in permissions:
            if check_permission(current_user, permission):
                return current_user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: one of {permissions} required",
        )

    return permission_checker


def require_role(role: str):
    """Dependency to require a specific role."""
    async def role_checker(
        current_user: Dict = Depends(get_current_active_user)
    ):
        if role not in current_user.get("roles", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role}",
            )
        return current_user

    return role_checker


# ============================================================================
# Optional Authentication
# ============================================================================

async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[Dict]:
    """Get current user if authenticated, None otherwise."""
    try:
        if credentials:
            return await get_current_user_from_token(credentials)
        elif api_key:
            return await get_current_user_from_api_key(api_key)
    except HTTPException:
        pass

    return None
