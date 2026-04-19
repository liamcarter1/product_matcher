"""
Authentication API endpoints (Optional).
Simple JWT-based authentication for distributor access.
"""

from datetime import datetime, timedelta
from typing import Optional
import hashlib
import secrets
import hmac

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from ..config import get_settings
from ..models.schemas import LoginRequest, LoginResponse, UserInfo

router = APIRouter()

# Security
security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Simple password hashing using SHA-256 with salt."""
    salt = "danfoss_rag_salt_v1"  # In production, use unique salt per user
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return hmac.compare_digest(hash_password(password), hashed)


# Demo users (in production, use a database with proper bcrypt)
DEMO_USERS = {
    "distributor1": {
        "password_hash": hash_password("demo123"),
        "distributor_id": "DIST001",
        "name": "Demo Distributor",
        "region": "North America"
    },
    "distributor2": {
        "password_hash": hash_password("demo456"),
        "distributor_id": "DIST002",
        "name": "European Partner",
        "region": "Europe"
    }
}


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    settings = get_settings()

    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """Verify a JWT token and return the payload."""
    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[UserInfo]:
    """Get the current authenticated user from the token."""
    if not credentials:
        return None

    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = payload.get("sub")
    if username not in DEMO_USERS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    user_data = DEMO_USERS[username]
    return UserInfo(
        distributor_id=user_data["distributor_id"],
        name=user_data["name"],
        region=user_data.get("region")
    )


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[UserInfo]:
    """Get the current user if authenticated, None otherwise."""
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate a distributor and return an access token.

    Demo credentials:
    - username: distributor1, password: demo123
    - username: distributor2, password: demo456
    """
    user = DEMO_USERS.get(request.username)

    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": request.username, "distributor_id": user["distributor_id"]}
    )

    return LoginResponse(access_token=access_token)


@router.get("/me", response_model=UserInfo)
async def get_current_user_info(
    current_user: UserInfo = Depends(get_current_user)
):
    """Get the current authenticated user's information."""
    return current_user


@router.post("/logout")
async def logout():
    """
    Logout endpoint.
    Note: JWT tokens are stateless, so this is mainly for client-side cleanup.
    In production, consider token blacklisting with Redis.
    """
    return {"status": "logged_out", "message": "Clear the token on client side"}
