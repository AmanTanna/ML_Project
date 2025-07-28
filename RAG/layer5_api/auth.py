"""
Authentication Module
====================

JWT-based authentication system for the RAG API.
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    """Manages JWT authentication and user sessions"""
    
    def __init__(self):
        self.secret_key = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this')
        self.algorithm = 'HS256'
        self.access_token_expire_minutes = int(os.getenv('JWT_EXPIRE_MINUTES', '60'))
        self.security = HTTPBearer()
        
        # In-memory user store (replace with database in production)
        self.users = {
            "admin": {
                "username": "admin",
                "password_hash": self._hash_password("admin123"),
                "email": "admin@example.com",
                "role": "admin",
                "permissions": ["read", "write", "admin"],
                "created_at": datetime.now()
            },
            "user": {
                "username": "user",
                "password_hash": self._hash_password("user123"),
                "email": "user@example.com",
                "role": "user",
                "permissions": ["read"],
                "created_at": datetime.now()
            }
        }
        
        # Active sessions tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def create_access_token(self, username: str, additional_claims: Optional[Dict] = None) -> str:
        """Create a new JWT access token"""
        try:
            user = self.users.get(username)
            if not user:
                raise ValueError(f"User {username} not found")
            
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            
            payload = {
                'sub': username,
                'exp': expire,
                'iat': datetime.utcnow(),
                'role': user['role'],
                'permissions': user['permissions']
            }
            
            if additional_claims:
                payload.update(additional_claims)
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            # Track active session
            self.active_sessions[token] = {
                'username': username,
                'created_at': datetime.utcnow(),
                'expires_at': expire,
                'last_activity': datetime.utcnow()
            }
            
            logger.info(f"Created access token for user: {username}")
            return token
            
        except Exception as e:
            logger.error(f"Error creating access token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username = payload.get('sub')
            
            if not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing subject"
                )
            
            # Check if user still exists
            if username not in self.users:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User no longer exists"
                )
            
            # Update last activity for active session
            if token in self.active_sessions:
                self.active_sessions[token]['last_activity'] = datetime.utcnow()
            
            return payload
            
        except jwt.ExpiredSignatureError:
            # Clean up expired session
            if token in self.active_sessions:
                del self.active_sessions[token]
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user with username and password"""
        user = self.users.get(username)
        if not user:
            return None
        
        if not self._verify_password(password, user['password_hash']):
            return None
        
        # Update last login
        user['last_login'] = datetime.utcnow()
        logger.info(f"User authenticated: {username}")
        return user
    
    def get_current_user(self, token: str) -> Dict[str, Any]:
        """Get current user from token"""
        payload = self.verify_token(token)
        username = payload['sub']
        user = self.users.get(username)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return {
            'username': user['username'],
            'email': user['email'],
            'role': user['role'],
            'permissions': user['permissions'],
            'created_at': user['created_at'],
            'last_login': user.get('last_login')
        }
    
    def check_permission(self, user: Dict[str, Any], required_permission: str) -> bool:
        """Check if user has required permission"""
        return required_permission in user.get('permissions', [])
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a specific token"""
        if token in self.active_sessions:
            del self.active_sessions[token]
            logger.info("Token revoked successfully")
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.utcnow()
        expired_tokens = []
        
        for token, session in self.active_sessions.items():
            if session['expires_at'] < current_time:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.active_sessions[token]
        
        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        self.cleanup_expired_sessions()
        return len(self.active_sessions)
    
    def create_user(self, username: str, password: str, email: str, role: str = "user") -> bool:
        """Create a new user"""
        if username in self.users:
            return False
        
        permissions = ["read"]
        if role == "admin":
            permissions = ["read", "write", "admin"]
        elif role == "editor":
            permissions = ["read", "write"]
        
        self.users[username] = {
            "username": username,
            "password_hash": self._hash_password(password),
            "email": email,
            "role": role,
            "permissions": permissions,
            "created_at": datetime.utcnow()
        }
        
        logger.info(f"Created new user: {username}")
        return True
    
    def update_user_permissions(self, username: str, permissions: List[str]) -> bool:
        """Update user permissions"""
        if username not in self.users:
            return False
        
        self.users[username]['permissions'] = permissions
        logger.info(f"Updated permissions for user: {username}")
        return True
    
    def delete_user(self, username: str) -> bool:
        """Delete a user"""
        if username not in self.users:
            return False
        
        # Revoke all active sessions for this user
        tokens_to_revoke = [
            token for token, session in self.active_sessions.items()
            if session['username'] == username
        ]
        
        for token in tokens_to_revoke:
            del self.active_sessions[token]
        
        del self.users[username]
        logger.info(f"Deleted user: {username}")
        return True

# Global auth manager instance
auth_manager = AuthManager()

# FastAPI dependencies
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """FastAPI dependency to get current authenticated user"""
    token = credentials.credentials
    return auth_manager.get_current_user(token)

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    """FastAPI dependency to get current active user"""
    return current_user

def require_permission(permission: str):
    """Factory function to create permission-checking dependencies"""
    def check_permission(current_user: dict = Depends(get_current_user)):
        if not auth_manager.check_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return check_permission

# Common permission dependencies
require_read = require_permission("read")
require_write = require_permission("write")  
require_admin = require_permission("admin")

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests: Dict[str, List[datetime]] = {}
        self.limits = {
            'default': {'requests': 100, 'window': 3600},  # 100 requests per hour
            'admin': {'requests': 1000, 'window': 3600},   # 1000 requests per hour
            'query': {'requests': 50, 'window': 300},      # 50 queries per 5 minutes
        }
    
    def is_allowed(self, identifier: str, limit_type: str = 'default') -> bool:
        """Check if request is allowed under rate limit"""
        now = datetime.utcnow()
        limit_config = self.limits.get(limit_type, self.limits['default'])
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests outside the window
        window_start = now - timedelta(seconds=limit_config['window'])
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) >= limit_config['requests']:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    def get_remaining_requests(self, identifier: str, limit_type: str = 'default') -> int:
        """Get remaining requests for identifier"""
        if identifier not in self.requests:
            return self.limits[limit_type]['requests']
        
        now = datetime.utcnow()
        limit_config = self.limits.get(limit_type, self.limits['default'])
        window_start = now - timedelta(seconds=limit_config['window'])
        
        current_requests = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]
        
        return max(0, limit_config['requests'] - len(current_requests))

# Global rate limiter instance
rate_limiter = RateLimiter()

async def check_rate_limit(
    request,
    current_user: dict = Depends(get_current_user),
    limit_type: str = 'default'
):
    """FastAPI dependency to check rate limits"""
    identifier = f"{current_user['username']}:{request.client.host}"
    
    if not rate_limiter.is_allowed(identifier, limit_type):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    return current_user
