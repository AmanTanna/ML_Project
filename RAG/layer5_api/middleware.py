"""
Middleware Module
================

Custom middleware for the RAG API server.
"""

import time
import uuid
import json
import logging
from datetime import datetime
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import traceback
from contextlib import asynccontextmanager

from .auth import rate_limiter, auth_manager
from .models import ErrorResponse

logger = logging.getLogger(__name__)

class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track requests and add request IDs"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'endpoints': {}
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Track request start time
        start_time = time.time()
        
        # Add request ID to headers
        request.headers.__dict__['_list'].append(
            (b'x-request-id', request_id.encode())
        )
        
        self.request_count += 1
        self.request_stats['total_requests'] += 1
        
        # Log request
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Track endpoint statistics
            endpoint = f"{request.method} {request.url.path}"
            if endpoint not in self.request_stats['endpoints']:
                self.request_stats['endpoints'][endpoint] = {
                    'count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'errors': 0
                }
            
            endpoint_stats = self.request_stats['endpoints'][endpoint]
            endpoint_stats['count'] += 1
            endpoint_stats['total_time'] += processing_time
            endpoint_stats['avg_time'] = endpoint_stats['total_time'] / endpoint_stats['count']
            
            # Update success stats
            if response.status_code < 400:
                self.request_stats['successful_requests'] += 1
            else:
                self.request_stats['failed_requests'] += 1
                endpoint_stats['errors'] += 1
            
            # Update average response time
            if self.request_stats['total_requests'] > 0:
                total_time = sum(
                    ep['total_time'] for ep in self.request_stats['endpoints'].values()
                )
                self.request_stats['avg_response_time'] = (
                    total_time / self.request_stats['total_requests']
                )
            
            # Add response headers
            response.headers['X-Request-ID'] = request_id
            response.headers['X-Processing-Time'] = f"{processing_time:.3f}s"
            response.headers['X-Request-Count'] = str(self.request_count)
            
            logger.info(
                f"Request {request_id} completed: {response.status_code} "
                f"in {processing_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            # Handle unexpected errors
            processing_time = time.time() - start_time
            self.request_stats['failed_requests'] += 1
            
            logger.error(
                f"Request {request_id} failed after {processing_time:.3f}s: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            
            # Return error response
            error_response = ErrorResponse(
                error="Internal server error",
                status_code=500,
                path=str(request.url.path),
                details=str(e),
                request_id=request_id
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.dict(),
                headers={
                    'X-Request-ID': request_id,
                    'X-Processing-Time': f"{processing_time:.3f}s"
                }
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get request statistics"""
        return self.request_stats.copy()

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(self, app, exempt_paths: Optional[list] = None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or ['/health', '/docs', '/openapi.json']
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Get client identifier
        client_ip = request.client.host if request.client else 'unknown'
        
        # Check if user is authenticated to get more specific limits
        auth_header = request.headers.get('authorization')
        limit_type = 'default'
        identifier = client_ip
        
        if auth_header and auth_header.startswith('Bearer '):
            try:
                token = auth_header.split(' ')[1]
                payload = auth_manager.verify_token(token)
                username = payload.get('sub')
                role = payload.get('role', 'user')
                
                identifier = f"{username}:{client_ip}"
                
                # Different limits based on role
                if role == 'admin':
                    limit_type = 'admin'
                elif request.url.path.startswith('/api/v1/query'):
                    limit_type = 'query'
                
            except Exception:
                # If token validation fails, use IP-based limiting
                pass
        
        # Check rate limit
        if not rate_limiter.is_allowed(identifier, limit_type):
            remaining = rate_limiter.get_remaining_requests(identifier, limit_type)
            
            logger.warning(f"Rate limit exceeded for {identifier}")
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "status_code": 429,
                    "path": str(request.url.path),
                    "details": "Too many requests. Please try again later.",
                    "remaining_requests": remaining,
                    "timestamp": datetime.now().isoformat()
                },
                headers={
                    'X-RateLimit-Remaining': str(remaining),
                    'Retry-After': '60'
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        remaining = rate_limiter.get_remaining_requests(identifier, limit_type)
        response.headers['X-RateLimit-Remaining'] = str(remaining)
        
        return response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        
        return response

class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with configuration"""
    
    def __init__(self, app, allowed_origins: list = None, allowed_methods: list = None):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ['*']
        self.allowed_methods = allowed_methods or ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
        self.allowed_headers = [
            'Authorization',
            'Content-Type',
            'X-Request-ID',
            'X-API-Key'
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        origin = request.headers.get('origin')
        
        # Handle preflight requests
        if request.method == 'OPTIONS':
            response = Response()
            response.headers['Access-Control-Allow-Methods'] = ', '.join(self.allowed_methods)
            response.headers['Access-Control-Allow-Headers'] = ', '.join(self.allowed_headers)
            response.headers['Access-Control-Max-Age'] = '86400'
        else:
            response = await call_next(request)
        
        # Add CORS headers
        if origin and (self.allowed_origins == ['*'] or origin in self.allowed_origins):
            response.headers['Access-Control-Allow-Origin'] = origin
        elif '*' in self.allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = '*'
        
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Expose-Headers'] = ', '.join([
            'X-Request-ID',
            'X-Processing-Time',
            'X-RateLimit-Remaining'
        ])
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except HTTPException as e:
            # FastAPI HTTPExceptions are handled by FastAPI itself
            raise e
        except ValueError as e:
            logger.error(f"ValueError in request {getattr(request.state, 'request_id', 'unknown')}: {str(e)}")
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error="Bad Request",
                    status_code=400,
                    path=str(request.url.path),
                    details=str(e),
                    request_id=getattr(request.state, 'request_id', None)
                ).dict()
            )
        except PermissionError as e:
            logger.error(f"Permission error in request {getattr(request.state, 'request_id', 'unknown')}: {str(e)}")
            return JSONResponse(
                status_code=403,
                content=ErrorResponse(
                    error="Forbidden",
                    status_code=403,
                    path=str(request.url.path),
                    details=str(e),
                    request_id=getattr(request.state, 'request_id', None)
                ).dict()
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in request {getattr(request.state, 'request_id', 'unknown')}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="Internal Server Error",
                    status_code=500,
                    path=str(request.url.path),
                    details="An unexpected error occurred",
                    request_id=getattr(request.state, 'request_id', None)
                ).dict()
            )

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging"""
    
    def __init__(self, app, log_level: str = 'INFO'):
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Create access logger
        self.access_logger = logging.getLogger('access')
        if not self.access_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.access_logger.addHandler(handler)
            self.access_logger.setLevel(self.log_level)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        request_id = getattr(request.state, 'request_id', 'unknown')
        client_ip = request.client.host if request.client else 'unknown'
        
        self.access_logger.info(
            f"[{request_id}] {request.method} {request.url} "
            f"from {client_ip} - User-Agent: {request.headers.get('user-agent', 'unknown')}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        processing_time = time.time() - start_time
        self.access_logger.info(
            f"[{request_id}] Response: {response.status_code} "
            f"in {processing_time:.3f}s - Size: {response.headers.get('content-length', 'unknown')}"
        )
        
        return response

class CacheMiddleware(BaseHTTPMiddleware):
    """Simple response caching middleware"""
    
    def __init__(self, app, cache_ttl: int = 300):  # 5 minutes default
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cacheable_methods = ['GET']
        self.cacheable_paths = ['/api/v1/health', '/api/v1/openai/models']
    
    def _get_cache_key(self, request: Request) -> str:
        """Generate cache key for request"""
        return f"{request.method}:{request.url}"
    
    def _is_cacheable(self, request: Request) -> bool:
        """Check if request is cacheable"""
        return (
            request.method in self.cacheable_methods and
            any(request.url.path.startswith(path) for path in self.cacheable_paths)
        )
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        cached_time = cache_entry.get('timestamp', 0)
        return time.time() - cached_time < self.cache_ttl
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if request is cacheable
        if not self._is_cacheable(request):
            return await call_next(request)
        
        cache_key = self._get_cache_key(request)
        
        # Check cache
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.debug(f"Cache hit for {cache_key}")
                cached_response = cache_entry['response']
                
                # Create response from cache
                response = Response(
                    content=cached_response['content'],
                    status_code=cached_response['status_code'],
                    headers=cached_response['headers']
                )
                response.headers['X-Cache'] = 'HIT'
                return response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            # Read response content
            response_body = b''
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Cache the response
            self.cache[cache_key] = {
                'response': {
                    'content': response_body,
                    'status_code': response.status_code,
                    'headers': dict(response.headers)
                },
                'timestamp': time.time()
            }
            
            # Create new response with cached content
            response = Response(
                content=response_body,
                status_code=response.status_code,
                headers=response.headers
            )
            
            logger.debug(f"Cached response for {cache_key}")
        
        response.headers['X-Cache'] = 'MISS'
        return response
    
    def clear_cache(self):
        """Clear all cached responses"""
        self.cache.clear()
        logger.info("Response cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.cache)
        valid_entries = sum(
            1 for entry in self.cache.values()
            if self._is_cache_valid(entry)
        )
        
        return {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'hit_rate': 0.0,  # Would need to track hits/misses
            'cache_ttl': self.cache_ttl
        }

# Global middleware instances
request_tracking_middleware = None
cache_middleware = None

def get_request_stats():
    """Get request statistics from middleware"""
    if request_tracking_middleware:
        return request_tracking_middleware.get_stats()
    return {}

def get_cache_stats():
    """Get cache statistics from middleware"""
    if cache_middleware:
        return cache_middleware.get_cache_stats()
    return {}

def clear_cache():
    """Clear response cache"""
    if cache_middleware:
        cache_middleware.clear_cache()

@asynccontextmanager
async def middleware_context():
    """Context manager for middleware initialization"""
    global request_tracking_middleware, cache_middleware
    
    try:
        # Initialize middleware instances if needed
        yield
    finally:
        # Cleanup if needed
        pass
