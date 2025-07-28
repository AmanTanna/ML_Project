"""
RAG API Server - Part 6
=======================

FastAPI-based REST API server for the RAG system with OpenAI integration.
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# Custom JSON encoder for datetime
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Monkey patch FastAPI's JSONResponse to use our encoder
original_render = JSONResponse.render

def custom_render(self, content):
    if content is None:
        return b""
    return json.dumps(
        content,
        cls=DateTimeEncoder,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
    ).encode("utf-8")

JSONResponse.render = custom_render

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from .models import ErrorResponse
from .auth import auth_manager, get_current_user
from .middleware import (
    RequestTrackingMiddleware, RateLimitMiddleware, SecurityHeadersMiddleware,
    CORSMiddleware, ErrorHandlingMiddleware, LoggingMiddleware, CacheMiddleware,
    request_tracking_middleware, cache_middleware
)
from .endpoints import (
    auth_router, query_router, openai_router, admin_router, health_router,
    set_rag_components
)
from .openai_integration import get_openai_manager

# RAG system imports
try:
    from layer4_query.manager import QueryManager
    from layer1_raw.manager import RawManager
    from layer2_structured.manager import StructuredManager
    from layer3_semantic.manager import SemanticManager
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import RAG components: {e}")
    QueryManager = None
    RawManager = None
    StructuredManager = None
    SemanticManager = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Global RAG system components
rag_system = None
query_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global rag_system, query_manager, request_tracking_middleware, cache_middleware
    
    logger.info("Starting RAG API Server...")
    
    try:
        # Initialize OpenAI manager
        logger.info("Initializing OpenAI integration...")
        try:
            openai_mgr = get_openai_manager()
            logger.info("OpenAI integration initialized successfully")
        except Exception as e:
            logger.warning(f"OpenAI integration failed: {e}")
        
        # Initialize RAG system components
        if all([QueryManager, RawManager, StructuredManager, SemanticManager]):
            logger.info("Initializing RAG system components...")
            
            try:
                # Initialize layers
                raw_manager = RawManager()
                structured_manager = StructuredManager()
                semantic_manager = SemanticManager()
                
                # Initialize query manager
                query_manager = QueryManager(
                    raw_manager=raw_manager,
                    structured_manager=structured_manager,
                    semantic_manager=semantic_manager
                )
                
                # Create global rag_system reference
                rag_system = {
                    'raw_manager': raw_manager,
                    'structured_manager': structured_manager,
                    'semantic_manager': semantic_manager,
                    'query_manager': query_manager
                }
                
                # Set components in endpoints
                set_rag_components(query_manager, rag_system)
                
                logger.info("RAG system initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize RAG system: {str(e)}")
                logger.info("API will run with limited functionality")
        else:
            logger.warning("RAG components not available. Running in limited mode.")
        
        # Initialize middleware instances
        request_tracking_middleware = app.user_middleware[0] if app.user_middleware else None
        cache_middleware = None
        for middleware in app.user_middleware:
            if isinstance(middleware, CacheMiddleware):
                cache_middleware = middleware
                break
        
        # Health check
        logger.info("Performing system health check...")
        health_status = await perform_health_check()
        logger.info(f"System health: {health_status['status']}")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        # Don't raise - allow API to start in degraded mode
    finally:
        logger.info("Shutting down RAG API Server...")
        # Cleanup if needed

async def perform_health_check() -> Dict[str, Any]:
    """Perform system health check"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Check query manager
        if query_manager:
            try:
                # Simple test query
                test_result = await query_manager.process_query({
                    'query_text': 'test',
                    'query_type': 'semantic',
                    'max_results': 1
                })
                status['components']['query_manager'] = 'healthy'
            except Exception as e:
                status['components']['query_manager'] = f'error: {str(e)}'
                status['status'] = 'degraded'
        else:
            status['components']['query_manager'] = 'not_initialized'
            status['status'] = 'degraded'
        
        # Check OpenAI integration
        try:
            openai_mgr = get_openai_manager()
            status['components']['openai'] = 'configured'
        except Exception as e:
            status['components']['openai'] = f'error: {str(e)}'
            status['status'] = 'degraded'
        
        # Check authentication
        try:
            auth_manager.cleanup_expired_sessions()
            status['components']['auth'] = 'healthy'
        except Exception as e:
            status['components']['auth'] = f'error: {str(e)}'
            status['status'] = 'degraded'
        
        return status
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Create FastAPI application
app = FastAPI(
    title="RAG Financial Document Analysis API",
    description="""
    A comprehensive RAG (Retrieval-Augmented Generation) system for financial document analysis.
    
    This API provides:
    - Multi-layer query engine (semantic, structured, raw document search)
    - OpenAI GPT integration for enhanced responses
    - Document upload and processing
    - Real-time analytics and monitoring
    - JWT-based authentication and authorization
    
    ## Authentication
    Most endpoints require authentication. Use the `/auth/login` endpoint to obtain an access token.
    
    Default users:
    - Admin: username=admin, password=admin123
    - User: username=user, password=user123
    
    ## Rate Limiting
    API calls are rate-limited per user. See the response headers for remaining quota.
    
    ## Query Types
    - **auto**: Automatically selects the best query strategy
    - **semantic**: Vector-based semantic search
    - **structured**: SQL-like queries on structured data
    - **raw**: Full-text search on raw documents
    - **hybrid**: Combines multiple approaches
    
    ## OpenAI Integration
    Use `/api/v1/openai/chat` for direct OpenAI chat or `/api/v1/openai/rag-chat` for RAG-enhanced responses.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware in correct order (last added = first executed)
from starlette.middleware.cors import CORSMiddleware as StarletterCORSMiddleware

app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, exempt_paths=['/health', '/docs', '/openapi.json', '/auth/login'])
app.add_middleware(CacheMiddleware, cache_ttl=300)
app.add_middleware(
    StarletterCORSMiddleware,
    allow_origins=os.getenv('CORS_ORIGINS', '*').split(','),
    allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allow_headers=['*'],
    allow_credentials=True
)

# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
app.include_router(openai_router, prefix="/api/v1") 
app.include_router(admin_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Financial Document Analysis API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "auth": "/api/v1/auth/login",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/info")
async def api_info():
    """Get API information"""
    return {
        "name": "RAG Financial Document Analysis API",
        "version": "1.0.0",
        "description": "Multi-layer RAG system for financial document analysis",
        "features": [
            "Multi-layer query engine",
            "OpenAI GPT integration",
            "JWT authentication",
            "Rate limiting",
            "Real-time monitoring",
            "Document processing"
        ],
        "endpoints": {
            "auth": "/api/v1/auth",
            "query": "/api/v1/query",
            "openai": "/api/v1/openai", 
            "admin": "/api/v1/admin",
            "health": "/api/v1/health"
        }
    }

# Custom error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
            path=str(request.url.path),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Bad Request",
            status_code=400,
            path=str(request.url.path),
            details=str(exc),
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
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

# OpenAPI customization
def custom_openapi():
    """Custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="RAG Financial Document Analysis API",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "HTTPBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add security requirement
    openapi_schema["security"] = [{"HTTPBearer": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Development server
class APIServer:
    """API Server runner class"""
    
    def __init__(self):
        self.host = os.getenv('API_HOST', 'localhost')
        self.port = int(os.getenv('API_PORT', '8000'))
        self.reload = os.getenv('API_RELOAD', 'false').lower() == 'true'
        self.log_level = os.getenv('API_LOG_LEVEL', 'info')
        
    def run(self):
        """Run the API server"""
        logger.info(f"Starting API server on {self.host}:{self.port}")
        
        uvicorn.run(
            "RAG.layer5_api.api_server:app",
            host=self.host,
            port=self.port,
            reload=self.reload,
            log_level=self.log_level,
            access_log=True
        )

if __name__ == "__main__":
    server = APIServer()
    server.run()
