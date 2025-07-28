"""
API Endpoints
============

FastAPI route handlers for all API endpoints.
"""

import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request
from fastapi.responses import StreamingResponse
import logging
import asyncio
import json

from .models import (
    QueryRequest, QueryResponse, HealthResponse, StatsResponse,
    LoginRequest, LoginResponse, ErrorResponse, OpenAIRequest, OpenAIResponse,
    RAGChatRequest, RAGChatResponse, BatchQueryRequest, BatchQueryResponse,
    DocumentUploadRequest, DocumentUploadResponse, DocumentSearchRequest,
    DocumentSearchResponse, SystemConfig, ConfigUpdateRequest
)
from .auth import (
    auth_manager, get_current_user, require_read, require_write, require_admin,
    check_rate_limit, rate_limiter
)
from .openai_integration import get_openai_manager, usage_tracker

logger = logging.getLogger(__name__)

# Create router instances
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
query_router = APIRouter(prefix="/query", tags=["Query Engine"])
openai_router = APIRouter(prefix="/openai", tags=["OpenAI Integration"])
admin_router = APIRouter(prefix="/admin", tags=["Administration"])
health_router = APIRouter(prefix="/health", tags=["Health & Monitoring"])

# Global references to RAG components (will be injected)
query_manager = None
rag_system = None

def set_rag_components(query_mgr, rag_sys):
    """Set RAG system components"""
    global query_manager, rag_system
    query_manager = query_mgr
    rag_system = rag_sys

# Authentication Endpoints
@auth_router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return access token"""
    try:
        user = auth_manager.authenticate_user(request.username, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        access_token = auth_manager.create_access_token(request.username)
        
        return LoginResponse(
            access_token=access_token,
            expires_in=auth_manager.access_token_expire_minutes * 60,
            user={
                'username': user['username'],
                'email': user['email'],
                'role': user['role'],
                'permissions': user['permissions'],
                'created_at': user['created_at'],
                'last_login': user.get('last_login')
            }
        )
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Login failed")

@auth_router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout current user"""
    # In a real implementation, you would revoke the token
    return {"message": "Logged out successfully"}

@auth_router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@auth_router.get("/sessions")
async def get_active_sessions(current_user: dict = Depends(require_admin)):
    """Get active sessions count (admin only)"""
    return {
        "active_sessions": auth_manager.get_active_sessions_count(),
        "timestamp": datetime.now()
    }

# Query Engine Endpoints
@query_router.post("/search", response_model=QueryResponse)
async def search_documents(
    request: QueryRequest,
    current_user: dict = Depends(require_read)
):
    """Search documents using the RAG query engine"""
    try:
        if not query_manager:
            raise HTTPException(status_code=503, detail="Query engine not available")
        
        start_time = time.time()
        logger.info(f"Processing query: {request.query_text[:100]}...")
        
        # Convert API request to internal format
        internal_request = {
            'query_text': request.query_text,
            'query_type': request.query_type.value,
            'companies': request.companies,
            'time_range': request.time_range,
            'sections': request.sections,
            'max_results': request.max_results,
            'include_sources': request.include_sources,
            'explain_reasoning': request.explain_reasoning,
            'use_cache': request.use_cache
        }
        
        # Execute query
        response = await query_manager.process_query(internal_request)
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        # Convert internal response to API format
        return QueryResponse.from_query_response(response)
        
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@query_router.post("/batch", response_model=BatchQueryResponse)
async def batch_search(
    request: BatchQueryRequest,
    current_user: dict = Depends(require_read)
):
    """Process multiple queries in batch"""
    try:
        if not query_manager:
            raise HTTPException(status_code=503, detail="Query engine not available")
        
        start_time = time.time()
        results = []
        successful = 0
        failed = 0
        
        if request.parallel_execution:
            # Process queries in parallel
            tasks = []
            for query_req in request.queries:
                internal_req = {
                    'query_text': query_req.query_text,
                    'query_type': query_req.query_type.value,
                    'companies': query_req.companies,
                    'time_range': query_req.time_range,
                    'sections': query_req.sections,
                    'max_results': query_req.max_results,
                    'include_sources': query_req.include_sources,
                    'explain_reasoning': query_req.explain_reasoning,
                    'use_cache': query_req.use_cache
                }
                tasks.append(query_manager.process_query(internal_req))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for response in responses:
                if isinstance(response, Exception):
                    failed += 1
                    logger.error(f"Batch query failed: {str(response)}")
                else:
                    successful += 1
                    results.append(QueryResponse.from_query_response(response))
        else:
            # Process queries sequentially
            for query_req in request.queries:
                try:
                    internal_req = {
                        'query_text': query_req.query_text,
                        'query_type': query_req.query_type.value,
                        'companies': query_req.companies,
                        'time_range': query_req.time_range,
                        'sections': query_req.sections,
                        'max_results': query_req.max_results,
                        'include_sources': query_req.include_sources,
                        'explain_reasoning': query_req.explain_reasoning,
                        'use_cache': query_req.use_cache
                    }
                    
                    response = await query_manager.process_query(internal_req)
                    results.append(QueryResponse.from_query_response(response))
                    successful += 1
                except Exception as e:
                    failed += 1
                    logger.error(f"Batch query failed: {str(e)}")
        
        total_time = time.time() - start_time
        
        return BatchQueryResponse(
            results=results,
            total_queries=len(request.queries),
            successful_queries=successful,
            failed_queries=failed,
            total_processing_time=total_time
        )
        
    except Exception as e:
        logger.error(f"Batch query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch query failed: {str(e)}")

# OpenAI Integration Endpoints
@openai_router.post("/chat", response_model=OpenAIResponse)
async def openai_chat(
    request: OpenAIRequest,
    current_user: dict = Depends(require_read)
):
    """Direct OpenAI chat completion"""
    try:
        openai_mgr = get_openai_manager()
        
        if not openai_mgr.validate_request(request):
            raise HTTPException(status_code=400, detail="Invalid request format")
        
        response = await openai_mgr.create_chat_completion(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        # Track usage
        usage_tracker.track_usage(response['usage'], request.model, True)
        
        return OpenAIResponse(**response)
        
    except Exception as e:
        usage_tracker.track_usage({}, request.model, False)
        logger.error(f"OpenAI chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@openai_router.post("/rag-chat", response_model=RAGChatResponse)
async def rag_enhanced_chat(
    request: RAGChatRequest,
    current_user: dict = Depends(require_read)
):
    """RAG-enhanced chat using document context"""
    try:
        if not query_manager:
            raise HTTPException(status_code=503, detail="Query engine not available")
        
        openai_mgr = get_openai_manager()
        
        # First, get relevant context from RAG system
        query_request = {
            'query_text': request.query,
            'query_type': 'semantic',
            'max_results': request.max_context_results,
            'include_sources': True,
            'explain_reasoning': False,
            'use_cache': True
        }
        
        rag_response = await query_manager.process_query(query_request)
        
        # Convert results to QueryResult format
        context_results = []
        for result in rag_response.results[:request.max_context_results]:
            context_results.append({
                'result_id': result.result_id,
                'content': result.content,
                'relevance_score': result.relevance_score,
                'source_type': result.source_type,
                'source_document': result.source_document,
                'company': result.company,
                'section': result.section,
                'metadata': result.metadata or {},
                'reasoning': result.reasoning
            })
        
        # Create RAG-enhanced response
        response = await openai_mgr.create_rag_enhanced_response(
            query=request.query,
            context_results=context_results,
            model=request.model,
            temperature=request.temperature
        )
        
        # Track usage
        usage_tracker.track_usage(response['usage'], request.model, True)
        
        return RAGChatResponse(
            response=response['response'],
            sources=context_results,
            rag_context=response['rag_context'],
            processing_time=response['processing_time'],
            model=response['model']
        )
        
    except Exception as e:
        usage_tracker.track_usage({}, request.model, False)
        logger.error(f"RAG chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@openai_router.get("/models")
async def get_available_models(current_user: dict = Depends(require_read)):
    """Get available OpenAI models"""
    try:
        openai_mgr = get_openai_manager()
        models = openai_mgr.get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return {"models": ["gpt-3.5-turbo"]}

@openai_router.get("/usage")
async def get_openai_usage(current_user: dict = Depends(require_admin)):
    """Get OpenAI usage statistics (admin only)"""
    return usage_tracker.get_usage_summary()

# Health & Monitoring Endpoints
@health_router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check"""
    try:
        status = "healthy"
        components = {}
        issues = []
        
        # Check query engine
        if query_manager:
            try:
                # Simple test query
                test_response = await query_manager.process_query({
                    'query_text': 'test',
                    'query_type': 'semantic',
                    'max_results': 1,
                    'use_cache': True
                })
                components['query_engine'] = {
                    'status': 'healthy',
                    'last_check': datetime.now()
                }
            except Exception as e:
                components['query_engine'] = {
                    'status': 'unhealthy',
                    'details': str(e),
                    'last_check': datetime.now()
                }
                issues.append(f"Query engine error: {str(e)}")
                status = "degraded"
        else:
            components['query_engine'] = {
                'status': 'unavailable',
                'details': 'Query manager not initialized'
            }
            issues.append("Query engine not available")
            status = "degraded"
        
        # Check OpenAI
        try:
            openai_mgr = get_openai_manager()
            components['openai'] = {
                'status': 'healthy',
                'last_check': datetime.now()
            }
        except Exception as e:
            components['openai'] = {
                'status': 'unhealthy',
                'details': str(e),
                'last_check': datetime.now()
            }
            issues.append(f"OpenAI integration error: {str(e)}")
            status = "degraded"
        
        # Check authentication
        try:
            auth_manager.cleanup_expired_sessions()
            components['auth'] = {
                'status': 'healthy',
                'details': f"Active sessions: {auth_manager.get_active_sessions_count()}",
                'last_check': datetime.now()
            }
        except Exception as e:
            components['auth'] = {
                'status': 'unhealthy',
                'details': str(e),
                'last_check': datetime.now()
            }
            issues.append(f"Authentication error: {str(e)}")
            status = "degraded"
        
        return HealthResponse(
            status=status,
            message="System health check completed",
            components=components,
            issues=issues
        )
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="error",
            message=f"Health check failed: {str(e)}",
            issues=[str(e)]
        )

@health_router.get("/stats", response_model=StatsResponse)
async def get_system_stats(current_user: dict = Depends(require_admin)):
    """Get detailed system statistics (admin only)"""
    try:
        # Get OpenAI usage stats
        openai_usage = usage_tracker.get_usage_summary()
        
        # Get authentication stats
        auth_stats = {
            'active_sessions': auth_manager.get_active_sessions_count(),
            'total_users': len(auth_manager.users)
        }
        
        # Get rate limiting stats
        rate_limit_stats = {
            'tracked_identifiers': len(rate_limiter.requests),
            'current_limits': rate_limiter.limits
        }
        
        return StatsResponse(
            overview={
                'total_queries': openai_usage.get('total_requests', 0),
                'successful_queries': openai_usage.get('successful_requests', 0),
                'failed_queries': openai_usage.get('failed_requests', 0),
                'success_rate': openai_usage.get('success_rate', 0.0),
                'avg_response_time': 0.0  # Would track this separately
            },
            query_distribution={
                'query_types': {},  # Would track from query manager
                'layer_usage': {}   # Would track from query manager
            },
            components={
                'query_engine': {},
                'query_router': {},
                'results_fusion': {}
            },
            performance={
                'status': 'healthy',
                'alerts': [],
                'thresholds': {
                    'max_response_time': 30.0,
                    'min_success_rate': 0.95
                }
            },
            recent_queries=[]
        )
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system stats")

# Administration Endpoints
@admin_router.post("/users")
async def create_user(
    username: str,
    password: str,
    email: str,
    role: str = "user",
    current_user: dict = Depends(require_admin)
):
    """Create a new user (admin only)"""
    try:
        success = auth_manager.create_user(username, password, email, role)
        if not success:
            raise HTTPException(status_code=400, detail="User already exists")
        
        return {"message": f"User {username} created successfully"}
        
    except Exception as e:
        logger.error(f"User creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@admin_router.delete("/users/{username}")
async def delete_user(
    username: str,
    current_user: dict = Depends(require_admin)
):
    """Delete a user (admin only)"""
    try:
        if username == current_user['username']:
            raise HTTPException(status_code=400, detail="Cannot delete yourself")
        
        success = auth_manager.delete_user(username)
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": f"User {username} deleted successfully"}
        
    except Exception as e:
        logger.error(f"User deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete user")

@admin_router.get("/config")
async def get_system_config(current_user: dict = Depends(require_admin)):
    """Get system configuration (admin only)"""
    return {
        "embedding_model": "text-embedding-ada-002",
        "vector_index_type": "FAISS",
        "chunk_size": 512,
        "max_results": 100,
        "cache_enabled": True,
        "rate_limiting": rate_limiter.limits
    }

@admin_router.post("/config")
async def update_system_config(
    request: ConfigUpdateRequest,
    current_user: dict = Depends(require_admin)
):
    """Update system configuration (admin only)"""
    # In a real implementation, this would update actual system configuration
    return {
        "message": "Configuration updated successfully",
        "restart_required": request.restart_required
    }

# Document Management Endpoints (placeholder)
@query_router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_write)
):
    """Upload a new document for processing"""
    # Placeholder implementation
    return {
        "message": "Document upload not yet implemented",
        "filename": file.filename
    }

@query_router.get("/documents/search")
async def search_documents_metadata(
    request: DocumentSearchRequest = Depends(),
    current_user: dict = Depends(require_read)
):
    """Search document metadata"""
    # Placeholder implementation
    return DocumentSearchResponse(
        documents=[],
        total_count=0,
        limit=request.limit,
        offset=request.offset
    )

# Error handlers would be defined in the main application
