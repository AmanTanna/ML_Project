"""
API Models
==========

Pydantic models for API requests and responses.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum
import json

class BaseResponseModel(BaseModel):
    """Base model with proper datetime serialization"""
    
    def dict(self, **kwargs):
        """Override dict method to properly serialize datetime objects"""
        data = super().dict(**kwargs)
        # Convert datetime objects to ISO format strings
        def convert_datetime(obj):
            if isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        return convert_datetime(data)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class QueryType(str, Enum):
    """Query types supported by the system"""
    AUTO = "auto"
    SEMANTIC = "semantic"
    STRUCTURED = "structured"
    RAW = "raw"
    HYBRID = "hybrid"

class QueryRequest(BaseModel):
    """Request model for document queries"""
    query_text: str = Field(..., description="The natural language query")
    query_type: QueryType = Field(QueryType.AUTO, description="Type of query to execute")
    companies: Optional[List[str]] = Field(None, description="List of company tickers to focus on")
    time_range: Optional[Tuple[str, str]] = Field(None, description="Date range (start, end) in YYYY-MM-DD format")
    sections: Optional[List[str]] = Field(None, description="Document sections to search")
    max_results: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    include_sources: bool = Field(True, description="Whether to include source information")
    explain_reasoning: bool = Field(False, description="Whether to include reasoning explanations")
    use_cache: bool = Field(True, description="Whether to use query caching")
    
    @validator('time_range')
    def validate_time_range(cls, v):
        if v is not None:
            start, end = v
            try:
                datetime.strptime(start, '%Y-%m-%d')
                datetime.strptime(end, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Date format must be YYYY-MM-DD")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_text": "What are Apple's main revenue streams?",
                "query_type": "semantic",
                "companies": ["AAPL"],
                "max_results": 5,
                "include_sources": True,
                "explain_reasoning": True
            }
        }

class QueryResult(BaseModel):
    """Individual query result"""
    result_id: str
    content: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    source_type: str
    source_document: Optional[str] = None
    company: Optional[str] = None
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None

class QueryResponse(BaseResponseModel):
    """Response model for document queries"""
    query_id: str
    query_text: str
    results: List[QueryResult]
    total_results: int
    processing_time: float = Field(..., ge=0.0)
    query_strategy: str
    layers_used: List[str]
    explanation: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @classmethod
    def from_query_response(cls, response):
        """Convert from internal QueryResponse to API model"""
        return cls(
            query_id=response.query_id,
            query_text=response.query_text,
            results=[
                QueryResult(
                    result_id=r.result_id,
                    content=r.content,
                    relevance_score=r.relevance_score,
                    source_type=r.source_type,
                    source_document=r.source_document,
                    company=r.company,
                    section=r.section,
                    metadata=r.metadata or {},
                    reasoning=r.reasoning
                ) for r in response.results
            ],
            total_results=response.total_results,
            processing_time=response.processing_time,
            query_strategy=response.query_strategy,
            layers_used=response.layers_used,
            explanation=response.explanation,
            suggestions=response.suggestions or []
        )

class HealthStatus(str, Enum):
    """System health statuses"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    ERROR = "error"

class ComponentHealth(BaseModel):
    """Health status of a system component"""
    status: HealthStatus
    details: Optional[str] = None
    last_check: Optional[datetime] = None
    issues: List[str] = Field(default_factory=list)

class HealthResponse(BaseModel):
    """System health check response"""
    status: HealthStatus
    message: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    components: Dict[str, ComponentHealth] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)
    uptime: Optional[float] = None

class StatsOverview(BaseModel):
    """Overview statistics"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0

class QueryDistribution(BaseModel):
    """Query distribution statistics"""
    query_types: Dict[str, int] = Field(default_factory=dict)
    layer_usage: Dict[str, int] = Field(default_factory=dict)

class ComponentStats(BaseModel):
    """Component-specific statistics"""
    query_engine: Dict[str, Any] = Field(default_factory=dict)
    query_router: Dict[str, Any] = Field(default_factory=dict)
    results_fusion: Dict[str, Any] = Field(default_factory=dict)

class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    status: str
    alerts: List[str] = Field(default_factory=list)
    thresholds: Dict[str, float] = Field(default_factory=dict)

class RecentQuery(BaseModel):
    """Recent query information"""
    timestamp: str
    query_id: str
    query_text: str
    processing_time: float
    result_count: int
    success: bool

class StatsResponse(BaseResponseModel):
    """System statistics response"""
    overview: StatsOverview
    query_distribution: QueryDistribution
    components: ComponentStats
    performance: PerformanceMetrics
    recent_queries: List[RecentQuery] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

class DocumentUploadRequest(BaseModel):
    """Document upload request"""
    filename: str
    content_type: str
    company: Optional[str] = None
    document_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentUploadResponse(BaseResponseModel):
    """Document upload response"""
    document_id: str
    filename: str
    size: int
    status: str
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class UserProfile(BaseResponseModel):
    """User profile model"""
    username: str
    email: Optional[str] = None
    role: str = "user"
    permissions: List[str] = Field(default_factory=list)
    created_at: datetime
    last_login: Optional[datetime] = None

class LoginRequest(BaseModel):
    """Login request model"""
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)

class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserProfile

class ErrorResponse(BaseResponseModel):
    """Error response model"""
    error: str
    status_code: int
    path: str
    timestamp: datetime = Field(default_factory=datetime.now)
    details: Optional[str] = None
    request_id: Optional[str] = None

class OpenAIRequest(BaseModel):
    """OpenAI API request model"""
    messages: List[Dict[str, str]]
    model: str = "gpt-3.5-turbo"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4000)
    stream: bool = False

class OpenAIResponse(BaseResponseModel):
    """OpenAI API response model"""
    response: str
    model: str
    usage: Dict[str, int]
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class RAGChatRequest(BaseModel):
    """RAG-enhanced chat request"""
    query: str = Field(..., min_length=1)
    model: str = "gpt-3.5-turbo"
    max_context_results: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    include_sources: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the main risks facing Apple in 2024?",
                "model": "gpt-3.5-turbo",
                "max_context_results": 5,
                "temperature": 0.7,
                "include_sources": True
            }
        }

class RAGChatResponse(BaseResponseModel):
    """RAG-enhanced chat response"""
    response: str
    sources: List[QueryResult] = Field(default_factory=list)
    rag_context: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float
    model: str
    timestamp: datetime = Field(default_factory=datetime.now)

class SystemConfig(BaseModel):
    """System configuration model"""
    embedding_model: str
    vector_index_type: str
    chunk_size: int
    max_results: int
    cache_enabled: bool
    rate_limiting: Dict[str, int]

class ConfigUpdateRequest(BaseModel):
    """Configuration update request"""
    config: Dict[str, Any]
    restart_required: bool = False

class BatchQueryRequest(BaseModel):
    """Batch query request"""
    queries: List[QueryRequest] = Field(..., min_items=1, max_items=10)
    parallel_execution: bool = True
    
class BatchQueryResponse(BaseModel):
    """Batch query response"""
    results: List[QueryResponse]
    total_queries: int
    successful_queries: int
    failed_queries: int
    total_processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class DocumentSearchRequest(BaseModel):
    """Document search request"""
    query: Optional[str] = None
    company: Optional[str] = None
    document_type: Optional[str] = None
    date_range: Optional[Tuple[str, str]] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)

class DocumentInfo(BaseModel):
    """Document information"""
    document_id: str
    filename: str
    company: Optional[str] = None
    document_type: Optional[str] = None
    size: int
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentSearchResponse(BaseModel):
    """Document search response"""
    documents: List[DocumentInfo]
    total_count: int
    limit: int
    offset: int
    timestamp: datetime = Field(default_factory=datetime.now)
