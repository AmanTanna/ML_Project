"""
Query Engine Core
================

Main query execution engine that orchestrates searches across all RAG layers.
Provides unified interface for complex financial queries.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json

from ..config import RAGConfig
from ..utils import setup_logging
from ..layer1_raw.manager import RawSourceManager
from ..layer2_structured.manager import StructuredDataManager
from ..layer3_semantic.manager import SemanticSearchManager

@dataclass
class QueryRequest:
    """Represents a user query request"""
    query_text: str
    query_type: Optional[str] = "auto"  # auto, semantic, structured, raw, hybrid
    filters: Optional[Dict[str, Any]] = field(default_factory=dict)
    companies: Optional[List[str]] = None
    time_range: Optional[Tuple[str, str]] = None
    sections: Optional[List[str]] = None
    max_results: int = 10
    include_sources: bool = True
    explain_reasoning: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_text': self.query_text,
            'query_type': self.query_type,
            'filters': self.filters,
            'companies': self.companies,
            'time_range': self.time_range,
            'sections': self.sections,
            'max_results': self.max_results,
            'include_sources': self.include_sources,
            'explain_reasoning': self.explain_reasoning
        }

@dataclass
class QueryResult:
    """Represents a unified query result"""
    result_id: str
    content: str
    relevance_score: float
    source_type: str  # semantic, structured, raw
    source_document: Optional[str] = None
    company: Optional[str] = None
    section: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'result_id': self.result_id,
            'content': self.content,
            'relevance_score': self.relevance_score,
            'source_type': self.source_type,
            'source_document': self.source_document,
            'company': self.company,
            'section': self.section,
            'metadata': self.metadata,
            'reasoning': self.reasoning
        }

@dataclass
class QueryResponse:
    """Complete response to a query request"""
    query_id: str
    query_text: str
    results: List[QueryResult]
    total_results: int
    processing_time: float
    query_strategy: str
    layers_used: List[str]
    explanation: Optional[str] = None
    suggestions: Optional[List[str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_id': self.query_id,
            'query_text': self.query_text,
            'results': [r.to_dict() for r in self.results],
            'total_results': self.total_results,
            'processing_time': self.processing_time,
            'query_strategy': self.query_strategy,
            'layers_used': self.layers_used,
            'explanation': self.explanation,
            'suggestions': self.suggestions
        }

class QueryEngine:
    """Main query execution engine for the RAG system"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Initialize layer managers
        self.raw_manager = RawSourceManager(config)
        self.structured_manager = StructuredDataManager(config)
        self.semantic_manager = SemanticSearchManager(config)
        
        # Query statistics
        self.query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_processing_time': 0.0,
            'layer_usage': {
                'semantic': 0,
                'structured': 0,
                'raw': 0,
                'hybrid': 0
            }
        }
        
        # Query cache
        self.query_cache = {}
        self.cache_max_size = config.SEARCH_CACHE_SIZE
        
        self.logger.info("Query Engine initialized")
    
    async def execute_query(self, request: QueryRequest, use_cache: bool = True) -> QueryResponse:
        """Execute a query across appropriate RAG layers"""
        start_time = datetime.now()
        query_id = f"query_{int(start_time.timestamp())}"
        
        try:
            self.logger.info(f"Executing query [{query_id}]: {request.query_text[:100]}...")
            
            # Check cache first
            if use_cache:
                cache_key = self._get_cache_key(request)
                if cache_key in self.query_cache:
                    self.logger.info(f"Cache hit for query [{query_id}]")
                    cached_response = self.query_cache[cache_key]
                    cached_response.query_id = query_id  # Update with new ID
                    return cached_response
            
            # Determine query strategy
            strategy = self._determine_query_strategy(request)
            self.logger.info(f"Query strategy for [{query_id}]: {strategy}")
            
            # Execute based on strategy
            if strategy == "semantic_only":
                results, layers_used = await self._execute_semantic_query(request)
            elif strategy == "structured_only":
                results, layers_used = await self._execute_structured_query(request)
            elif strategy == "raw_only":
                results, layers_used = await self._execute_raw_query(request)
            elif strategy == "hybrid":
                results, layers_used = await self._execute_hybrid_query(request)
            else:  # auto
                results, layers_used = await self._execute_auto_query(request)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create response
            response = QueryResponse(
                query_id=query_id,
                query_text=request.query_text,
                results=results[:request.max_results],
                total_results=len(results),
                processing_time=processing_time,
                query_strategy=strategy,
                layers_used=layers_used,
                explanation=self._generate_explanation(request, strategy, layers_used) if request.explain_reasoning else None,
                suggestions=self._generate_suggestions(request, results)
            )
            
            # Update statistics
            self._update_stats(strategy, processing_time, success=True)
            
            # Cache response
            if use_cache:
                self._cache_response(request, response)
            
            self.logger.info(f"Query [{query_id}] completed: {len(results)} results in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Query [{query_id}] failed: {e}")
            self._update_stats("error", processing_time, success=False)
            
            # Return error response
            return QueryResponse(
                query_id=query_id,
                query_text=request.query_text,
                results=[],
                total_results=0,
                processing_time=processing_time,
                query_strategy="error",
                layers_used=[],
                explanation=f"Query failed: {str(e)}"
            )
    
    def _determine_query_strategy(self, request: QueryRequest) -> str:
        """Determine the best query strategy based on the request"""
        if request.query_type and request.query_type != "auto":
            return request.query_type
        
        query_text = request.query_text.lower()
        
        # Financial metrics/numbers suggest structured query
        financial_keywords = [
            'revenue', 'profit', 'margin', 'ratio', 'earnings', 'ebitda',
            'assets', 'liabilities', 'cash flow', 'debt', 'equity',
            'p/e', 'market cap', 'valuation', 'growth rate'
        ]
        
        # Semantic concepts suggest semantic search  
        semantic_keywords = [
            'risk', 'strategy', 'business model', 'competition', 'market',
            'management', 'outlook', 'challenges', 'opportunities',
            'regulatory', 'technology', 'innovation'
        ]
        
        # Check for financial metrics
        financial_score = sum(1 for keyword in financial_keywords if keyword in query_text)
        semantic_score = sum(1 for keyword in semantic_keywords if keyword in query_text)
        
        # Specific company mentions suggest targeted search
        has_companies = bool(request.companies)
        has_time_range = bool(request.time_range)
        
        # Decision logic
        if financial_score > semantic_score and (has_companies or has_time_range):
            return "structured_only"
        elif semantic_score > financial_score:
            return "semantic_only"
        elif financial_score > 0 and semantic_score > 0:
            return "hybrid"
        else:
            return "semantic_only"  # Default to semantic for general queries
    
    async def _execute_semantic_query(self, request: QueryRequest) -> Tuple[List[QueryResult], List[str]]:
        """Execute semantic search query"""
        try:
            # Build filters
            filters = request.filters.copy() if request.filters else {}
            if request.companies:
                filters['companies'] = request.companies
            if request.sections:
                filters['sections'] = request.sections
            
            # Execute semantic search
            semantic_results = self.semantic_manager.search(
                query=request.query_text,
                k=request.max_results * 2,  # Get more to allow for filtering
                filters=filters
            )
            
            # Convert to QueryResult objects
            results = []
            for i, result in enumerate(semantic_results):
                query_result = QueryResult(
                    result_id=f"sem_{i}_{result.chunk_id}",
                    content=result.chunk_text,
                    relevance_score=result.similarity_score,
                    source_type="semantic",
                    source_document=result.source_document,
                    company=result.ticker,
                    section=result.section,
                    metadata={
                        'chunk_id': result.chunk_id,
                        'chunk_index': result.chunk_index,
                        'form_type': result.form_type,
                        'filing_date': result.filing_date
                    }
                )
                results.append(query_result)
            
            return results, ["semantic"]
            
        except Exception as e:
            self.logger.error(f"Semantic query failed: {e}")
            return [], ["semantic"]
    
    async def _execute_structured_query(self, request: QueryRequest) -> Tuple[List[QueryResult], List[str]]:
        """Execute structured data query"""
        try:
            # This is a simplified implementation - in practice, you'd parse the query
            # to extract specific financial metrics and time periods
            
            # For now, let's do a basic query for financial metrics
            companies = request.companies if request.companies else ["AAPL", "GOOGL", "MSFT"]
            
            results = []
            
            # Query financial metrics from structured layer
            for company in companies[:5]:  # Limit to avoid too many results
                try:
                    # Get company financials (this would be more sophisticated in practice)
                    company_data = self.structured_manager.get_company_profile(company)
                    
                    if company_data:
                        # Create result from company data
                        content = f"{company} Financial Overview:\n"
                        content += f"Market Cap: ${company_data.get('market_cap', 'N/A')}\n"
                        content += f"Revenue: ${company_data.get('revenue', 'N/A')}\n"
                        content += f"Sector: {company_data.get('sector', 'N/A')}\n"
                        
                        query_result = QueryResult(
                            result_id=f"struct_{company}",
                            content=content,
                            relevance_score=0.8,  # Static score for now
                            source_type="structured",
                            company=company,
                            metadata={
                                'data_type': 'company_profile',
                                'source': 'structured_database'
                            }
                        )
                        results.append(query_result)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get structured data for {company}: {e}")
                    continue
            
            return results, ["structured"]
            
        except Exception as e:
            self.logger.error(f"Structured query failed: {e}")
            return [], ["structured"]
    
    async def _execute_raw_query(self, request: QueryRequest) -> Tuple[List[QueryResult], List[str]]:
        """Execute raw document query"""
        try:
            # Get document metadata from raw layer
            storage_stats = self.raw_manager.get_storage_stats()
            
            results = []
            
            # Create a summary result about available documents
            if storage_stats['total_documents'] > 0:
                content = f"Document Repository Summary:\n"
                content += f"Total Documents: {storage_stats['total_documents']}\n"
                content += f"Total Size: {storage_stats.get('total_size_formatted', 'N/A')}\n"
                content += f"Companies: {len(storage_stats.get('companies', []))}\n"
                
                query_result = QueryResult(
                    result_id="raw_summary",
                    content=content,
                    relevance_score=0.7,
                    source_type="raw",
                    metadata={
                        'data_type': 'document_summary',
                        'source': 'raw_storage'
                    }
                )
                results.append(query_result)
            
            return results, ["raw"]
            
        except Exception as e:
            self.logger.error(f"Raw query failed: {e}")
            return [], ["raw"]
    
    async def _execute_hybrid_query(self, request: QueryRequest) -> Tuple[List[QueryResult], List[str]]:
        """Execute hybrid query across multiple layers"""
        try:
            # Execute queries in parallel
            semantic_task = self._execute_semantic_query(request)
            structured_task = self._execute_structured_query(request)
            
            # Wait for both to complete
            semantic_results, semantic_layers = await semantic_task
            structured_results, structured_layers = await structured_task
            
            # Combine and rerank results
            all_results = semantic_results + structured_results
            
            # Simple reranking based on source type and relevance
            for result in all_results:
                if result.source_type == "semantic":
                    result.relevance_score *= 1.0  # No adjustment
                elif result.source_type == "structured":
                    result.relevance_score *= 0.9  # Slight penalty for structured
            
            # Sort by relevance score
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return all_results, ["semantic", "structured"]
            
        except Exception as e:
            self.logger.error(f"Hybrid query failed: {e}")
            return [], ["hybrid"]
    
    async def _execute_auto_query(self, request: QueryRequest) -> Tuple[List[QueryResult], List[str]]:
        """Auto-determine and execute the best query strategy"""
        # Re-determine strategy and execute
        strategy = self._determine_query_strategy(request)
        
        if strategy == "semantic_only":
            return await self._execute_semantic_query(request)
        elif strategy == "structured_only":
            return await self._execute_structured_query(request)
        elif strategy == "hybrid":
            return await self._execute_hybrid_query(request)
        else:
            return await self._execute_semantic_query(request)
    
    def _generate_explanation(self, request: QueryRequest, strategy: str, layers_used: List[str]) -> str:
        """Generate explanation of query processing"""
        explanation = f"Query processed using {strategy} strategy.\n"
        explanation += f"Layers accessed: {', '.join(layers_used)}\n"
        
        if strategy == "semantic_only":
            explanation += "Used semantic search to find contextually relevant content."
        elif strategy == "structured_only":
            explanation += "Used structured database to find specific financial metrics."
        elif strategy == "hybrid":
            explanation += "Combined semantic search with structured data for comprehensive results."
        
        return explanation
    
    def _generate_suggestions(self, request: QueryRequest, results: List[QueryResult]) -> List[str]:
        """Generate query suggestions based on results"""
        suggestions = []
        
        if len(results) == 0:
            suggestions.append("Try broadening your search terms")
            suggestions.append("Check if company tickers are correct")
        elif len(results) < 3:
            suggestions.append("Try using related keywords")
            suggestions.append("Consider expanding the time range")
        
        # Add company-specific suggestions
        companies_found = set(r.company for r in results if r.company)
        if companies_found:
            suggestions.append(f"Also search for: {', '.join(list(companies_found)[:3])}")
        
        return suggestions[:3]  # Limit suggestions
    
    def _get_cache_key(self, request: QueryRequest) -> str:
        """Generate cache key for request"""
        import hashlib
        
        cache_data = {
            'query_text': request.query_text,
            'query_type': request.query_type,
            'companies': sorted(request.companies) if request.companies else None,
            'max_results': request.max_results
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _cache_response(self, request: QueryRequest, response: QueryResponse):
        """Cache query response"""
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        cache_key = self._get_cache_key(request)
        self.query_cache[cache_key] = response
    
    def _update_stats(self, strategy: str, processing_time: float, success: bool):
        """Update query statistics"""
        self.query_stats['total_queries'] += 1
        
        if success:
            self.query_stats['successful_queries'] += 1
            # Update average processing time
            total_successful = self.query_stats['successful_queries']
            current_avg = self.query_stats['avg_processing_time']
            self.query_stats['avg_processing_time'] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
            
            # Update layer usage
            if strategy in self.query_stats['layer_usage']:
                self.query_stats['layer_usage'][strategy] += 1
        else:
            self.query_stats['failed_queries'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get query engine statistics"""
        return {
            'query_stats': self.query_stats.copy(),
            'cache_size': len(self.query_cache),
            'layers_available': {
                'semantic': self.semantic_manager is not None,
                'structured': self.structured_manager is not None,
                'raw': self.raw_manager is not None
            }
        }
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        self.logger.info("Query cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of query engine"""
        health = {
            'status': 'healthy',
            'issues': [],
            'components': {}
        }
        
        # Check layer managers
        try:
            semantic_health = self.semantic_manager.health_check()
            health['components']['semantic'] = semantic_health['status']
            if semantic_health['issues']:
                health['issues'].extend(f"Semantic: {issue}" for issue in semantic_health['issues'])
        except Exception as e:
            health['components']['semantic'] = 'error'
            health['issues'].append(f"Semantic layer error: {e}")
        
        try:
            structured_health = self.structured_manager.health_check()
            health['components']['structured'] = structured_health['status']
            if structured_health['issues']:
                health['issues'].extend(f"Structured: {issue}" for issue in structured_health['issues'])
        except Exception as e:
            health['components']['structured'] = 'error'
            health['issues'].append(f"Structured layer error: {e}")
        
        # Overall status
        if health['issues']:
            health['status'] = 'degraded' if len(health['issues']) < 3 else 'unhealthy'
        
        return health
