"""
Query Engine Manager
===================

High-level orchestrator for the multi-layer query engine system.
Provides the main interface for executing complex financial queries.
"""

import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json

from ..config import RAGConfig
from ..utils import setup_logging
from .query_engine import QueryEngine, QueryRequest, QueryResponse
from .query_router import QueryRouter, QueryClassification, RoutingDecision
from .results_fusion import ResultsFusion

@dataclass
class QueryEngineStats:
    """Statistics for the query engine system"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_response_time: float = 0.0
    query_types: Dict[str, int] = None
    layer_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.query_types is None:
            self.query_types = {}
        if self.layer_usage is None:
            self.layer_usage = {}

class QueryEngineManager:
    """
    Main orchestrator for the multi-layer query engine system.
    Provides unified interface for complex financial document queries.
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Initialize core components
        self.query_engine = QueryEngine(config)
        self.query_router = QueryRouter(config)
        self.results_fusion = ResultsFusion(config)
        
        # System statistics
        self.stats = QueryEngineStats()
        
        # Query history (for debugging and optimization)
        self.query_history = []
        self.max_history_size = 1000
        
        # Performance monitoring
        self.performance_thresholds = {
            'max_response_time': 30.0,  # seconds
            'min_success_rate': 0.95,
            'max_error_rate': 0.05
        }
        
        self.logger.info("Query Engine Manager initialized successfully")
    
    async def process_query(self, 
                          query_text: str,
                          query_type: Optional[str] = "auto",
                          companies: Optional[List[str]] = None,
                          time_range: Optional[tuple] = None,
                          sections: Optional[List[str]] = None,
                          max_results: int = 10,
                          include_sources: bool = True,
                          explain_reasoning: bool = False,
                          use_cache: bool = True) -> QueryResponse:
        """
        Process a complex financial query through the multi-layer RAG system
        
        Args:
            query_text: The natural language query
            query_type: Query type hint ("auto", "semantic", "structured", "raw", "hybrid")
            companies: List of company tickers to focus on
            time_range: Tuple of (start_date, end_date) strings
            sections: List of document sections to search
            max_results: Maximum number of results to return
            include_sources: Whether to include source information
            explain_reasoning: Whether to include reasoning explanations
            use_cache: Whether to use query caching
            
        Returns:
            QueryResponse: Complete response with results and metadata
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing query: {query_text[:100]}...")
            
            # Create query request
            request = QueryRequest(
                query_text=query_text,
                query_type=query_type,
                companies=companies,
                time_range=time_range,
                sections=sections,
                max_results=max_results,
                include_sources=include_sources,
                explain_reasoning=explain_reasoning
            )
            
            # Log query details
            self._log_query_start(request)
            
            # Execute query through engine
            response = await self.query_engine.execute_query(request, use_cache=use_cache)
            
            # Process and enhance response
            enhanced_response = await self._enhance_response(response, request)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(request, enhanced_response, processing_time, success=True)
            
            # Add to query history
            self._add_to_history(request, enhanced_response, processing_time)
            
            self.logger.info(f"Query processed successfully: {len(enhanced_response.results)} results in {processing_time:.3f}s")
            return enhanced_response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Query processing failed: {e}")
            
            # Create error response
            error_response = QueryResponse(
                query_id=f"error_{int(start_time.timestamp())}",
                query_text=query_text,
                results=[],
                total_results=0,
                processing_time=processing_time,
                query_strategy="error",
                layers_used=[],
                explanation=f"Query processing failed: {str(e)}"
            )
            
            # Update error statistics
            self._update_stats(request if 'request' in locals() else None, error_response, processing_time, success=False)
            
            return error_response
    
    async def process_batch_queries(self, 
                                  queries: List[Dict[str, Any]], 
                                  max_concurrent: int = 5) -> List[QueryResponse]:
        """
        Process multiple queries in batch with concurrency control
        
        Args:
            queries: List of query dictionaries with query parameters
            max_concurrent: Maximum number of concurrent queries
            
        Returns:
            List of QueryResponse objects
        """
        self.logger.info(f"Processing batch of {len(queries)} queries")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_query(query_params: Dict[str, Any]) -> QueryResponse:
            async with semaphore:
                return await self.process_query(**query_params)
        
        # Process all queries concurrently
        tasks = [process_single_query(query) for query in queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Batch query {i} failed: {response}")
                # Create error response
                error_response = QueryResponse(
                    query_id=f"batch_error_{i}",
                    query_text=queries[i].get('query_text', 'Unknown'),
                    results=[],
                    total_results=0,
                    processing_time=0.0,
                    query_strategy="error",
                    layers_used=[],
                    explanation=f"Batch processing error: {str(response)}"
                )
                valid_responses.append(error_response)
            else:
                valid_responses.append(response)
        
        self.logger.info(f"Batch processing completed: {len(valid_responses)} responses")
        return valid_responses
    
    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        """
        Analyze a query without executing it - useful for debugging and optimization
        
        Args:
            query_text: The query to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Classify the query
            classification = self.query_router.classify_query(query_text)
            
            # Get routing decision
            routing = self.query_router.route_query(query_text, classification)
            
            # Analyze query complexity
            complexity = self._analyze_query_complexity(query_text)
            
            return {
                'query_text': query_text,
                'classification': {
                    'primary_type': classification.primary_type.value,
                    'secondary_types': [t.value for t in classification.secondary_types],
                    'confidence': classification.confidence,
                    'keywords_found': classification.keywords_found,
                    'entities_found': classification.entities_found,
                    'reasoning': classification.reasoning
                },
                'routing': {
                    'primary_layer': routing.primary_layer.value,
                    'secondary_layers': [l.value for l in routing.secondary_layers],
                    'layer_weights': {k.value: v for k, v in routing.layer_weights.items()},
                    'strategy': routing.routing_strategy,
                    'confidence': routing.confidence,
                    'reasoning': routing.reasoning
                },
                'complexity': complexity,
                'estimated_performance': self._estimate_query_performance(classification, routing, complexity)
            }
            
        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}")
            return {
                'query_text': query_text,
                'error': str(e),
                'classification': None,
                'routing': None
            }
    
    async def _enhance_response(self, response: QueryResponse, request: QueryRequest) -> QueryResponse:
        """Enhance query response with additional information"""
        
        # Add query analysis if requested
        if request.explain_reasoning:
            if not response.explanation:
                response.explanation = "Query processed through multi-layer RAG system"
            
            # Add detailed layer information
            layer_info = f"\nLayers used: {', '.join(response.layers_used)}"
            layer_info += f"\nStrategy: {response.query_strategy}"
            layer_info += f"\nProcessing time: {response.processing_time:.3f}s"
            
            response.explanation += layer_info
        
        # Enhance suggestions based on results
        if len(response.results) < request.max_results // 2:
            if not response.suggestions:
                response.suggestions = []
            
            response.suggestions.extend([
                "Try using more general search terms",
                "Consider expanding the time range",
                "Check if company tickers are spelled correctly"
            ])
        
        return response
    
    def _analyze_query_complexity(self, query_text: str) -> Dict[str, Any]:
        """Analyze query complexity"""
        complexity = {
            'length': len(query_text),
            'word_count': len(query_text.split()),
            'has_operators': any(op in query_text.lower() for op in ['and', 'or', 'not', '(', ')']),
            'has_dates': bool(re.search(r'\b(20\d{2}|19\d{2}|Q[1-4])\b', query_text)),
            'has_numbers': bool(re.search(r'\$?[\d,]+(?:\.\d+)?[BMK%]?', query_text)),
            'has_companies': bool(re.search(r'\b[A-Z]{2,5}\b', query_text)),
            'complexity_score': 0
        }
        
        # Calculate overall complexity score
        score = 0
        if complexity['word_count'] > 10:
            score += 1
        if complexity['has_operators']:
            score += 2
        if complexity['has_dates']:
            score += 1
        if complexity['has_numbers']:
            score += 1
        if complexity['has_companies']:
            score += 1
        
        complexity['complexity_score'] = score
        complexity['complexity_level'] = (
            'simple' if score <= 2 else 
            'moderate' if score <= 4 else 
            'complex'
        )
        
        return complexity
    
    def _estimate_query_performance(self, classification: QueryClassification, 
                                  routing: RoutingDecision, 
                                  complexity: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate query performance characteristics"""
        
        # Base estimates
        estimated_time = 2.0  # seconds
        estimated_results = 10
        confidence = 0.8
        
        # Adjust based on routing
        if routing.primary_layer.value == 'semantic':
            estimated_time += 1.0  # Semantic search is slower
            estimated_results += 5
        elif routing.primary_layer.value == 'structured':
            estimated_time += 0.5
            estimated_results += 3
        
        # Adjust based on complexity
        complexity_multiplier = 1.0 + (complexity['complexity_score'] * 0.2)
        estimated_time *= complexity_multiplier
        
        # Adjust based on classification confidence
        confidence *= classification.confidence
        
        return {
            'estimated_response_time': estimated_time,
            'estimated_result_count': estimated_results,
            'confidence': confidence,
            'performance_category': (
                'fast' if estimated_time <= 3.0 else
                'moderate' if estimated_time <= 8.0 else
                'slow'
            )
        }
    
    def _log_query_start(self, request: QueryRequest):
        """Log query start details"""
        self.logger.debug(f"Query details: {request.to_dict()}")
    
    def _update_stats(self, request: Optional[QueryRequest], response: QueryResponse, 
                     processing_time: float, success: bool):
        """Update system statistics"""
        self.stats.total_queries += 1
        
        if success:
            self.stats.successful_queries += 1
            
            # Update average response time
            if self.stats.successful_queries == 1:
                self.stats.avg_response_time = processing_time
            else:
                self.stats.avg_response_time = (
                    (self.stats.avg_response_time * (self.stats.successful_queries - 1) + processing_time) 
                    / self.stats.successful_queries
                )
            
            # Update query type statistics
            if request and request.query_type:
                if request.query_type not in self.stats.query_types:
                    self.stats.query_types[request.query_type] = 0
                self.stats.query_types[request.query_type] += 1
            
            # Update layer usage statistics
            for layer in response.layers_used:
                if layer not in self.stats.layer_usage:
                    self.stats.layer_usage[layer] = 0
                self.stats.layer_usage[layer] += 1
        else:
            self.stats.failed_queries += 1
    
    def _add_to_history(self, request: QueryRequest, response: QueryResponse, processing_time: float):
        """Add query to history for analysis"""
        if len(self.query_history) >= self.max_history_size:
            self.query_history.pop(0)  # Remove oldest
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'query_id': response.query_id,
            'query_text': request.query_text,
            'query_type': request.query_type,
            'processing_time': processing_time,
            'result_count': len(response.results),
            'layers_used': response.layers_used,
            'strategy': response.query_strategy,
            'success': len(response.results) > 0
        }
        
        self.query_history.append(history_entry)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        # Calculate success rate
        success_rate = (
            self.stats.successful_queries / self.stats.total_queries 
            if self.stats.total_queries > 0 else 0.0
        )
        
        # Get component stats
        engine_stats = self.query_engine.get_stats()
        router_stats = self.query_router.get_routing_stats()
        fusion_stats = self.results_fusion.get_fusion_stats()
        
        return {
            'overview': {
                'total_queries': self.stats.total_queries,
                'successful_queries': self.stats.successful_queries,
                'failed_queries': self.stats.failed_queries,
                'success_rate': success_rate,
                'avg_response_time': self.stats.avg_response_time
            },
            'query_distribution': {
                'query_types': self.stats.query_types,
                'layer_usage': self.stats.layer_usage
            },
            'components': {
                'query_engine': engine_stats,
                'query_router': router_stats,
                'results_fusion': fusion_stats
            },
            'performance': self._get_performance_metrics(),
            'recent_queries': self.query_history[-10:] if self.query_history else []
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and alerts"""
        metrics = {
            'status': 'healthy',
            'alerts': [],
            'thresholds': self.performance_thresholds
        }
        
        # Check response time
        if self.stats.avg_response_time > self.performance_thresholds['max_response_time']:
            metrics['status'] = 'degraded'
            metrics['alerts'].append(f"Average response time ({self.stats.avg_response_time:.2f}s) exceeds threshold")
        
        # Check success rate
        if self.stats.total_queries > 0:
            success_rate = self.stats.successful_queries / self.stats.total_queries
            if success_rate < self.performance_thresholds['min_success_rate']:
                metrics['status'] = 'degraded'
                metrics['alerts'].append(f"Success rate ({success_rate:.2f}) below threshold")
        
        return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'issues': [],
            'components': {}
        }
        
        # Check core components
        try:
            engine_health = self.query_engine.health_check()
            health['components']['query_engine'] = engine_health
            if engine_health['issues']:
                health['issues'].extend(f"Engine: {issue}" for issue in engine_health['issues'])
        except Exception as e:
            health['components']['query_engine'] = {'status': 'error', 'error': str(e)}
            health['issues'].append(f"Engine health check failed: {e}")
        
        try:
            router_health = self.query_router.health_check()
            health['components']['query_router'] = router_health
            if router_health['issues']:
                health['issues'].extend(f"Router: {issue}" for issue in router_health['issues'])
        except Exception as e:
            health['components']['query_router'] = {'status': 'error', 'error': str(e)}
            health['issues'].append(f"Router health check failed: {e}")
        
        try:
            fusion_health = self.results_fusion.health_check()
            health['components']['results_fusion'] = fusion_health
            if fusion_health['issues']:
                health['issues'].extend(f"Fusion: {issue}" for issue in fusion_health['issues'])
        except Exception as e:
            health['components']['results_fusion'] = {'status': 'error', 'error': str(e)}
            health['issues'].append(f"Fusion health check failed: {e}")
        
        # Overall health status
        if health['issues']:
            health['status'] = 'degraded' if len(health['issues']) < 5 else 'unhealthy'
        
        return health
    
    def optimize_performance(self):
        """Optimize system performance based on usage patterns"""
        try:
            self.logger.info("Starting performance optimization")
            
            # Analyze query history for patterns
            if len(self.query_history) < 10:
                self.logger.info("Insufficient query history for optimization")
                return
            
            # Get recent performance data
            recent_queries = self.query_history[-100:]
            
            # Analyze slow queries
            slow_queries = [q for q in recent_queries if q['processing_time'] > 10.0]
            if slow_queries:
                self.logger.info(f"Found {len(slow_queries)} slow queries - analyzing patterns")
                # In practice, you could optimize based on slow query patterns
            
            # Analyze popular query types
            query_type_counts = {}
            for query in recent_queries:
                qtype = query.get('query_type', 'unknown')
                query_type_counts[qtype] = query_type_counts.get(qtype, 0) + 1
            
            most_popular = max(query_type_counts.items(), key=lambda x: x[1]) if query_type_counts else None
            if most_popular:
                self.logger.info(f"Most popular query type: {most_popular[0]} ({most_popular[1]} queries)")
            
            # Clear old cache entries if needed
            self.query_engine.clear_cache()
            
            self.logger.info("Performance optimization completed")
            
        except Exception as e:
            self.logger.error(f"Performance optimization failed: {e}")
    
    def export_query_history(self, filepath: Optional[str] = None) -> str:
        """Export query history to JSON file"""
        if filepath is None:
            filepath = f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'export_timestamp': datetime.now().isoformat(),
                    'total_queries': len(self.query_history),
                    'statistics': self.stats.__dict__,
                    'queries': self.query_history
                }, f, indent=2)
            
            self.logger.info(f"Query history exported to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to export query history: {e}")
            raise
    
    def reset_statistics(self):
        """Reset all statistics (useful for testing)"""
        self.stats = QueryEngineStats()
        self.query_history.clear()
        self.logger.info("Statistics reset")
    
    def shutdown(self):
        """Graceful shutdown of the query engine system"""
        try:
            self.logger.info("Shutting down Query Engine Manager")
            
            # Export final statistics
            final_stats = self.get_system_stats()
            stats_file = f"final_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(stats_file, 'w') as f:
                json.dump(final_stats, f, indent=2)
            
            self.logger.info(f"Final statistics saved to {stats_file}")
            self.logger.info("Query Engine Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Import for convenience
import re
