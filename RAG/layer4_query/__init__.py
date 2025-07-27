"""
Multi-Layer Query Engine
========================

Part 5 of the RAG system - Unified query interface that intelligently
routes queries across all layers and provides ranked, fused results.
"""

from .manager import QueryEngineManager
from .query_router import QueryRouter, QueryType, LayerType, QueryClassification, RoutingDecision
from .results_fusion import ResultsFusion, FusionConfig, ResultMetrics
from .query_engine import QueryEngine, QueryRequest, QueryResponse, QueryResult

__all__ = [
    'QueryEngineManager',
    'QueryRouter', 
    'QueryType',
    'LayerType', 
    'QueryClassification',
    'RoutingDecision',
    'ResultsFusion',
    'FusionConfig',
    'ResultMetrics',
    'QueryEngine',
    'QueryRequest',
    'QueryResponse', 
    'QueryResult'
]
