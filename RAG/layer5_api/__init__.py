"""
API Layer Init
==============

Part 6 of the RAG system - REST API layer with FastAPI.
Provides HTTP endpoints for the multi-layer query engine.
"""

# Minimal imports to avoid circular dependencies
# Components can be imported directly when needed

__all__ = [
    'APIServer',
    'app',
    'setup_middleware',
    'setup_routes',
    'rate_limiter',
    'AuthManager',
    'OpenAIService'
]
