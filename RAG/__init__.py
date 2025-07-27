"""
RAG System for S&P 500 Annual Reports Analysis
==============================================

A three-layer retrieval-augmented generation system for analyzing ~2,500 annual reports
(S&P 500 × 5 years) with fast semantic search, rich metadata filtering, and provenance tracking.

Architecture:
- Layer 1: Raw Source (PDFs, HTML, text dumps)
- Layer 2: Structured & Numeric (Parquet + DuckDB)  
- Layer 3: Semantic Retrieval (FAISS + metadata)
"""

__version__ = "1.0.0"
__author__ = "ML_Project Team"

from .config import RAGConfig
from .utils import setup_logging, get_file_hash, save_jsonl, load_jsonl, progress_bar
from .storage import StorageManager

# Layer imports (placeholder implementations)
from .layer1_raw import RawSourceManager
from .layer2_structured import StructuredDataManager  
from .layer3_semantic import SemanticSearchManager

__all__ = [
    'RAGConfig',
    'StorageManager', 
    'RawSourceManager',
    'StructuredDataManager',
    'SemanticSearchManager',
    'setup_logging',
    'get_file_hash',
    'save_jsonl', 
    'load_jsonl',
    'progress_bar'
]
