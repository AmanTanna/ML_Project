"""
RAG System Configuration
========================

Central configuration for the three-layer RAG system.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

@dataclass
class RAGConfig:
    """Configuration class for the RAG system"""
    
    # Base directories
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Layer 1: Raw Source Storage
    RAW_PDFS_DIR: Path = DATA_DIR / "raw_pdfs"
    MANIFEST_FILE: Path = DATA_DIR / "manifest.jsonl"
    FILING_REGISTRY_FILE: Path = DATA_DIR / "filing_registry.json"
    
    # Layer 2: Structured & Numeric Storage  
    PARQUET_DIR: Path = DATA_DIR / "parquet_store"
    DUCKDB_FILE: Path = DATA_DIR / "financial_data.duckdb"
    
    # Layer 3: Semantic Retrieval Storage
    VECTOR_INDEX_DIR: Path = DATA_DIR / "vector_index"
    CHUNKS_FILE: Path = DATA_DIR / "chunks.jsonl"
    METADATA_DB: Path = DATA_DIR / "metadata.db"
    
    # SEC EDGAR API Configuration
    SEC_API_BASE_URL: str = "https://www.sec.gov/Archives/edgar/data"
    USER_AGENT: str = "ML_Project RAG System 1.0 (contact@example.com)"
    REQUEST_DELAY: float = 0.1  # Seconds between requests
    
    # Text Processing Configuration
    CHUNK_SIZE: int = 500  # Target tokens per chunk
    CHUNK_OVERLAP: int = 50  # Overlap tokens between chunks
    MAX_CHUNK_SIZE: int = 800  # Maximum tokens per chunk
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    VECTOR_INDEX_TYPE: str = "IVF4096,Flat"  # FAISS index type
    
    # S&P 500 Configuration
    SP500_YEARS: List[int] = field(default_factory=lambda: [2020, 2021, 2022, 2023, 2024])
    FILING_TYPES: List[str] = field(default_factory=lambda: ["10-K", "10-Q"])  # Focus on annual and quarterly reports
    
    # Performance Configuration
    BATCH_SIZE: int = 32
    MAX_WORKERS: int = 4
    CACHE_SIZE: int = 1000
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = DATA_DIR / "rag_system.log"
    
    def __post_init__(self):
        """Ensure all directories exist"""
        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, Path) and attr_name.endswith('_DIR'):
                attr_value.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls, env_prefix: str = "RAG_") -> "RAGConfig":
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        for attr_name in dir(config):
            if not attr_name.startswith('_'):
                env_var = f"{env_prefix}{attr_name}"
                env_value = os.getenv(env_var)
                if env_value is not None:
                    attr_type = type(getattr(config, attr_name))
                    if attr_type == Path:
                        setattr(config, attr_name, Path(env_value))
                    elif attr_type == bool:
                        setattr(config, attr_name, env_value.lower() in ('true', '1', 'yes'))
                    elif attr_type == int:
                        setattr(config, attr_name, int(env_value))
                    elif attr_type == float:
                        setattr(config, attr_name, float(env_value))
                    elif attr_type == list:
                        setattr(config, attr_name, env_value.split(','))
                    else:
                        setattr(config, attr_name, env_value)
        
        return config
    
    def get_ticker_path(self, ticker: str, year: int, filing_type: str = "10-K") -> Path:
        """Get the file path for a specific ticker/year combination"""
        return self.RAW_PDFS_DIR / ticker / str(year) / f"{filing_type}.pdf"
    
    def get_parquet_path(self, table_name: str) -> Path:
        """Get the parquet file path for a specific table"""
        return self.PARQUET_DIR / f"{table_name}.parquet"

# Global configuration instance
config = RAGConfig()
