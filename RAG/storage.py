"""
Storage Manager for the Three-Layer RAG System
==============================================

Manages the coordination between all three storage layers:
- Layer 1: Raw source files (PDFs, HTML, text)
- Layer 2: Structured numeric data (Parquet + DuckDB)
- Layer 3: Semantic retrieval (FAISS + metadata)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

from .config import RAGConfig
from .utils import setup_logging, generate_uuid, get_file_hash

@dataclass
class StorageStats:
    """Statistics about the storage system"""
    raw_files_count: int = 0
    raw_files_size: int = 0
    parquet_tables_count: int = 0  
    parquet_size: int = 0
    vector_chunks_count: int = 0
    vector_index_size: int = 0
    last_updated: Optional[str] = None

class StorageManager:
    """Central manager for all three storage layers"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.logger = setup_logging(
            log_level=self.config.LOG_LEVEL,
            log_file=self.config.LOG_FILE
        )
        
        # Lazy loading of layer managers
        self._layer1_manager = None
        self._layer2_manager = None  
        self._layer3_manager = None
        
        self.logger.info("StorageManager initialized")
    
    @property
    def layer1(self):
        """Get Layer 1 (Raw Source) manager"""
        if self._layer1_manager is None:
            from .layer1_raw.manager import RawSourceManager
            self._layer1_manager = RawSourceManager(self.config)
        return self._layer1_manager
    
    @property
    def layer2(self):
        """Get Layer 2 (Structured) manager"""
        if self._layer2_manager is None:
            from .layer2_structured.manager import StructuredDataManager
            self._layer2_manager = StructuredDataManager(self.config)
        return self._layer2_manager
    
    @property
    def layer3(self):
        """Get Layer 3 (Semantic) manager"""
        if self._layer3_manager is None:
            from .layer3_semantic.manager import SemanticRetrievalManager
            self._layer3_manager = SemanticRetrievalManager(self.config)
        return self._layer3_manager
    
    def initialize_storage(self) -> bool:
        """Initialize all storage layers"""
        try:
            self.logger.info("Initializing storage layers...")
            
            # Ensure all directories exist
            self.config.RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
            self.config.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
            self.config.VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            
            # Initialize each layer
            self.layer1.initialize()
            self.layer2.initialize()
            self.layer3.initialize()
            
            self.logger.info("All storage layers initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage: {e}")
            return False
    
    def get_storage_stats(self) -> StorageStats:
        """Get comprehensive storage statistics"""
        stats = StorageStats()
        
        try:
            # Layer 1 stats
            if self.config.RAW_PDFS_DIR.exists():
                raw_files = list(self.config.RAW_PDFS_DIR.rglob("*.pdf"))
                stats.raw_files_count = len(raw_files)
                stats.raw_files_size = sum(f.stat().st_size for f in raw_files if f.exists())
            
            # Layer 2 stats
            if self.config.PARQUET_DIR.exists():
                parquet_files = list(self.config.PARQUET_DIR.glob("*.parquet"))
                stats.parquet_tables_count = len(parquet_files)
                stats.parquet_size = sum(f.stat().st_size for f in parquet_files if f.exists())
            
            # Layer 3 stats
            if self.config.CHUNKS_FILE.exists():
                # Estimate chunk count from file size (rough approximation)
                file_size = self.config.CHUNKS_FILE.stat().st_size
                stats.vector_chunks_count = max(0, file_size // 500)  # Rough estimate
            
            if self.config.VECTOR_INDEX_DIR.exists():
                index_files = list(self.config.VECTOR_INDEX_DIR.glob("*"))
                stats.vector_index_size = sum(f.stat().st_size for f in index_files if f.is_file())
            
            stats.last_updated = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"Error calculating storage stats: {e}")
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all storage layers"""
        health_report = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "layers": {}
        }
        
        try:
            # Check Layer 1
            layer1_healthy = (
                self.config.RAW_PDFS_DIR.exists() and
                self.config.MANIFEST_FILE.parent.exists()
            )
            health_report["layers"]["layer1_raw"] = {
                "status": "healthy" if layer1_healthy else "warning",
                "details": "Raw storage directories accessible"
            }
            
            # Check Layer 2
            layer2_healthy = (
                self.config.PARQUET_DIR.exists() and
                self.config.DUCKDB_FILE.parent.exists()
            )
            health_report["layers"]["layer2_structured"] = {
                "status": "healthy" if layer2_healthy else "warning", 
                "details": "Structured data storage accessible"
            }
            
            # Check Layer 3
            layer3_healthy = (
                self.config.VECTOR_INDEX_DIR.exists() and
                self.config.CHUNKS_FILE.parent.exists()
            )
            health_report["layers"]["layer3_semantic"] = {
                "status": "healthy" if layer3_healthy else "warning",
                "details": "Semantic retrieval storage accessible"
            }
            
            # Overall status
            if not all([layer1_healthy, layer2_healthy, layer3_healthy]):
                health_report["overall_status"] = "warning"
            
        except Exception as e:
            health_report["overall_status"] = "error"
            health_report["error"] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return health_report
    
    def cleanup_storage(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up orphaned or temporary files"""
        cleanup_report = {
            "dry_run": dry_run,
            "files_to_remove": [],
            "space_to_free": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Find temporary files
            temp_patterns = ["*.tmp", "*.temp", "*.lock", "*~", ".DS_Store"]
            
            for pattern in temp_patterns:
                for temp_file in self.config.DATA_DIR.rglob(pattern):
                    if temp_file.is_file():
                        cleanup_report["files_to_remove"].append(str(temp_file))
                        cleanup_report["space_to_free"] += temp_file.stat().st_size
                        
                        if not dry_run:
                            temp_file.unlink()
                            self.logger.info(f"Removed temporary file: {temp_file}")
            
        except Exception as e:
            cleanup_report["error"] = str(e)
            self.logger.error(f"Cleanup failed: {e}")
        
        return cleanup_report
    
    def backup_metadata(self, backup_path: Path) -> bool:
        """Backup critical metadata files"""
        try:
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Files to backup
            metadata_files = [
                self.config.MANIFEST_FILE,
                self.config.CHUNKS_FILE,
                self.config.METADATA_DB
            ]
            
            for source_file in metadata_files:
                if source_file.exists():
                    backup_file = backup_path / source_file.name
                    backup_file.write_bytes(source_file.read_bytes())
                    self.logger.info(f"Backed up {source_file.name}")
            
            self.logger.info(f"Metadata backup completed to {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
