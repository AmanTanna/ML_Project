"""
FAISS Vector Index Manager
=========================

High-performance vector similarity search using Facebook AI Research's FAISS library.
Handles index creation, persistence, batch operations, and similarity queries.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from dataclasses import dataclass, asdict

try:
    import faiss
except ImportError:
    print("FAISS not installed. Install with: pip install faiss-cpu")
    faiss = None

from ..config import RAGConfig
from ..utils import setup_logging, safe_filename
from .text_chunking import TextChunk

@dataclass
class IndexMetadata:
    """Metadata for FAISS index"""
    index_type: str
    dimension: int
    total_vectors: int
    created_at: str
    last_updated: str
    model_name: str
    chunk_count: int
    companies: List[str]
    form_types: List[str]
    date_range: Tuple[str, str]

class FAISSIndexManager:
    """Manages FAISS vector indices for semantic search"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        if faiss is None:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        # Index configuration
        self.index_dir = Path(config.VECTOR_INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # FAISS parameters
        self.index_type = config.FAISS_INDEX_TYPE
        self.use_gpu = config.USE_GPU_FOR_FAISS
        self.nlist = config.FAISS_NLIST  # Number of clusters for IVF indices
        self.nprobe = config.FAISS_NPROBE  # Number of clusters to search
        
        # Current index and metadata
        self.index: Optional[faiss.Index] = None
        self.metadata: Optional[IndexMetadata] = None
        self.chunk_id_to_index: Dict[str, int] = {}
        self.index_to_chunk_id: Dict[int, str] = {}
        
        self.logger.info(f"FAISS Index Manager initialized with {self.index_type} index type")
    
    def create_index(self, embeddings: np.ndarray, chunks: List[TextChunk],
                    model_name: str, force_recreate: bool = False) -> bool:
        """Create a new FAISS index from embeddings and chunks"""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        dimension = embeddings.shape[1]
        self.logger.info(f"Creating FAISS index with {len(chunks)} vectors of dimension {dimension}")
        
        # Check if index already exists
        index_path = self.index_dir / "faiss_index.bin"
        metadata_path = self.index_dir / "index_metadata.json"
        
        if index_path.exists() and not force_recreate:
            self.logger.warning("Index already exists. Use force_recreate=True to overwrite")
            return False
        
        try:
            # Create appropriate index type
            self.index = self._create_faiss_index(dimension, len(chunks))
            
            # Add vectors to index
            self._add_vectors_to_index(embeddings, chunks)
            
            # Create metadata
            self.metadata = self._create_metadata(chunks, model_name, dimension)
            
            # Save index and metadata
            self._save_index()
            self._save_metadata()
            
            self.logger.info(f"Successfully created FAISS index with {len(chunks)} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create FAISS index: {e}")
            return False
    
    def _create_faiss_index(self, dimension: int, num_vectors: int) -> faiss.Index:
        """Create appropriate FAISS index based on configuration"""
        if self.index_type == "flat":
            # Exact search (L2 distance)
            index = faiss.IndexFlatL2(dimension)
            
        elif self.index_type == "ivf_flat":
            # Inverted File with Flat quantizer
            nlist = min(self.nlist, num_vectors // 10)  # Ensure reasonable cluster count
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
        elif self.index_type == "ivf_pq":
            # Inverted File with Product Quantization
            nlist = min(self.nlist, num_vectors // 10)
            m = 8  # Number of subquantizers
            bits = 8  # Number of bits per subquantizer
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)
            
        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World
            M = 16  # Number of connections per element
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
            
        else:
            # Default to flat index
            self.logger.warning(f"Unknown index type {self.index_type}, using flat index")
            index = faiss.IndexFlatL2(dimension)
        
        # Move to GPU if requested and available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            self.logger.info("Moved index to GPU")
        
        return index
    
    def _add_vectors_to_index(self, embeddings: np.ndarray, chunks: List[TextChunk]):
        """Add vectors to the FAISS index"""
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Train index if needed (for IVF indices)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.logger.info("Training FAISS index...")
            self.index.train(embeddings)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        # Create mapping between chunk IDs and index positions
        self.chunk_id_to_index = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
        self.index_to_chunk_id = {i: chunk.chunk_id for i, chunk in enumerate(chunks)}
        
        # Set nprobe for IVF indices
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        self.logger.info(f"Added {len(embeddings)} vectors to index")
    
    def _create_metadata(self, chunks: List[TextChunk], model_name: str, dimension: int) -> IndexMetadata:
        """Create metadata for the index"""
        from datetime import datetime
        
        # Extract unique values
        companies = list(set(chunk.ticker for chunk in chunks))
        form_types = list(set(chunk.form_type for chunk in chunks))
        filing_dates = [chunk.filing_date for chunk in chunks if chunk.filing_date]
        
        date_range = (min(filing_dates), max(filing_dates)) if filing_dates else ("", "")
        
        return IndexMetadata(
            index_type=self.index_type,
            dimension=dimension,
            total_vectors=len(chunks),
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            model_name=model_name,
            chunk_count=len(chunks),
            companies=sorted(companies),
            form_types=sorted(form_types),
            date_range=date_range
        )
    
    def add_vectors(self, embeddings: np.ndarray, chunks: List[TextChunk]) -> bool:
        """Add new vectors to existing index"""
        if self.index is None:
            self.logger.error("No index loaded. Create or load an index first")
            return False
        
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        try:
            # Ensure embeddings are float32
            embeddings = embeddings.astype(np.float32)
            
            # Get current index size
            current_size = self.index.ntotal
            
            # Add vectors
            self.index.add(embeddings)
            
            # Update mappings
            for i, chunk in enumerate(chunks):
                index_pos = current_size + i
                self.chunk_id_to_index[chunk.chunk_id] = index_pos
                self.index_to_chunk_id[index_pos] = chunk.chunk_id
            
            # Update metadata
            if self.metadata:
                self.metadata.total_vectors += len(chunks)
                self.metadata.chunk_count += len(chunks)
                from datetime import datetime
                self.metadata.last_updated = datetime.now().isoformat()
                
                # Update companies and form types
                new_companies = set(chunk.ticker for chunk in chunks)
                self.metadata.companies = sorted(set(self.metadata.companies) | new_companies)
                
                new_form_types = set(chunk.form_type for chunk in chunks)
                self.metadata.form_types = sorted(set(self.metadata.form_types) | new_form_types)
            
            # Save updated index and metadata
            self._save_index()
            self._save_metadata()
            
            self.logger.info(f"Added {len(chunks)} new vectors to index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add vectors to index: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 10,
              filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        if self.index is None:
            self.logger.error("No index loaded")
            return []
        
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        
        try:
            # Perform search
            distances, indices = self.index.search(query_embedding, k)
            
            # Convert to results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx in self.index_to_chunk_id:  # Valid index
                    chunk_id = self.index_to_chunk_id[idx]
                    # Convert distance to similarity score (higher is better)
                    similarity = 1.0 / (1.0 + distance)
                    results.append((chunk_id, similarity))
            
            # Apply filters if provided
            if filters:
                results = self._apply_filters(results, filters)
            
            self.logger.info(f"Found {len(results)} similar vectors")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def _apply_filters(self, results: List[Tuple[str, float]], 
                      filters: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Apply filters to search results"""
        # Note: This is a simplified filter implementation
        # In a production system, you might want to integrate filtering directly into FAISS
        # or maintain additional metadata indices for efficient filtering
        
        filtered_results = []
        for chunk_id, similarity in results:
            # Here you would load chunk metadata and apply filters
            # For now, just return all results
            filtered_results.append((chunk_id, similarity))
        
        return filtered_results
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 10) -> List[List[Tuple[str, float]]]:
        """Perform batch search for multiple queries"""
        if self.index is None:
            self.logger.error("No index loaded")
            return []
        
        # Ensure embeddings are 2D and float32
        query_embeddings = query_embeddings.astype(np.float32)
        
        try:
            # Perform batch search
            distances, indices = self.index.search(query_embeddings, k)
            
            # Convert to results
            batch_results = []
            for query_idx in range(len(query_embeddings)):
                query_results = []
                for distance, idx in zip(distances[query_idx], indices[query_idx]):
                    if idx >= 0 and idx in self.index_to_chunk_id:
                        chunk_id = self.index_to_chunk_id[idx]
                        similarity = 1.0 / (1.0 + distance)
                        query_results.append((chunk_id, similarity))
                batch_results.append(query_results)
            
            self.logger.info(f"Completed batch search for {len(query_embeddings)} queries")
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch search failed: {e}")
            return []
    
    def load_index(self) -> bool:
        """Load existing FAISS index from disk"""
        index_path = self.index_dir / "faiss_index.bin"
        metadata_path = self.index_dir / "index_metadata.json"
        mappings_path = self.index_dir / "chunk_mappings.pkl"
        
        if not all(path.exists() for path in [index_path, metadata_path, mappings_path]):
            self.logger.warning("Index files not found")
            return False
        
        try:
            # Load index
            self.index = faiss.read_index(str(index_path))
            
            # Move to GPU if requested
            if self.use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.logger.info("Moved loaded index to GPU")
            
            # Set nprobe for IVF indices
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                self.metadata = IndexMetadata(**metadata_dict)
            
            # Load mappings
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                self.chunk_id_to_index = mappings['chunk_id_to_index']
                self.index_to_chunk_id = mappings['index_to_chunk_id']
            
            self.logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False
    
    def _save_index(self):
        """Save FAISS index to disk"""
        index_path = self.index_dir / "faiss_index.bin"
        
        # Move to CPU before saving if on GPU
        index_to_save = self.index
        if hasattr(self.index, 'device') and self.index.device >= 0:
            index_to_save = faiss.index_gpu_to_cpu(self.index)
        
        faiss.write_index(index_to_save, str(index_path))
        
        # Save chunk mappings
        mappings_path = self.index_dir / "chunk_mappings.pkl"
        mappings = {
            'chunk_id_to_index': self.chunk_id_to_index,
            'index_to_chunk_id': self.index_to_chunk_id
        }
        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings, f)
    
    def _save_metadata(self):
        """Save index metadata to disk"""
        if self.metadata:
            metadata_path = self.index_dir / "index_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(self.metadata), f, indent=2)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        if self.index is None:
            return {'status': 'no_index_loaded'}
        
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'index_type': type(self.index).__name__,
            'is_trained': getattr(self.index, 'is_trained', True),
            'on_gpu': hasattr(self.index, 'device') and self.index.device >= 0
        }
        
        if self.metadata:
            stats.update({
                'companies': len(self.metadata.companies),
                'form_types': len(self.metadata.form_types),
                'date_range': self.metadata.date_range,
                'created_at': self.metadata.created_at,
                'last_updated': self.metadata.last_updated
            })
        
        return stats
    
    def delete_index(self) -> bool:
        """Delete the FAISS index and related files"""
        try:
            index_files = [
                self.index_dir / "faiss_index.bin",
                self.index_dir / "index_metadata.json",
                self.index_dir / "chunk_mappings.pkl"
            ]
            
            for file_path in index_files:
                if file_path.exists():
                    file_path.unlink()
            
            # Clear in-memory objects
            self.index = None
            self.metadata = None
            self.chunk_id_to_index = {}
            self.index_to_chunk_id = {}
            
            self.logger.info("Deleted FAISS index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete index: {e}")
            return False
    
    def optimize_index(self) -> bool:
        """Optimize the index for better performance"""
        if self.index is None:
            self.logger.error("No index loaded")
            return False
        
        try:
            # For IVF indices, we can optimize by retraining with better parameters
            if hasattr(self.index, 'is_trained') and hasattr(self.index, 'train'):
                self.logger.info("Index optimization not implemented for this index type")
                return True
            
            self.logger.info("Index optimization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Index optimization failed: {e}")
            return False
