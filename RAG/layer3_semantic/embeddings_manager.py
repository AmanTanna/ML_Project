"""
Embeddings Manager
==================

Manages text embeddings using sentence-transformers models.
Handles batch processing, caching, and embedding optimization.
"""

import logging
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from ..config import RAGConfig
from ..utils import setup_logging, get_current_timestamp, progress_bar, ProgressTracker

@dataclass
class EmbeddingMetadata:
    """Metadata for an embedding vector"""
    chunk_id: str
    source_document: str
    chunk_text: str
    chunk_start: int
    chunk_end: int
    ticker: str
    form_type: str
    filing_date: str
    section: Optional[str] = None
    embedding_model: str = ""
    embedding_timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingMetadata':
        return cls(**data)

class EmbeddingsManager:
    """Manages text embeddings for semantic search"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Check if sentence-transformers is available
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        # Initialize embedding model
        self.model_name = config.EMBEDDING_MODEL
        self.embedding_dim = config.EMBEDDING_DIMENSION
        
        # Load model
        self._load_model()
        
        # Cache configuration
        self.cache_dir = Path(config.VECTOR_INDEX_DIR) / "embeddings_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "embeddings.pkl"
        
        # Initialize cache and statistics
        self._embedding_cache = {}
        self.stats = {
            'total_embeddings_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Load existing cache
        self._load_cache()
        
        self.logger.info(f"Embeddings Manager initialized with model: {self.model_name}")
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Verify embedding dimension
            test_embedding = self.model.encode(["test"])
            actual_dim = test_embedding.shape[1]
            
            if actual_dim != self.embedding_dim:
                self.logger.warning(
                    f"Model dimension {actual_dim} != config dimension {self.embedding_dim}. "
                    f"Using actual dimension: {actual_dim}"
                )
                self.embedding_dim = actual_dim
            
            self.logger.info(f"Model loaded successfully. Dimension: {self.embedding_dim}")
            
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            raise
    
    def _load_cache(self):
        """Load embedding cache from disk"""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                self.logger.warning(f"Could not load embedding cache: {e}")
                self._embedding_cache = {}
        else:
            self._embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
            self.logger.debug(f"Saved {len(self._embedding_cache)} embeddings to cache")
        except Exception as e:
            self.logger.error(f"Error saving embedding cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Get hash for text to use as cache key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for a single text"""
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)
        
        text = text.strip()
        text_hash = self._get_text_hash(text)
        
        # Check cache
        if use_cache and text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        try:
            # Generate embedding
            embedding = self.model.encode([text], show_progress_bar=False)[0]
            
            # Cache result
            if use_cache:
                self._embedding_cache[text_hash] = embedding
                
                # Periodically save cache
                if len(self._embedding_cache) % 100 == 0:
                    self._save_cache()
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, 
                   show_progress: bool = True, use_cache: bool = True) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with batching"""
        if not texts:
            return []
        
        embeddings = []
        cache_hits = 0
        cache_misses = 0
        
        # Check cache for all texts first
        if use_cache:
            cached_embeddings = {}
            texts_to_embed = []
            text_indices = []
            
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    cached_embeddings[i] = np.zeros(self.embedding_dim)
                    continue
                    
                text_hash = self._get_text_hash(text.strip())
                if text_hash in self._embedding_cache:
                    cached_embeddings[i] = self._embedding_cache[text_hash]
                    cache_hits += 1
                else:
                    texts_to_embed.append(text.strip())
                    text_indices.append(i)
                    cache_misses += 1
        else:
            texts_to_embed = [t.strip() for t in texts if t and t.strip()]
            text_indices = list(range(len(texts_to_embed)))
            cached_embeddings = {}
        
        # Embed uncached texts
        if texts_to_embed:
            if show_progress:
                self.logger.info(f"Embedding {len(texts_to_embed)} texts (cache hits: {cache_hits}, misses: {cache_misses})")
            
            progress = ProgressTracker(len(texts_to_embed), "Generating Embeddings") if show_progress else None
            
            # Process in batches
            new_embeddings = []
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i:i + batch_size]
                
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts, 
                        batch_size=len(batch_texts),
                        show_progress_bar=False
                    )
                    new_embeddings.extend(batch_embeddings)
                    
                    # Update cache
                    if use_cache:
                        for text, embedding in zip(batch_texts, batch_embeddings):
                            text_hash = self._get_text_hash(text)
                            self._embedding_cache[text_hash] = embedding
                    
                    if progress:
                        progress.update(len(batch_texts))
                        
                except Exception as e:
                    self.logger.error(f"Error in batch embedding: {e}")
                    # Add zero embeddings for failed batch
                    new_embeddings.extend([np.zeros(self.embedding_dim)] * len(batch_texts))
                    if progress:
                        progress.update(len(batch_texts))
            
            if progress:
                progress.finish()
        else:
            new_embeddings = []
        
        # Reconstruct full embedding list
        embeddings = [None] * len(texts)
        
        # Fill cached embeddings
        for i, embedding in cached_embeddings.items():
            embeddings[i] = embedding
        
        # Fill new embeddings
        new_embedding_idx = 0
        for i in text_indices:
            if i not in cached_embeddings:
                embeddings[i] = new_embeddings[new_embedding_idx]
                new_embedding_idx += 1
        
        # Fill any None values with zero embeddings
        for i in range(len(embeddings)):
            if embeddings[i] is None:
                embeddings[i] = np.zeros(self.embedding_dim)
        
        # Save cache if we added new embeddings
        if use_cache and new_embeddings:
            self._save_cache()
        
        # Log embedding statistics
        self.stats['total_embeddings_generated'] += len([e for e in embeddings if e is not None])
        
        # Convert to numpy array and return
        result = np.array(embeddings)
        self.logger.info(f"Generated embeddings array: {result.shape}")
        return result
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Normalize embeddings
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(norm1, norm2)
        return float(similarity)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'cache_size': len(self._embedding_cache),
            'total_embeddings_generated': self.stats['total_embeddings_generated'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses']
        }
    
    def embed_document_chunks(self, chunks: List[Dict[str, Any]], 
                            show_progress: bool = True) -> List[Tuple[np.ndarray, EmbeddingMetadata]]:
        """Embed document chunks with metadata"""
        if not chunks:
            return []
        
        self.logger.info(f"Embedding {len(chunks)} document chunks")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts, show_progress=show_progress)
        
        # Create metadata objects
        results = []
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            metadata = EmbeddingMetadata(
                chunk_id=chunk.get('chunk_id', f"chunk_{i}"),
                source_document=chunk.get('source_document', ''),
                chunk_text=chunk['text'],
                chunk_start=chunk.get('start_pos', 0),
                chunk_end=chunk.get('end_pos', len(chunk['text'])),
                ticker=chunk.get('ticker', ''),
                form_type=chunk.get('form_type', ''),
                filing_date=chunk.get('filing_date', ''),
                section=chunk.get('section'),
                embedding_model=self.model_name,
                embedding_timestamp=get_current_timestamp()
            )
            
            results.append((embedding, metadata))
        
        return results
    
    def semantic_similarity(self, query_embedding: np.ndarray, 
                          document_embeddings: List[np.ndarray],
                          method: str = 'cosine') -> List[float]:
        """Calculate semantic similarity scores"""
        if not document_embeddings:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        doc_matrix = np.vstack(document_embeddings)
        
        if method == 'cosine':
            # Cosine similarity
            query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
            
            # Avoid division by zero
            query_norm = np.where(query_norm == 0, 1, query_norm)
            doc_norms = np.where(doc_norms == 0, 1, doc_norms)
            
            similarities = np.dot(query_embedding, doc_matrix.T) / (query_norm * doc_norms.T)
            return similarities.flatten().tolist()
            
        elif method == 'dot':
            # Dot product similarity
            similarities = np.dot(query_embedding, doc_matrix.T)
            return similarities.flatten().tolist()
            
        elif method == 'euclidean':
            # Negative euclidean distance (higher = more similar)
            distances = np.linalg.norm(doc_matrix - query_embedding, axis=1)
            similarities = -distances
            return similarities.tolist()
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def find_similar_chunks(self, query: str, 
                          chunk_embeddings: List[Tuple[np.ndarray, EmbeddingMetadata]],
                          top_k: int = 10,
                          min_similarity: float = 0.3) -> List[Tuple[float, EmbeddingMetadata]]:
        """Find most similar chunks to a query"""
        if not chunk_embeddings:
            return []
        
        # Generate query embedding
        query_embedding = self.embed_text(query)
        
        # Extract embeddings
        embeddings = [emb for emb, _ in chunk_embeddings]
        metadatas = [meta for _, meta in chunk_embeddings]
        
        # Calculate similarities
        similarities = self.semantic_similarity(query_embedding, embeddings)
        
        # Create results with metadata
        results = []
        for similarity, metadata in zip(similarities, metadatas):
            if similarity >= min_similarity:
                results.append((similarity, metadata))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'cache_size': len(self._embedding_cache),
            'cache_file_exists': self._cache_file.exists(),
            'cache_file_size': self._cache_file.stat().st_size if self._cache_file.exists() else 0
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        self._embedding_cache.clear()
        if self._cache_file.exists():
            self._cache_file.unlink()
        self.logger.info("Embedding cache cleared")
    
    def optimize_cache(self, max_size: int = 10000):
        """Optimize cache by removing least recently used embeddings"""
        if len(self._embedding_cache) <= max_size:
            return
        
        # This is a simple implementation - in production you'd want LRU tracking
        cache_items = list(self._embedding_cache.items())
        # Keep the first max_size items (you could implement proper LRU here)
        self._embedding_cache = dict(cache_items[:max_size])
        self._save_cache()
        
        self.logger.info(f"Cache optimized to {len(self._embedding_cache)} embeddings")
