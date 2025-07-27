"""
Semantic Search Manager
======================

Main orchestrator for semantic search operations in the RAG system.
Integrates embeddings, text chunking, and FAISS vector search.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
from datetime import datetime

from ..config import RAGConfig
from ..utils import setup_logging, safe_filename
from ..layer1_raw.manager import RawSourceManager
from ..layer2_structured.manager import StructuredDataManager
from .embeddings_manager import EmbeddingsManager
from .text_chunking import TextChunkingManager, TextChunk
from .faiss_manager import FAISSIndexManager

@dataclass
class SearchResult:
    """Represents a semantic search result"""
    chunk_id: str
    chunk_text: str
    similarity_score: float
    source_document: str
    ticker: str
    form_type: str
    filing_date: str
    section: Optional[str] = None
    chunk_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'chunk_text': self.chunk_text,
            'similarity_score': self.similarity_score,
            'source_document': self.source_document,
            'ticker': self.ticker,
            'form_type': self.form_type,
            'filing_date': self.filing_date,
            'section': self.section,
            'chunk_index': self.chunk_index
        }

class SemanticSearchManager:
    """Main manager for semantic search operations"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Initialize component managers
        self.embeddings_manager = EmbeddingsManager(config)
        self.chunking_manager = TextChunkingManager(config)
        self.faiss_manager = FAISSIndexManager(config)
        
        # Cache for chunks metadata
        self.chunks_metadata: Dict[str, TextChunk] = {}
        
        # Search cache
        self.search_cache: Dict[str, List[SearchResult]] = {}
        self.cache_max_size = config.SEARCH_CACHE_SIZE
        
        self.logger.info("Semantic Search Manager initialized")
    
    def build_search_index(self, use_existing_data: bool = True, 
                          companies: Optional[List[str]] = None,
                          max_documents: Optional[int] = None) -> bool:
        """Build the complete semantic search index"""
        self.logger.info("Starting semantic search index build...")
        
        try:
            # Step 1: Get documents from raw source layer
            raw_manager = RawSourceManager(self.config)
            
            if use_existing_data:
                documents = self._load_documents_from_storage(companies, max_documents)
            else:
                # Download fresh documents
                if companies is None:
                    companies = self.config.SP500_COMPANIES[:10]  # Default to first 10 for testing
                
                documents = []
                for company in companies:
                    company_docs = raw_manager.download_annual_reports(
                        company, limit=3  # Limit per company for testing
                    )
                    documents.extend(company_docs)
                    
                    if max_documents and len(documents) >= max_documents:
                        documents = documents[:max_documents]
                        break
            
            if not documents:
                self.logger.error("No documents found to index")
                return False
            
            self.logger.info(f"Processing {len(documents)} documents for indexing")
            
            # Step 2: Chunk documents
            all_chunks = []
            for doc in documents:
                doc_chunks = self.chunking_manager.chunk_document(
                    text=doc['content'],
                    source_document=doc['filename'],
                    ticker=doc['ticker'],
                    form_type=doc['form_type'],
                    filing_date=doc['filing_date']
                )
                all_chunks.extend(doc_chunks)
            
            if not all_chunks:
                self.logger.error("No chunks created from documents")
                return False
            
            self.logger.info(f"Created {len(all_chunks)} text chunks")
            
            # Step 3: Generate embeddings
            chunk_texts = [chunk.text for chunk in all_chunks]
            embeddings = self.embeddings_manager.embed_texts(chunk_texts)
            
            if embeddings is None or len(embeddings) == 0:
                self.logger.error("Failed to generate embeddings")
                return False
            
            # Step 4: Create FAISS index
            success = self.faiss_manager.create_index(
                embeddings=embeddings,
                chunks=all_chunks,
                model_name=self.embeddings_manager.model_name,
                force_recreate=True
            )
            
            if not success:
                self.logger.error("Failed to create FAISS index")
                return False
            
            # Step 5: Cache chunk metadata
            self.chunks_metadata = {chunk.chunk_id: chunk for chunk in all_chunks}
            self._save_chunks_metadata()
            
            # Step 6: Save index statistics
            self._save_index_statistics(all_chunks, embeddings)
            
            self.logger.info("Successfully built semantic search index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build search index: {e}")
            return False
    
    def _load_documents_from_storage(self, companies: Optional[List[str]] = None,
                                   max_documents: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load documents from existing storage"""
        documents = []
        
        # Load from raw storage
        raw_storage_path = Path(self.config.RAW_PDFS_DIR) / "by_ticker"
        
        if not raw_storage_path.exists():
            self.logger.warning(f"Raw storage path not found: {raw_storage_path}")
            return documents
        
        company_dirs = companies or [d.name for d in raw_storage_path.iterdir() if d.is_dir()]
        
        for company in company_dirs:
            company_path = raw_storage_path / company
            if not company_path.exists():
                continue
            
            for file_path in company_path.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract metadata from filename
                    filename = file_path.name
                    parts = filename.replace('.txt', '').split('_')
                    
                    if len(parts) >= 3:
                        ticker = parts[0]
                        form_type = parts[1]
                        filing_date = parts[2]
                    else:
                        ticker = company
                        form_type = "10-K"
                        filing_date = "2023-01-01"
                    
                    documents.append({
                        'content': content,
                        'filename': filename,
                        'ticker': ticker,
                        'form_type': form_type,
                        'filing_date': filing_date,
                        'file_path': str(file_path)
                    })
                    
                    if max_documents and len(documents) >= max_documents:
                        return documents
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load document {file_path}: {e}")
                    continue
        
        self.logger.info(f"Loaded {len(documents)} documents from storage")
        return documents
    
    def search(self, query: str, k: int = 10, 
              filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform semantic search"""
        if not query or not query.strip():
            return []
        
        # Check cache
        cache_key = f"{query}|{k}|{str(filters)}"
        if cache_key in self.search_cache:
            self.logger.info("Returning cached search results")
            return self.search_cache[cache_key]
        
        try:
            # Load index if not loaded
            if self.faiss_manager.index is None:
                if not self.faiss_manager.load_index():
                    self.logger.error("No search index available")
                    return []
            
            # Load chunks metadata if not loaded
            if not self.chunks_metadata:
                self._load_chunks_metadata()
            
            # Generate query embedding
            query_embedding = self.embeddings_manager.embed_texts([query])
            if query_embedding is None or len(query_embedding) == 0:
                self.logger.error("Failed to generate query embedding")
                return []
            
            # Search in FAISS index
            search_results = self.faiss_manager.search(
                query_embedding[0], 
                k=k, 
                filters=filters
            )
            
            # Convert to SearchResult objects
            results = []
            for chunk_id, similarity_score in search_results:
                if chunk_id in self.chunks_metadata:
                    chunk = self.chunks_metadata[chunk_id]
                    result = SearchResult(
                        chunk_id=chunk_id,
                        chunk_text=chunk.text,
                        similarity_score=similarity_score,
                        source_document=chunk.source_document,
                        ticker=chunk.ticker,
                        form_type=chunk.form_type,
                        filing_date=chunk.filing_date,
                        section=chunk.section,
                        chunk_index=chunk.chunk_index
                    )
                    results.append(result)
            
            # Cache results
            self._cache_search_results(cache_key, results)
            
            self.logger.info(f"Found {len(results)} search results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def batch_search(self, queries: List[str], k: int = 10) -> List[List[SearchResult]]:
        """Perform batch semantic search"""
        if not queries:
            return []
        
        try:
            # Load index if not loaded
            if self.faiss_manager.index is None:
                if not self.faiss_manager.load_index():
                    self.logger.error("No search index available")
                    return []
            
            # Load chunks metadata if not loaded
            if not self.chunks_metadata:
                self._load_chunks_metadata()
            
            # Generate query embeddings
            query_embeddings = self.embeddings_manager.embed_texts(queries)
            if query_embeddings is None or len(query_embeddings) == 0:
                self.logger.error("Failed to generate query embeddings")
                return []
            
            # Batch search in FAISS index
            batch_search_results = self.faiss_manager.batch_search(query_embeddings, k=k)
            
            # Convert to SearchResult objects
            all_results = []
            for i, search_results in enumerate(batch_search_results):
                query_results = []
                for chunk_id, similarity_score in search_results:
                    if chunk_id in self.chunks_metadata:
                        chunk = self.chunks_metadata[chunk_id]
                        result = SearchResult(
                            chunk_id=chunk_id,
                            chunk_text=chunk.text,
                            similarity_score=similarity_score,
                            source_document=chunk.source_document,
                            ticker=chunk.ticker,
                            form_type=chunk.form_type,
                            filing_date=chunk.filing_date,
                            section=chunk.section,
                            chunk_index=chunk.chunk_index
                        )
                        query_results.append(result)
                all_results.append(query_results)
            
            self.logger.info(f"Completed batch search for {len(queries)} queries")
            return all_results
            
        except Exception as e:
            self.logger.error(f"Batch search failed: {e}")
            return []
    
    def get_similar_chunks(self, chunk_id: str, k: int = 5) -> List[SearchResult]:
        """Find chunks similar to a given chunk"""
        if chunk_id not in self.chunks_metadata:
            self.logger.error(f"Chunk ID not found: {chunk_id}")
            return []
        
        chunk = self.chunks_metadata[chunk_id]
        return self.search(chunk.text, k=k+1)[1:]  # Exclude the chunk itself
    
    def search_by_company(self, query: str, ticker: str, k: int = 10) -> List[SearchResult]:
        """Search within documents of a specific company"""
        filters = {'ticker': ticker.upper()}
        return self.search(query, k=k, filters=filters)
    
    def search_by_section(self, query: str, section: str, k: int = 10) -> List[SearchResult]:
        """Search within a specific section type"""
        filters = {'section': section}
        return self.search(query, k=k, filters=filters)
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the search index"""
        stats = {
            'faiss_stats': self.faiss_manager.get_index_stats(),
            'embeddings_stats': self.embeddings_manager.get_stats(),
            'chunks_count': len(self.chunks_metadata),
            'cache_size': len(self.search_cache)
        }
        
        if self.chunks_metadata:
            chunks = list(self.chunks_metadata.values())
            chunking_stats = self.chunking_manager.get_chunking_stats(chunks)
            stats['chunking_stats'] = chunking_stats
        
        return stats
    
    def clear_cache(self):
        """Clear the search cache"""
        self.search_cache.clear()
        self.logger.info("Cleared search cache")
    
    def _cache_search_results(self, cache_key: str, results: List[SearchResult]):
        """Cache search results with size limit"""
        if len(self.search_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.search_cache))
            del self.search_cache[oldest_key]
        
        self.search_cache[cache_key] = results
    
    def _save_chunks_metadata(self):
        """Save chunks metadata to disk"""
        metadata_path = Path(self.config.VECTOR_INDEX_DIR) / "chunks_metadata.json"
        
        # Convert to serializable format
        serializable_metadata = {}
        for chunk_id, chunk in self.chunks_metadata.items():
            serializable_metadata[chunk_id] = chunk.to_dict()
        
        with open(metadata_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        
        self.logger.info(f"Saved metadata for {len(self.chunks_metadata)} chunks")
    
    def _load_chunks_metadata(self) -> bool:
        """Load chunks metadata from disk"""
        metadata_path = Path(self.config.VECTOR_INDEX_DIR) / "chunks_metadata.json"
        
        if not metadata_path.exists():
            self.logger.warning("Chunks metadata file not found")
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                serializable_metadata = json.load(f)
            
            # Convert back to TextChunk objects
            self.chunks_metadata = {}
            for chunk_id, chunk_data in serializable_metadata.items():
                chunk = TextChunk(
                    chunk_id=chunk_data['chunk_id'],
                    text=chunk_data['text'],
                    start_pos=chunk_data['start_pos'],
                    end_pos=chunk_data['end_pos'],
                    source_document=chunk_data['source_document'],
                    ticker=chunk_data['ticker'],
                    form_type=chunk_data['form_type'],
                    filing_date=chunk_data['filing_date'],
                    section=chunk_data.get('section'),
                    chunk_index=chunk_data.get('chunk_index', 0),
                    token_count=chunk_data.get('token_count', 0)
                )
                self.chunks_metadata[chunk_id] = chunk
            
            self.logger.info(f"Loaded metadata for {len(self.chunks_metadata)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load chunks metadata: {e}")
            return False
    
    def _save_index_statistics(self, chunks: List[TextChunk], embeddings: np.ndarray):
        """Save index building statistics"""
        stats = {
            'build_timestamp': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'total_embeddings': len(embeddings),
            'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'companies': list(set(chunk.ticker for chunk in chunks)),
            'form_types': list(set(chunk.form_type for chunk in chunks)),
            'date_range': [
                min(chunk.filing_date for chunk in chunks if chunk.filing_date),
                max(chunk.filing_date for chunk in chunks if chunk.filing_date)
            ],
            'sections': list(set(chunk.section for chunk in chunks if chunk.section)),
            'avg_chunk_length': sum(len(chunk.text) for chunk in chunks) / len(chunks),
            'model_name': self.embeddings_manager.model_name
        }
        
        stats_path = Path(self.config.VECTOR_INDEX_DIR) / "build_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info("Saved index building statistics")
    
    def rebuild_index(self, companies: Optional[List[str]] = None, 
                     max_documents: Optional[int] = None) -> bool:
        """Rebuild the entire search index"""
        self.logger.info("Rebuilding search index...")
        
        # Clear existing data
        self.faiss_manager.delete_index()
        self.chunks_metadata.clear()
        self.search_cache.clear()
        
        # Rebuild
        return self.build_search_index(
            use_existing_data=True,
            companies=companies,
            max_documents=max_documents
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the semantic search system"""
        health = {
            'status': 'healthy',
            'issues': [],
            'components': {}
        }
        
        # Check FAISS index
        try:
            if self.faiss_manager.index is None:
                self.faiss_manager.load_index()
            
            if self.faiss_manager.index is not None:
                health['components']['faiss'] = 'loaded'
            else:
                health['components']['faiss'] = 'not_loaded'
                health['issues'].append('FAISS index not loaded')
        except Exception as e:
            health['components']['faiss'] = 'error'
            health['issues'].append(f'FAISS error: {e}')
        
        # Check embeddings model
        try:
            test_embedding = self.embeddings_manager.embed_texts(["test"])
            if test_embedding is not None:
                health['components']['embeddings'] = 'working'
            else:
                health['components']['embeddings'] = 'failed'
                health['issues'].append('Embeddings model not working')
        except Exception as e:
            health['components']['embeddings'] = 'error'
            health['issues'].append(f'Embeddings error: {e}')
        
        # Check chunks metadata
        if not self.chunks_metadata:
            self._load_chunks_metadata()
        
        if self.chunks_metadata:
            health['components']['chunks_metadata'] = 'loaded'
        else:
            health['components']['chunks_metadata'] = 'not_loaded'
            health['issues'].append('Chunks metadata not loaded')
        
        # Overall status
        if health['issues']:
            health['status'] = 'degraded' if len(health['issues']) < 3 else 'unhealthy'
        
        return health
