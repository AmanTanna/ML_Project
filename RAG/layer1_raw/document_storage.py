"""
Document Storage System
======================

Manages local storage of SEC filings with organized directory structure,
metadata tracking, and integrity validation.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .sec_client import Filing, SECClient
from ..config import RAGConfig
from ..utils import (
    setup_logging, get_file_hash, safe_filename, 
    get_current_timestamp, ensure_directory, format_bytes
)

@dataclass
class StoredDocument:
    """Represents a stored document with metadata"""
    filing: Filing
    file_path: Path
    file_size: int
    file_hash: str
    download_timestamp: str
    storage_format: str  # 'txt', 'html', 'pdf'
    validation_status: str  # 'valid', 'invalid', 'pending'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'filing': asdict(self.filing),
            'file_path': str(self.file_path),
            'file_size': self.file_size,
            'file_hash': self.file_hash,
            'download_timestamp': self.download_timestamp,
            'storage_format': self.storage_format,
            'validation_status': self.validation_status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredDocument':
        """Create from dictionary"""
        filing_data = data['filing']
        filing = Filing(**filing_data)
        
        return cls(
            filing=filing,
            file_path=Path(data['file_path']),
            file_size=data['file_size'],
            file_hash=data['file_hash'],
            download_timestamp=data['download_timestamp'],
            storage_format=data['storage_format'],
            validation_status=data['validation_status']
        )

class DocumentStorage:
    """Manages local storage of SEC filing documents"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Storage paths
        self.raw_pdfs_dir = config.RAW_PDFS_DIR
        self.registry_file = config.FILING_REGISTRY_FILE
        
        # Document registry
        self._registry: Dict[str, StoredDocument] = {}
        self._load_registry()
        
        # Initialize storage directories
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Create necessary storage directories"""
        directories = [
            self.raw_pdfs_dir,
            self.raw_pdfs_dir / "by_ticker",
            self.raw_pdfs_dir / "by_year", 
            self.raw_pdfs_dir / "by_form_type",
            self.registry_file.parent
        ]
        
        for directory in directories:
            ensure_directory(directory)
            
        self.logger.info("Document storage directories initialized")
    
    def _load_registry(self):
        """Load document registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                    
                for key, doc_data in registry_data.items():
                    self._registry[key] = StoredDocument.from_dict(doc_data)
                    
                self.logger.info(f"Loaded registry with {len(self._registry)} documents")
                
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Error loading registry: {e}")
                self._registry = {}
        else:
            self._registry = {}
    
    def _save_registry(self):
        """Save document registry to disk"""
        try:
            registry_data = {
                key: doc.to_dict() for key, doc in self._registry.items()
            }
            
            # Atomic write with backup
            temp_file = self.registry_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
                
            # Replace original file
            temp_file.replace(self.registry_file)
            
            self.logger.debug(f"Registry saved with {len(self._registry)} documents")
            
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}")
    
    def _generate_document_key(self, filing: Filing) -> str:
        """Generate unique key for a filing document"""
        return f"{filing.ticker}_{filing.form_type}_{filing.accession_number}"
    
    def _get_storage_path(self, filing: Filing, format_type: str = 'txt') -> Path:
        """Generate storage path for a filing"""
        # Organize by ticker/year/form_type
        safe_ticker = safe_filename(filing.ticker)
        safe_company = safe_filename(filing.company_name)[:50]  # Limit length
        year = filing.filing_year or filing.filing_date.split('-')[0]
        
        filename = f"{safe_ticker}_{filing.form_type}_{filing.accession_number}.{format_type}"
        
        return (self.raw_pdfs_dir / "by_ticker" / safe_ticker / 
                str(year) / filename)
    
    def _validate_document(self, file_path: Path, expected_size: int = None) -> bool:
        """Validate a stored document"""
        if not file_path.exists():
            return False
            
        # Check file size
        actual_size = file_path.stat().st_size
        if actual_size == 0:
            return False
            
        if expected_size and abs(actual_size - expected_size) > 1024:  # Allow 1KB difference
            self.logger.warning(f"Size mismatch for {file_path}: {actual_size} vs {expected_size}")
            
        # Basic content validation
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Check first 1KB
                
            # Check for SEC document markers
            sec_markers = ['SECURITIES AND EXCHANGE COMMISSION', 'FORM 10-', 'EDGAR']
            if not any(marker in content.upper() for marker in sec_markers):
                self.logger.warning(f"Document may not be valid SEC filing: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error validating document content: {e}")
            return False
            
        return True
    
    def store_document(self, filing: Filing, content: bytes, format_type: str = 'txt') -> Optional[StoredDocument]:
        """Store a filing document with metadata"""
        doc_key = self._generate_document_key(filing)
        
        # Check if already stored
        if doc_key in self._registry:
            existing_doc = self._registry[doc_key]
            if existing_doc.file_path.exists():
                self.logger.debug(f"Document already stored: {doc_key}")
                return existing_doc
        
        # Generate storage path
        storage_path = self._get_storage_path(filing, format_type)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write document
            with open(storage_path, 'wb') as f:
                f.write(content)
            
            # Calculate metadata
            file_size = len(content)
            file_hash = get_file_hash(storage_path)
            timestamp = get_current_timestamp()
            
            # Validate
            is_valid = self._validate_document(storage_path, file_size)
            validation_status = 'valid' if is_valid else 'invalid'
            
            # Create stored document record
            stored_doc = StoredDocument(
                filing=filing,
                file_path=storage_path,
                file_size=file_size,
                file_hash=file_hash,
                download_timestamp=timestamp,
                storage_format=format_type,
                validation_status=validation_status
            )
            
            # Update registry
            self._registry[doc_key] = stored_doc
            self._save_registry()
            
            # Create symlinks for easy access
            self._create_access_links(stored_doc)
            
            self.logger.info(f"Stored {filing.form_type} for {filing.ticker} ({format_bytes(file_size)})")
            return stored_doc
            
        except Exception as e:
            self.logger.error(f"Error storing document {doc_key}: {e}")
            # Cleanup partial file
            if storage_path.exists():
                storage_path.unlink()
            return None
    
    def _create_access_links(self, stored_doc: StoredDocument):
        """Create symlinks for alternative access patterns"""
        filing = stored_doc.filing
        
        # Link by year
        year_dir = self.raw_pdfs_dir / "by_year" / str(filing.filing_year)
        year_dir.mkdir(parents=True, exist_ok=True)
        year_link = year_dir / stored_doc.file_path.name
        
        # Link by form type
        form_dir = self.raw_pdfs_dir / "by_form_type" / filing.form_type
        form_dir.mkdir(parents=True, exist_ok=True)
        form_link = form_dir / stored_doc.file_path.name
        
        # Create symlinks (ignore if they already exist)
        for link_path in [year_link, form_link]:
            try:
                if not link_path.exists():
                    link_path.symlink_to(stored_doc.file_path.resolve())
            except Exception as e:
                self.logger.debug(f"Could not create symlink {link_path}: {e}")
    
    def get_document(self, ticker: str, form_type: str, accession_number: str) -> Optional[StoredDocument]:
        """Retrieve a stored document by identifiers"""
        doc_key = f"{ticker}_{form_type}_{accession_number}"
        return self._registry.get(doc_key)
    
    def list_documents(self, ticker: str = None, form_type: str = None, 
                      year: int = None) -> List[StoredDocument]:
        """List stored documents with optional filtering"""
        documents = list(self._registry.values())
        
        if ticker:
            documents = [d for d in documents if d.filing.ticker.upper() == ticker.upper()]
        if form_type:
            documents = [d for d in documents if d.filing.form_type == form_type]
        if year:
            documents = [d for d in documents if d.filing.filing_year == year]
            
        return documents
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        documents = list(self._registry.values())
        
        if not documents:
            return {
                'total_documents': 0,
                'total_size': 0,
                'total_size_formatted': format_bytes(0),
                'by_ticker': {},
                'by_form_type': {},
                'by_year': {},
                'validation_status': {}
            }
        
        total_size = sum(d.file_size for d in documents)
        
        # Group by various dimensions
        by_ticker = {}
        by_form_type = {}
        by_year = {}
        validation_status = {}
        
        for doc in documents:
            # By ticker
            ticker = doc.filing.ticker
            if ticker not in by_ticker:
                by_ticker[ticker] = {'count': 0, 'size': 0}
            by_ticker[ticker]['count'] += 1
            by_ticker[ticker]['size'] += doc.file_size
            
            # By form type
            form_type = doc.filing.form_type
            if form_type not in by_form_type:
                by_form_type[form_type] = {'count': 0, 'size': 0}
            by_form_type[form_type]['count'] += 1
            by_form_type[form_type]['size'] += doc.file_size
            
            # By year
            year = doc.filing.filing_year
            if year not in by_year:
                by_year[year] = {'count': 0, 'size': 0}
            by_year[year]['count'] += 1
            by_year[year]['size'] += doc.file_size
            
            # By validation status
            status = doc.validation_status
            validation_status[status] = validation_status.get(status, 0) + 1
        
        return {
            'total_documents': len(documents),
            'total_size': total_size,
            'total_size_formatted': format_bytes(total_size),
            'by_ticker': by_ticker,
            'by_form_type': by_form_type,
            'by_year': by_year,
            'validation_status': validation_status
        }
    
    def cleanup_invalid_documents(self) -> int:
        """Remove invalid documents and update registry"""
        invalid_docs = [
            key for key, doc in self._registry.items() 
            if doc.validation_status == 'invalid' or not doc.file_path.exists()
        ]
        
        removed_count = 0
        for key in invalid_docs:
            doc = self._registry[key]
            
            # Remove file if it exists
            if doc.file_path.exists():
                try:
                    doc.file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    self.logger.error(f"Error removing invalid file {doc.file_path}: {e}")
            
            # Remove from registry
            del self._registry[key]
        
        if removed_count > 0:
            self._save_registry()
            self.logger.info(f"Cleaned up {removed_count} invalid documents")
            
        return removed_count
    
    def revalidate_all_documents(self) -> Dict[str, int]:
        """Revalidate all stored documents"""
        validation_counts = {'valid': 0, 'invalid': 0, 'missing': 0}
        
        for doc in self._registry.values():
            if not doc.file_path.exists():
                doc.validation_status = 'invalid'
                validation_counts['missing'] += 1
            elif self._validate_document(doc.file_path, doc.file_size):
                doc.validation_status = 'valid'
                validation_counts['valid'] += 1
            else:
                doc.validation_status = 'invalid'
                validation_counts['invalid'] += 1
        
        self._save_registry()
        self.logger.info(f"Revalidation complete: {validation_counts}")
        
        return validation_counts
