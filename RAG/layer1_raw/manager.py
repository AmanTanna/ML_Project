"""
Raw Source Layer Manager
========================

Manages Layer 1 of the RAG system - raw SEC filing documents.
Orchestrates SEC EDGAR API client and document storage system.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .sec_client import SECClient, Filing
from .document_storage import DocumentStorage, StoredDocument
from ..config import RAGConfig
from ..utils import setup_logging, progress_bar, ProgressTracker

class RawSourceManager:
    """Manager for Layer 1 Raw Source - SEC EDGAR filings"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Initialize components
        self.sec_client = SECClient(config)
        self.document_storage = DocumentStorage(config)
        
        self.logger.info("Raw Source Manager initialized")
    
    def initialize(self):
        """Initialize Layer 1 storage and components"""
        self.logger.info("Initializing Raw Source Layer (Layer 1)")
        
        # Validate configuration
        required_dirs = [
            self.config.RAW_PDFS_DIR,
            self.config.DATA_DIR
        ]
        
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Test SEC API connectivity
        test_ticker = 'AAPL'
        test_cik = self.sec_client.get_company_cik(test_ticker)
        if test_cik:
            self.logger.info(f"SEC API connectivity verified (CIK for {test_ticker}: {test_cik})")
        else:
            self.logger.warning("Could not verify SEC API connectivity")
            
        self.logger.info("Raw Source Layer initialization complete")
    
    def download_sp500_filings(self, years: List[int] = None, 
                              form_types: List[str] = None,
                              limit_companies: int = None) -> Dict[str, Any]:
        """Download S&P 500 filings for specified years"""
        if years is None:
            years = self.config.SP500_YEARS
        if form_types is None:
            form_types = self.config.FILING_TYPES
            
        self.logger.info(f"Starting S&P 500 bulk download: {years} years, {form_types} forms")
        
        # Get S&P 500 tickers
        tickers = self.sec_client.get_sp500_tickers()
        if limit_companies:
            tickers = tickers[:limit_companies]
            
        # Download filings metadata
        all_filings = self.sec_client.bulk_download_filings(tickers, years, form_types)
        
        # Download and store documents
        download_stats = self._download_and_store_documents(all_filings)
        
        return {
            'companies_processed': len(all_filings),
            'years': years,
            'form_types': form_types,
            'download_stats': download_stats,
            'storage_stats': self.document_storage.get_storage_stats()
        }
    
    def _download_and_store_documents(self, all_filings: Dict[str, List[Filing]]) -> Dict[str, Any]:
        """Download and store filing documents"""
        total_filings = sum(len(filings) for filings in all_filings.values())
        
        if total_filings == 0:
            return {'downloaded': 0, 'failed': 0, 'skipped': 0}
        
        self.logger.info(f"Downloading {total_filings} filing documents")
        
        progress = ProgressTracker(total_filings, "Document Download")
        downloaded = 0
        failed = 0
        skipped = 0
        
        for ticker, filings in all_filings.items():
            self.logger.info(f"Downloading {len(filings)} filings for {ticker}")
            
            for filing in filings:
                try:
                    # Check if already stored
                    existing = self.document_storage.get_document(
                        filing.ticker, filing.form_type, filing.accession_number
                    )
                    
                    if existing and existing.file_path.exists():
                        skipped += 1
                        progress.update()
                        continue
                    
                    # Download document content
                    response = self.sec_client._make_request(filing.document_url)
                    if not response:
                        failed += 1
                        progress.update()
                        continue
                        
                    # Store document
                    stored_doc = self.document_storage.store_document(
                        filing, response.content, 'txt'
                    )
                    
                    if stored_doc:
                        downloaded += 1
                    else:
                        failed += 1
                        
                    progress.update()
                    
                except Exception as e:
                    self.logger.error(f"Error processing {filing.ticker} {filing.form_type}: {e}")
                    failed += 1
                    progress.update()
        
        progress.finish()
        
        stats = {
            'downloaded': downloaded,
            'failed': failed,
            'skipped': skipped,
            'total': total_filings
        }
        
        self.logger.info(f"Download complete: {stats}")
        return stats
    
    def get_company_filings(self, ticker: str, years: List[int] = None,
                           form_types: List[str] = None) -> List[StoredDocument]:
        """Get stored filings for a specific company"""
        stored_docs = []
        
        for year in (years or self.config.SP500_YEARS):
            for form_type in (form_types or self.config.FILING_TYPES):
                docs = self.document_storage.list_documents(
                    ticker=ticker, form_type=form_type, year=year
                )
                stored_docs.extend(docs)
                
        return stored_docs
    
    def search_filings(self, ticker: str = None, form_type: str = None, 
                      year: int = None) -> List[StoredDocument]:
        """Search stored filings with filters"""
        return self.document_storage.list_documents(ticker, form_type, year)
    
    def get_filing_content(self, ticker: str, form_type: str, 
                          accession_number: str) -> Optional[str]:
        """Get the text content of a specific filing"""
        doc = self.document_storage.get_document(ticker, form_type, accession_number)
        
        if not doc or not doc.file_path.exists():
            return None
            
        try:
            with open(doc.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading filing content: {e}")
            return None
    
    def get_layer_stats(self) -> Dict[str, Any]:
        """Get comprehensive Layer 1 statistics"""
        storage_stats = self.document_storage.get_storage_stats()
        
        return {
            'layer': 'Layer 1 - Raw Source',
            'sec_requests_made': self.sec_client.request_count,
            'companies_in_cache': len(self.sec_client._company_cache),
            'storage_stats': storage_stats,
            'health_status': self._health_check()
        }
    
    def _health_check(self) -> Dict[str, Any]:
        """Perform health check on Layer 1 components"""
        health = {
            'sec_api_accessible': False,
            'storage_directories_exist': True,
            'registry_file_valid': False,
            'recent_downloads': 0
        }
        
        # Test SEC API
        try:
            test_cik = self.sec_client.get_company_cik('AAPL')
            health['sec_api_accessible'] = test_cik is not None
        except Exception:
            pass
        
        # Check storage directories
        required_dirs = [self.config.RAW_PDFS_DIR, self.config.DATA_DIR]
        for directory in required_dirs:
            if not directory.exists():
                health['storage_directories_exist'] = False
                break
        
        # Check registry file
        health['registry_file_valid'] = self.config.FILING_REGISTRY_FILE.exists()
        
        # Count recent downloads (last 24 hours)
        cutoff = datetime.now().timestamp() - (24 * 3600)
        recent_count = 0
        
        for doc in self.document_storage.list_documents():
            try:
                doc_time = datetime.fromisoformat(doc.download_timestamp).timestamp()
                if doc_time > cutoff:
                    recent_count += 1
            except:
                pass
                
        health['recent_downloads'] = recent_count
        
        return health
    
    def cleanup(self) -> Dict[str, int]:
        """Clean up invalid documents and optimize storage"""
        self.logger.info("Starting Layer 1 cleanup")
        
        # Revalidate all documents
        validation_stats = self.document_storage.revalidate_all_documents()
        
        # Remove invalid documents
        removed_count = self.document_storage.cleanup_invalid_documents()
        
        cleanup_stats = {
            'revalidated': sum(validation_stats.values()),
            'removed_invalid': removed_count,
            'final_valid': validation_stats.get('valid', 0)
        }
        
        self.logger.info(f"Layer 1 cleanup complete: {cleanup_stats}")
        return cleanup_stats
