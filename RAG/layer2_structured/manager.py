"""
Structured Data Layer Manager
=============================

Manages Layer 2 of the RAG system - structured financial data processing.
Orchestrates financial data parsing, DuckDB storage, and Parquet archival.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date

from .financial_parser import FinancialDataParser, FinancialMetric, CompanyProfile
from .duckdb_manager import DuckDBManager
from .parquet_storage import ParquetStorageManager
from ..config import RAGConfig
from ..utils import setup_logging, progress_bar, ProgressTracker
from ..layer1_raw import RawSourceManager

class StructuredDataManager:
    """Manager for Layer 2 Structured Data - Financial metrics and analysis"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Initialize components
        self.financial_parser = FinancialDataParser(config)
        self.duckdb_manager = DuckDBManager(config)
        self.parquet_storage = ParquetStorageManager(config)
        
        # Cache for processed documents
        self._processed_cache = set()
        
        self.logger.info("Structured Data Manager initialized")
    
    def initialize(self):
        """Initialize Layer 2 storage and components"""
        self.logger.info("Initializing Structured Data Layer (Layer 2)")
        
        # Validate configuration
        required_dirs = [
            self.config.PARQUET_DIR,
            self.config.DUCKDB_FILE.parent
        ]
        
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.duckdb_manager.initialize()
        self.parquet_storage.initialize()
        
        # Test functionality
        self._test_components()
        
        self.logger.info("Structured Data Layer initialization complete")
    
    def _test_components(self):
        """Test Layer 2 components functionality"""
        try:
            # Test DuckDB
            stats = self.duckdb_manager.get_database_stats()
            self.logger.info(f"DuckDB operational: {stats.get('financial_metrics_count', 0)} metrics")
            
            # Test Parquet storage
            parquet_stats = self.parquet_storage.get_storage_stats()
            self.logger.info(f"Parquet storage operational: {parquet_stats.get('total_files', 0)} files")
            
        except Exception as e:
            self.logger.warning(f"Component test warning: {e}")
    
    def process_raw_filings(self, raw_source_manager: RawSourceManager,
                           tickers: List[str] = None,
                           years: List[int] = None,
                           form_types: List[str] = None) -> Dict[str, Any]:
        """Process raw SEC filings into structured financial data"""
        if tickers is None:
            # Get all available tickers from raw storage
            all_docs = raw_source_manager.search_filings()
            tickers = list(set(doc.filing.ticker for doc in all_docs))
        
        if years is None:
            years = self.config.SP500_YEARS
        if form_types is None:
            form_types = self.config.FILING_TYPES
        
        self.logger.info(f"Processing filings for {len(tickers)} tickers, {len(years)} years")
        
        processing_stats = {
            'documents_processed': 0,
            'metrics_extracted': 0,
            'companies_profiled': 0,
            'errors': 0
        }
        
        total_expected = len(tickers) * len(years) * len(form_types)
        progress = ProgressTracker(total_expected, "Processing SEC Filings")
        
        # Process each ticker
        for ticker in tickers:
            try:
                # Get stored documents for this ticker
                ticker_docs = raw_source_manager.search_filings(
                    ticker=ticker, 
                    form_type=None, 
                    year=None
                )
                
                # Filter by years and form types
                relevant_docs = [
                    doc for doc in ticker_docs
                    if (doc.filing.filing_year in years and 
                        doc.filing.form_type in form_types)
                ]
                
                if not relevant_docs:
                    self.logger.debug(f"No relevant documents found for {ticker}")
                    progress.update(len(years) * len(form_types))
                    continue
                
                # Process company profile (use latest 10-K)
                profile_doc = next(
                    (doc for doc in relevant_docs if doc.filing.form_type == '10-K'),
                    relevant_docs[0] if relevant_docs else None
                )
                
                if profile_doc:
                    self._process_company_profile(profile_doc, raw_source_manager)
                    processing_stats['companies_profiled'] += 1
                
                # Process each document for financial metrics
                for doc in relevant_docs:
                    try:
                        metrics_count = self._process_document_metrics(doc, raw_source_manager)
                        processing_stats['metrics_extracted'] += metrics_count
                        processing_stats['documents_processed'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {doc.filing.ticker} document: {e}")
                        processing_stats['errors'] += 1
                    
                    progress.update()
                
            except Exception as e:
                self.logger.error(f"Error processing ticker {ticker}: {e}")
                processing_stats['errors'] += 1
                progress.update(len(years) * len(form_types))
        
        progress.finish()
        
        # Store processed data in Parquet for archival
        self._archive_to_parquet()
        
        processing_stats['layer_stats'] = self.get_layer_stats()
        
        self.logger.info(f"Processing complete: {processing_stats}")
        return processing_stats
    
    def _process_company_profile(self, doc, raw_source_manager) -> bool:
        """Process company profile from a filing document"""
        try:
            # Get document content
            content = raw_source_manager.get_filing_content(
                doc.filing.ticker, 
                doc.filing.form_type, 
                doc.filing.accession_number
            )
            
            if not content:
                return False
            
            # Parse company profile
            profile = self.financial_parser.parse_company_profile(
                content, doc.filing.ticker, doc.filing.cik
            )
            
            # Store in DuckDB
            success = self.duckdb_manager.insert_company_profile(profile)
            
            if success:
                self.logger.debug(f"Processed company profile for {doc.filing.ticker}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing company profile: {e}")
            return False
    
    def _process_document_metrics(self, doc, raw_source_manager) -> int:
        """Process financial metrics from a document"""
        try:
            # Check if already processed
            doc_key = f"{doc.filing.ticker}_{doc.filing.accession_number}"
            if doc_key in self._processed_cache:
                return 0
            
            # Get document content
            content = raw_source_manager.get_filing_content(
                doc.filing.ticker,
                doc.filing.form_type,
                doc.filing.accession_number
            )
            
            if not content:
                return 0
            
            # Parse financial metrics
            metrics = self.financial_parser.parse_filing_text(
                content,
                doc.filing.ticker,
                doc.filing.form_type,
                doc.filing.filing_date,
                doc.filing.accession_number
            )
            
            if not metrics:
                return 0
            
            # Store in DuckDB
            stored_count = self.duckdb_manager.insert_financial_metrics(metrics)
            
            # Add to processed cache
            self._processed_cache.add(doc_key)
            
            return stored_count
            
        except Exception as e:
            self.logger.error(f"Error processing document metrics: {e}")
            return 0
    
    def _archive_to_parquet(self):
        """Archive DuckDB data to Parquet for long-term storage"""
        try:
            self.logger.info("Archiving data to Parquet storage")
            
            # Export financial metrics
            metrics_df = self.duckdb_manager.connection.execute(
                "SELECT * FROM financial_metrics"
            ).df()
            
            if not metrics_df.empty:
                # Convert back to FinancialMetric objects for storage
                metrics = []
                for _, row in metrics_df.iterrows():
                    metric = FinancialMetric(
                        ticker=row['ticker'],
                        company_name=row['company_name'],
                        metric_name=row['metric_name'],
                        metric_value=row['metric_value'],
                        metric_unit=row['metric_unit'],
                        period_type=row['period_type'],
                        period_end_date=str(row['period_end_date']),
                        filing_date=str(row['filing_date']),
                        form_type=row['form_type'],
                        accession_number=row['accession_number'],
                        extraction_timestamp=str(row['extraction_timestamp'])
                    )
                    metrics.append(metric)
                
                self.parquet_storage.store_financial_metrics(metrics, partition_by='year_ticker')
            
            # Export company profiles
            profiles_df = self.duckdb_manager.connection.execute(
                "SELECT * FROM company_profiles"
            ).df()
            
            if not profiles_df.empty:
                profiles = []
                for _, row in profiles_df.iterrows():
                    profile = CompanyProfile(
                        ticker=row['ticker'],
                        cik=row['cik'],
                        company_name=row['company_name'],
                        industry=row.get('industry'),
                        sector=row.get('sector'),
                        sic_code=row.get('sic_code'),
                        employees=row.get('employees'),
                        headquarters=row.get('headquarters'),
                        business_description=row.get('business_description'),
                        fiscal_year_end=row.get('fiscal_year_end')
                    )
                    profiles.append(profile)
                
                self.parquet_storage.store_company_profiles(profiles)
            
            self.logger.info("Parquet archival complete")
            
        except Exception as e:
            self.logger.error(f"Error archiving to Parquet: {e}")
    
    def get_company_financials(self, ticker: str, 
                              start_date: str = None, 
                              end_date: str = None,
                              metrics: List[str] = None) -> pd.DataFrame:
        """Get financial data for a company"""
        return self.duckdb_manager.get_company_metrics(
            ticker, start_date, end_date, metrics
        )
    
    def get_sector_analysis(self, sector: str, metric: str, 
                           year: int = None) -> pd.DataFrame:
        """Get sector-wide analysis"""
        return self.duckdb_manager.get_sector_analysis(sector, metric, year)
    
    def get_time_series(self, ticker: str, metric: str, 
                       period_type: str = 'annual') -> pd.DataFrame:
        """Get time series for a metric"""
        return self.duckdb_manager.get_time_series(ticker, metric, period_type)
    
    def calculate_financial_ratios(self, ticker: str, 
                                  date: str = None) -> Dict[str, float]:
        """Calculate financial ratios for a company"""
        return self.duckdb_manager.calculate_financial_ratios(ticker, date)
    
    def run_custom_analysis(self, sql_query: str) -> pd.DataFrame:
        """Run custom SQL analysis on financial data"""
        return self.duckdb_manager.execute_custom_query(sql_query)
    
    def export_data(self, export_type: str, ticker: str = None, 
                   output_path: Path = None) -> bool:
        """Export data in various formats"""
        try:
            if export_type == 'csv':
                # Export financial metrics to CSV
                if ticker:
                    df = self.get_company_financials(ticker)
                    if output_path is None:
                        output_path = self.config.PARQUET_DIR / f'{ticker}_financials.csv'
                else:
                    df = self.duckdb_manager.connection.execute(
                        "SELECT * FROM financial_metrics"
                    ).df()
                    if output_path is None:
                        output_path = self.config.PARQUET_DIR / 'all_financials.csv'
                
                df.to_csv(output_path, index=False)
                return True
                
            elif export_type == 'parquet_backup':
                # Create backup of all Parquet data
                backup_dir = self.config.DATA_DIR / 'parquet_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
                
                for partition_name in self.parquet_storage.partitions.keys():
                    self.parquet_storage.backup_partition(partition_name, backup_dir)
                
                return True
                
            else:
                raise ValueError(f"Unsupported export type: {export_type}")
                
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False
    
    def optimize_storage(self) -> Dict[str, Any]:
        """Optimize both DuckDB and Parquet storage"""
        results = {}
        
        try:
            # Optimize Parquet storage
            parquet_results = self.parquet_storage.optimize_storage()
            results['parquet_optimization'] = parquet_results
            
            # Optimize DuckDB (analyze tables for better query performance)
            self.duckdb_manager.connection.execute("ANALYZE financial_metrics")
            self.duckdb_manager.connection.execute("ANALYZE company_profiles")
            results['duckdb_optimization'] = {'status': 'completed'}
            
            self.logger.info("Storage optimization complete")
            
        except Exception as e:
            self.logger.error(f"Error optimizing storage: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_layer_stats(self) -> Dict[str, Any]:
        """Get comprehensive Layer 2 statistics"""
        try:
            # DuckDB stats
            duckdb_stats = self.duckdb_manager.get_database_stats()
            
            # Parquet stats
            parquet_stats = self.parquet_storage.get_storage_stats()
            
            # Processing stats
            processing_stats = {
                'processed_documents_cache_size': len(self._processed_cache),
                'available_metrics': list(self.financial_parser.all_patterns.keys())
            }
            
            return {
                'layer': 'Layer 2 - Structured Data',
                'duckdb_stats': duckdb_stats,
                'parquet_stats': parquet_stats,
                'processing_stats': processing_stats,
                'health_status': self._health_check()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _health_check(self) -> Dict[str, Any]:
        """Perform health check on Layer 2 components"""
        health = {
            'duckdb_accessible': False,
            'parquet_storage_accessible': False,
            'data_consistency': False,
            'recent_processing': False
        }
        
        try:
            # Test DuckDB
            test_result = self.duckdb_manager.connection.execute("SELECT 1").fetchone()
            health['duckdb_accessible'] = test_result is not None
            
            # Test Parquet storage
            storage_stats = self.parquet_storage.get_storage_stats()
            health['parquet_storage_accessible'] = 'error' not in storage_stats
            
            # Check data consistency
            duckdb_count = self.duckdb_manager.connection.execute(
                "SELECT COUNT(*) FROM financial_metrics"
            ).fetchone()[0]
            
            # Simple consistency check - DuckDB should have data if we've processed anything
            health['data_consistency'] = duckdb_count >= 0
            
            # Check for recent processing
            health['recent_processing'] = len(self._processed_cache) > 0
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
        
        return health
    
    def cleanup(self) -> Dict[str, Any]:
        """Clean up Layer 2 storage and optimize"""
        self.logger.info("Starting Layer 2 cleanup")
        
        cleanup_stats = {
            'cache_cleared': len(self._processed_cache),
            'optimization_results': {},
            'storage_stats_before': self.get_layer_stats(),
        }
        
        # Clear processing cache
        self._processed_cache.clear()
        
        # Optimize storage
        cleanup_stats['optimization_results'] = self.optimize_storage()
        
        # Final stats
        cleanup_stats['storage_stats_after'] = self.get_layer_stats()
        
        self.logger.info(f"Layer 2 cleanup complete: {cleanup_stats['cache_cleared']} cache entries cleared")
        return cleanup_stats
    
    def close(self):
        """Close all connections and cleanup"""
        try:
            self.duckdb_manager.close()
            self.logger.info("Layer 2 connections closed")
        except Exception as e:
            self.logger.error(f"Error closing Layer 2: {e}")
