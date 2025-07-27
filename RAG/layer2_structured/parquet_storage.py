"""
Parquet Storage Manager
======================

Manages Parquet file storage for efficient data archival and retrieval.
Provides columnar storage with compression and partitioning.
"""

import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date
import json

from .financial_parser import FinancialMetric, CompanyProfile
from ..config import RAGConfig
from ..utils import setup_logging, get_current_timestamp, format_bytes

class ParquetStorageManager:
    """Manages Parquet storage for financial data"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Storage paths
        self.parquet_dir = config.PARQUET_DIR
        
        # Partition structure
        self.partitions = {
            'financial_metrics': self.parquet_dir / 'financial_metrics',
            'company_profiles': self.parquet_dir / 'company_profiles',
            'time_series': self.parquet_dir / 'time_series',
            'ratios': self.parquet_dir / 'ratios'
        }
        
        # Parquet write options
        self.write_options = {
            'compression': 'snappy',
            'row_group_size': 50000,
            'use_dictionary': True,
            'write_statistics': True
        }
        
        # Initialize storage
        self.initialize()
    
    def initialize(self):
        """Initialize Parquet storage directories"""
        # Create partition directories
        for partition_name, partition_path in self.partitions.items():
            partition_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created partition directory: {partition_name}")
        
        self.logger.info(f"Parquet storage initialized at {self.parquet_dir}")
    
    def store_financial_metrics(self, metrics: List[FinancialMetric], 
                               partition_by: str = 'year') -> bool:
        """Store financial metrics with partitioning"""
        if not metrics:
            return True
        
        try:
            # Convert to DataFrame
            data = [metric.to_dict() for metric in metrics]
            df = pd.DataFrame(data)
            
            # Add partition column
            df['period_end_date'] = pd.to_datetime(df['period_end_date'])
            
            if partition_by == 'year':
                df['partition_year'] = df['period_end_date'].dt.year
                partition_cols = ['partition_year']
            elif partition_by == 'ticker':
                partition_cols = ['ticker']
            elif partition_by == 'year_ticker':
                df['partition_year'] = df['period_end_date'].dt.year
                partition_cols = ['partition_year', 'ticker']
            else:
                partition_cols = None
            
            # Convert to PyArrow table
            table = pa.Table.from_pandas(df)
            
            # Write partitioned dataset
            base_path = self.partitions['financial_metrics']
            
            pq.write_to_dataset(
                table,
                root_path=str(base_path),
                partition_cols=partition_cols,
                **self.write_options
            )
            
            self.logger.info(f"Stored {len(metrics)} financial metrics in Parquet format")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing financial metrics: {e}")
            return False
    
    def store_company_profiles(self, profiles: List[CompanyProfile]) -> bool:
        """Store company profiles"""
        if not profiles:
            return True
        
        try:
            # Convert to DataFrame
            data = [profile.to_dict() for profile in profiles]
            df = pd.DataFrame(data)
            
            # Write to single file (profiles are relatively small)
            file_path = self.partitions['company_profiles'] / 'company_profiles.parquet'
            
            df.to_parquet(
                file_path,
                engine='pyarrow',
                **self.write_options
            )
            
            self.logger.info(f"Stored {len(profiles)} company profiles")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing company profiles: {e}")
            return False
    
    def store_time_series_data(self, ticker: str, metric_name: str, 
                              time_series_data: pd.DataFrame) -> bool:
        """Store time series data for specific ticker/metric combinations"""
        try:
            # Add metadata columns
            time_series_data = time_series_data.copy()
            time_series_data['ticker'] = ticker
            time_series_data['metric_name'] = metric_name
            time_series_data['stored_at'] = datetime.now()
            
            # Partition by ticker
            partition_dir = self.partitions['time_series'] / f'ticker={ticker}'
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = partition_dir / f'{metric_name}_time_series.parquet'
            
            time_series_data.to_parquet(
                file_path,
                engine='pyarrow',
                **self.write_options
            )
            
            self.logger.debug(f"Stored time series for {ticker}:{metric_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing time series data: {e}")
            return False
    
    def store_financial_ratios(self, ratios_data: pd.DataFrame) -> bool:
        """Store calculated financial ratios"""
        try:
            # Add timestamp
            ratios_data = ratios_data.copy()
            ratios_data['calculated_at'] = datetime.now()
            
            # Partition by year if period_end_date exists
            if 'period_end_date' in ratios_data.columns:
                ratios_data['period_end_date'] = pd.to_datetime(ratios_data['period_end_date'])
                ratios_data['partition_year'] = ratios_data['period_end_date'].dt.year
                partition_cols = ['partition_year']
            else:
                partition_cols = None
            
            # Convert to PyArrow table
            table = pa.Table.from_pandas(ratios_data)
            
            # Write partitioned dataset
            base_path = self.partitions['ratios']
            
            pq.write_to_dataset(
                table,
                root_path=str(base_path),
                partition_cols=partition_cols,
                **self.write_options
            )
            
            self.logger.info(f"Stored {len(ratios_data)} financial ratios")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing financial ratios: {e}")
            return False
    
    def read_financial_metrics(self, filters: List[Tuple] = None,
                              columns: List[str] = None) -> pd.DataFrame:
        """Read financial metrics with optional filters"""
        try:
            dataset_path = self.partitions['financial_metrics']
            
            if not any(dataset_path.iterdir()):
                return pd.DataFrame()
            
            # Create dataset
            dataset = pq.ParquetDataset(str(dataset_path))
            
            # Apply filters and read
            table = dataset.read(
                filters=filters,
                columns=columns,
                use_pandas_metadata=True
            )
            
            return table.to_pandas()
            
        except Exception as e:
            self.logger.error(f"Error reading financial metrics: {e}")
            return pd.DataFrame()
    
    def read_company_profiles(self, tickers: List[str] = None) -> pd.DataFrame:
        """Read company profiles"""
        try:
            file_path = self.partitions['company_profiles'] / 'company_profiles.parquet'
            
            if not file_path.exists():
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            if tickers:
                df = df[df['ticker'].isin(tickers)]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading company profiles: {e}")
            return pd.DataFrame()
    
    def read_time_series(self, ticker: str, metric_name: str) -> pd.DataFrame:
        """Read time series data for specific ticker/metric"""
        try:
            file_path = (self.partitions['time_series'] / 
                        f'ticker={ticker}' / f'{metric_name}_time_series.parquet')
            
            if not file_path.exists():
                return pd.DataFrame()
            
            return pd.read_parquet(file_path, engine='pyarrow')
            
        except Exception as e:
            self.logger.error(f"Error reading time series: {e}")
            return pd.DataFrame()
    
    def read_financial_ratios(self, filters: List[Tuple] = None) -> pd.DataFrame:
        """Read financial ratios with optional filters"""
        try:
            dataset_path = self.partitions['ratios']
            
            if not any(dataset_path.iterdir()):
                return pd.DataFrame()
            
            dataset = pq.ParquetDataset(str(dataset_path))
            table = dataset.read(filters=filters, use_pandas_metadata=True)
            
            return table.to_pandas()
            
        except Exception as e:
            self.logger.error(f"Error reading financial ratios: {e}")
            return pd.DataFrame()
    
    def get_available_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data in Parquet storage"""
        summary = {}
        
        for partition_name, partition_path in self.partitions.items():
            try:
                if not partition_path.exists():
                    summary[partition_name] = {'exists': False}
                    continue
                
                partition_info = {
                    'exists': True,
                    'files': [],
                    'total_size': 0,
                    'row_count': 0
                }
                
                # Scan for parquet files
                parquet_files = list(partition_path.rglob('*.parquet'))
                
                for file_path in parquet_files:
                    file_size = file_path.stat().st_size
                    partition_info['total_size'] += file_size
                    
                    try:
                        # Get row count
                        parquet_file = pq.ParquetFile(file_path)
                        row_count = parquet_file.metadata.num_rows
                        partition_info['row_count'] += row_count
                        
                        partition_info['files'].append({
                            'path': str(file_path.relative_to(partition_path)),
                            'size': file_size,
                            'size_formatted': format_bytes(file_size),
                            'rows': row_count
                        })
                        
                    except Exception as e:
                        self.logger.debug(f"Could not read metadata for {file_path}: {e}")
                
                partition_info['total_size_formatted'] = format_bytes(partition_info['total_size'])
                summary[partition_name] = partition_info
                
            except Exception as e:
                summary[partition_name] = {'exists': False, 'error': str(e)}
        
        return summary
    
    def optimize_storage(self) -> Dict[str, Any]:
        """Optimize Parquet storage by compacting small files"""
        optimization_results = {}
        
        for partition_name, partition_path in self.partitions.items():
            try:
                if not partition_path.exists():
                    continue
                
                # Find small parquet files that could be combined
                parquet_files = list(partition_path.rglob('*.parquet'))
                small_files = [
                    f for f in parquet_files 
                    if f.stat().st_size < 10 * 1024 * 1024  # Files smaller than 10MB
                ]
                
                if len(small_files) > 3:  # Only optimize if there are multiple small files
                    self.logger.info(f"Optimizing {len(small_files)} small files in {partition_name}")
                    
                    # Read all small files
                    dfs = []
                    for file_path in small_files:
                        df = pd.read_parquet(file_path, engine='pyarrow')
                        dfs.append(df)
                    
                    if dfs:
                        # Combine into single DataFrame
                        combined_df = pd.concat(dfs, ignore_index=True)
                        
                        # Write optimized file
                        optimized_path = partition_path / 'optimized.parquet'
                        combined_df.to_parquet(
                            optimized_path,
                            engine='pyarrow',
                            **self.write_options
                        )
                        
                        # Remove original small files
                        for file_path in small_files:
                            file_path.unlink()
                        
                        optimization_results[partition_name] = {
                            'files_combined': len(small_files),
                            'rows_processed': len(combined_df),
                            'optimized_file': str(optimized_path)
                        }
                
            except Exception as e:
                self.logger.error(f"Error optimizing {partition_name}: {e}")
                optimization_results[partition_name] = {'error': str(e)}
        
        return optimization_results
    
    def export_to_csv(self, partition_name: str, output_path: Path = None) -> bool:
        """Export Parquet data to CSV for external analysis"""
        try:
            if partition_name not in self.partitions:
                raise ValueError(f"Unknown partition: {partition_name}")
            
            # Read data based on partition type
            if partition_name == 'financial_metrics':
                df = self.read_financial_metrics()
            elif partition_name == 'company_profiles':
                df = self.read_company_profiles()
            elif partition_name == 'ratios':
                df = self.read_financial_ratios()
            else:
                return False
            
            if df.empty:
                self.logger.warning(f"No data found in {partition_name}")
                return False
            
            # Determine output path
            if output_path is None:
                output_path = self.parquet_dir / f'{partition_name}_export.csv'
            
            # Export to CSV
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Exported {len(df)} rows from {partition_name} to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting {partition_name} to CSV: {e}")
            return False
    
    def backup_partition(self, partition_name: str, backup_dir: Path) -> bool:
        """Create backup of a partition"""
        try:
            if partition_name not in self.partitions:
                raise ValueError(f"Unknown partition: {partition_name}")
            
            source_path = self.partitions[partition_name]
            backup_path = backup_dir / partition_name
            
            if not source_path.exists():
                self.logger.warning(f"Source partition {partition_name} does not exist")
                return False
            
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy all parquet files
            parquet_files = list(source_path.rglob('*.parquet'))
            
            for source_file in parquet_files:
                relative_path = source_file.relative_to(source_path)
                backup_file = backup_path / relative_path
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                import shutil
                shutil.copy2(source_file, backup_file)
            
            self.logger.info(f"Backed up {len(parquet_files)} files from {partition_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error backing up {partition_name}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        try:
            stats = {
                'partitions': {},
                'total_size': 0,
                'total_files': 0,
                'total_rows': 0
            }
            
            summary = self.get_available_data_summary()
            
            for partition_name, partition_info in summary.items():
                if partition_info.get('exists', False):
                    stats['partitions'][partition_name] = {
                        'files': len(partition_info.get('files', [])),
                        'size': partition_info.get('total_size', 0),
                        'size_formatted': partition_info.get('total_size_formatted', '0 B'),
                        'rows': partition_info.get('row_count', 0)
                    }
                    
                    stats['total_size'] += partition_info.get('total_size', 0)
                    stats['total_files'] += len(partition_info.get('files', []))
                    stats['total_rows'] += partition_info.get('row_count', 0)
            
            stats['total_size_formatted'] = format_bytes(stats['total_size'])
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}
