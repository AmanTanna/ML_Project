"""
DuckDB Integration
==================

Fast columnar database for financial data analysis.
Provides SQL interface to structured financial metrics.
"""

import logging
import duckdb
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date

from .financial_parser import FinancialMetric, CompanyProfile
from ..config import RAGConfig
from ..utils import setup_logging, get_current_timestamp

class DuckDBManager:
    """Manages DuckDB database for structured financial data"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Database connection
        self.db_path = config.DUCKDB_FILE
        self.connection = None
        
        # Table schemas
        self.schemas = {
            'financial_metrics': '''
                CREATE TABLE IF NOT EXISTS financial_metrics (
                    ticker VARCHAR(10) NOT NULL,
                    company_name VARCHAR(255),
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DOUBLE,
                    metric_unit VARCHAR(20),
                    period_type VARCHAR(20),
                    period_end_date DATE,
                    filing_date DATE,
                    form_type VARCHAR(10),
                    accession_number VARCHAR(50),
                    extraction_timestamp TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'company_profiles': '''
                CREATE TABLE IF NOT EXISTS company_profiles (
                    ticker VARCHAR(10) PRIMARY KEY,
                    cik VARCHAR(20),
                    company_name VARCHAR(255),
                    industry VARCHAR(100),
                    sector VARCHAR(50),
                    sic_code VARCHAR(10),
                    employees INTEGER,
                    headquarters VARCHAR(255),
                    business_description TEXT,
                    fiscal_year_end VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'metric_definitions': '''
                CREATE TABLE IF NOT EXISTS metric_definitions (
                    metric_name VARCHAR(100) PRIMARY KEY,
                    display_name VARCHAR(150),
                    category VARCHAR(50),
                    description TEXT,
                    unit VARCHAR(20),
                    calculation_method TEXT
                )
            '''
        }
        
        # Initialize database
        self.initialize()
    
    def initialize(self):
        """Initialize DuckDB database and tables"""
        try:
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.connection = duckdb.connect(str(self.db_path))
            
            # Create tables
            for table_name, schema in self.schemas.items():
                self.connection.execute(schema)
                self.logger.debug(f"Created/verified table: {table_name}")
            
            # Create indexes for performance
            self._create_indexes()
            
            # Insert metric definitions
            self._insert_metric_definitions()
            
            self.logger.info(f"DuckDB initialized at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Error initializing DuckDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for better query performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_metrics_ticker ON financial_metrics(ticker)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_date ON financial_metrics(period_end_date)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_name ON financial_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_form ON financial_metrics(form_type)",
            "CREATE INDEX IF NOT EXISTS idx_profiles_sector ON company_profiles(sector)",
            "CREATE INDEX IF NOT EXISTS idx_profiles_industry ON company_profiles(industry)"
        ]
        
        for index_sql in indexes:
            try:
                self.connection.execute(index_sql)
            except Exception as e:
                self.logger.debug(f"Index creation note: {e}")
    
    def _insert_metric_definitions(self):
        """Insert standardized metric definitions"""
        definitions = [
            ('revenue', 'Total Revenue', 'Income Statement', 'Total revenue from operations', 'USD', 'Sum of all revenue sources'),
            ('gross_profit', 'Gross Profit', 'Income Statement', 'Revenue minus cost of goods sold', 'USD', 'Revenue - COGS'),
            ('operating_income', 'Operating Income', 'Income Statement', 'Income from core business operations', 'USD', 'Gross Profit - Operating Expenses'),
            ('net_income', 'Net Income', 'Income Statement', 'Final profit after all expenses', 'USD', 'Operating Income - Interest - Taxes'),
            ('ebitda', 'EBITDA', 'Income Statement', 'Earnings before interest, taxes, depreciation, amortization', 'USD', 'Net Income + Interest + Taxes + Depreciation + Amortization'),
            ('total_assets', 'Total Assets', 'Balance Sheet', 'Sum of all company assets', 'USD', 'Current Assets + Non-Current Assets'),
            ('total_liabilities', 'Total Liabilities', 'Balance Sheet', 'Sum of all company liabilities', 'USD', 'Current Liabilities + Long-term Liabilities'),
            ('shareholders_equity', 'Shareholders Equity', 'Balance Sheet', 'Owners stake in the company', 'USD', 'Total Assets - Total Liabilities'),
            ('cash_and_equivalents', 'Cash & Equivalents', 'Balance Sheet', 'Highly liquid assets', 'USD', 'Cash + Short-term Investments'),
            ('total_debt', 'Total Debt', 'Balance Sheet', 'All interest-bearing obligations', 'USD', 'Short-term Debt + Long-term Debt'),
            ('operating_cash_flow', 'Operating Cash Flow', 'Cash Flow', 'Cash generated from operations', 'USD', 'Net Income + Non-cash Items + Working Capital Changes'),
            ('investing_cash_flow', 'Investing Cash Flow', 'Cash Flow', 'Cash used in investment activities', 'USD', 'Capital Expenditures + Acquisitions - Disposals'),
            ('financing_cash_flow', 'Financing Cash Flow', 'Cash Flow', 'Cash from financing activities', 'USD', 'Debt Issuance - Debt Repayment + Equity Changes'),
            ('free_cash_flow', 'Free Cash Flow', 'Cash Flow', 'Cash available after necessary investments', 'USD', 'Operating Cash Flow - Capital Expenditures')
        ]
        
        # Check if definitions already exist
        existing_count = self.connection.execute(
            "SELECT COUNT(*) FROM metric_definitions"
        ).fetchone()[0]
        
        if existing_count == 0:
            insert_sql = """
                INSERT INTO metric_definitions 
                (metric_name, display_name, category, description, unit, calculation_method)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            
            for definition in definitions:
                self.connection.execute(insert_sql, definition)
            
            self.logger.info(f"Inserted {len(definitions)} metric definitions")
    
    def insert_financial_metrics(self, metrics: List[FinancialMetric]) -> int:
        """Insert financial metrics into database"""
        if not metrics:
            return 0
        
        # Convert to DataFrame for bulk insert
        data = [metric.to_dict() for metric in metrics]
        df = pd.DataFrame(data)
        
        # Convert date columns
        df['period_end_date'] = pd.to_datetime(df['period_end_date'])
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        df['extraction_timestamp'] = pd.to_datetime(df['extraction_timestamp'])
        
        try:
            # Use DuckDB's DataFrame integration
            self.connection.execute("BEGIN TRANSACTION")
            
            # Insert data
            self.connection.register('temp_metrics', df)
            insert_sql = """
                INSERT INTO financial_metrics 
                (ticker, company_name, metric_name, metric_value, metric_unit, 
                 period_type, period_end_date, filing_date, form_type, 
                 accession_number, extraction_timestamp)
                SELECT ticker, company_name, metric_name, metric_value, metric_unit,
                       period_type, period_end_date, filing_date, form_type,
                       accession_number, extraction_timestamp
                FROM temp_metrics
            """
            
            result = self.connection.execute(insert_sql)
            self.connection.execute("COMMIT")
            
            inserted_count = len(metrics)
            self.logger.info(f"Inserted {inserted_count} financial metrics")
            
            return inserted_count
            
        except Exception as e:
            self.connection.execute("ROLLBACK")
            self.logger.error(f"Error inserting metrics: {e}")
            return 0
    
    def insert_company_profile(self, profile: CompanyProfile) -> bool:
        """Insert or update company profile"""
        try:
            # Use UPSERT (INSERT OR REPLACE in DuckDB)
            sql = """
                INSERT OR REPLACE INTO company_profiles
                (ticker, cik, company_name, industry, sector, sic_code, 
                 employees, headquarters, business_description, fiscal_year_end, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            
            values = (
                profile.ticker, profile.cik, profile.company_name,
                profile.industry, profile.sector, profile.sic_code,
                profile.employees, profile.headquarters,
                profile.business_description, profile.fiscal_year_end
            )
            
            self.connection.execute(sql, values)
            self.logger.debug(f"Upserted company profile for {profile.ticker}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error inserting company profile: {e}")
            return False
    
    def get_company_metrics(self, ticker: str, 
                           start_date: str = None, end_date: str = None,
                           metric_names: List[str] = None) -> pd.DataFrame:
        """Get financial metrics for a company"""
        conditions = [f"ticker = '{ticker.upper()}'"]
        
        if start_date:
            conditions.append(f"period_end_date >= '{start_date}'")
        if end_date:
            conditions.append(f"period_end_date <= '{end_date}'")
        if metric_names:
            metric_list = "', '".join(metric_names)
            conditions.append(f"metric_name IN ('{metric_list}')")
        
        where_clause = " AND ".join(conditions)
        
        sql = f"""
            SELECT 
                fm.*,
                md.display_name,
                md.category,
                md.description
            FROM financial_metrics fm
            LEFT JOIN metric_definitions md ON fm.metric_name = md.metric_name
            WHERE {where_clause}
            ORDER BY period_end_date DESC, metric_name
        """
        
        try:
            return self.connection.execute(sql).df()
        except Exception as e:
            self.logger.error(f"Error querying company metrics: {e}")
            return pd.DataFrame()
    
    def get_sector_analysis(self, sector: str, metric_name: str, 
                           year: int = None) -> pd.DataFrame:
        """Get sector-wide analysis for a specific metric"""
        conditions = [
            f"cp.sector = '{sector}'",
            f"fm.metric_name = '{metric_name}'"
        ]
        
        if year:
            conditions.append(f"EXTRACT(YEAR FROM fm.period_end_date) = {year}")
        
        where_clause = " AND ".join(conditions)
        
        sql = f"""
            SELECT 
                cp.ticker,
                cp.company_name,
                cp.industry,
                fm.metric_value,
                fm.period_end_date,
                fm.form_type,
                RANK() OVER (ORDER BY fm.metric_value DESC) as rank_in_sector
            FROM company_profiles cp
            JOIN financial_metrics fm ON cp.ticker = fm.ticker
            WHERE {where_clause}
            ORDER BY fm.metric_value DESC
        """
        
        try:
            return self.connection.execute(sql).df()
        except Exception as e:
            self.logger.error(f"Error in sector analysis: {e}")
            return pd.DataFrame()
    
    def get_time_series(self, ticker: str, metric_name: str, 
                       period_type: str = 'annual') -> pd.DataFrame:
        """Get time series data for a specific metric"""
        sql = """
            SELECT 
                period_end_date,
                metric_value,
                form_type,
                filing_date
            FROM financial_metrics
            WHERE ticker = ? AND metric_name = ? AND period_type = ?
            ORDER BY period_end_date ASC
        """
        
        try:
            return self.connection.execute(sql, [ticker.upper(), metric_name, period_type]).df()
        except Exception as e:
            self.logger.error(f"Error getting time series: {e}")
            return pd.DataFrame()
    
    def calculate_financial_ratios(self, ticker: str, 
                                  period_end_date: str = None) -> Dict[str, float]:
        """Calculate common financial ratios"""
        # Get latest metrics if no date specified
        date_condition = f"AND period_end_date = '{period_end_date}'" if period_end_date else ""
        
        sql = f"""
            SELECT 
                metric_name,
                metric_value
            FROM financial_metrics
            WHERE ticker = '{ticker.upper()}'
            {date_condition}
            ORDER BY period_end_date DESC
        """
        
        try:
            df = self.connection.execute(sql).df()
            
            if df.empty:
                return {}
            
            # Convert to dictionary for easier access
            metrics = dict(zip(df['metric_name'], df['metric_value']))
            
            ratios = {}
            
            # Profitability ratios
            if 'gross_profit' in metrics and 'revenue' in metrics and metrics['revenue']:
                ratios['gross_margin'] = metrics['gross_profit'] / metrics['revenue']
            
            if 'operating_income' in metrics and 'revenue' in metrics and metrics['revenue']:
                ratios['operating_margin'] = metrics['operating_income'] / metrics['revenue']
            
            if 'net_income' in metrics and 'revenue' in metrics and metrics['revenue']:
                ratios['net_margin'] = metrics['net_income'] / metrics['revenue']
            
            # Leverage ratios
            if 'total_debt' in metrics and 'shareholders_equity' in metrics and metrics['shareholders_equity']:
                ratios['debt_to_equity'] = metrics['total_debt'] / metrics['shareholders_equity']
            
            if 'total_debt' in metrics and 'total_assets' in metrics and metrics['total_assets']:
                ratios['debt_to_assets'] = metrics['total_debt'] / metrics['total_assets']
            
            # Efficiency ratios
            if 'net_income' in metrics and 'total_assets' in metrics and metrics['total_assets']:
                ratios['roa'] = metrics['net_income'] / metrics['total_assets']  # Return on Assets
            
            if 'net_income' in metrics and 'shareholders_equity' in metrics and metrics['shareholders_equity']:
                ratios['roe'] = metrics['net_income'] / metrics['shareholders_equity']  # Return on Equity
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating ratios: {e}")
            return {}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}
            
            # Table counts
            for table in ['financial_metrics', 'company_profiles', 'metric_definitions']:
                count = self.connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                stats[f"{table}_count"] = count
            
            # Unique companies
            unique_companies = self.connection.execute(
                "SELECT COUNT(DISTINCT ticker) FROM financial_metrics"
            ).fetchone()[0]
            stats['unique_companies'] = unique_companies
            
            # Date range
            date_range = self.connection.execute("""
                SELECT 
                    MIN(period_end_date) as earliest_date,
                    MAX(period_end_date) as latest_date
                FROM financial_metrics
            """).fetchone()
            
            if date_range[0]:
                stats['date_range'] = {
                    'earliest': str(date_range[0]),
                    'latest': str(date_range[1])
                }
            
            # Metrics by category
            category_stats = self.connection.execute("""
                SELECT 
                    md.category,
                    COUNT(DISTINCT fm.metric_name) as metric_count,
                    COUNT(fm.id) as total_values
                FROM metric_definitions md
                LEFT JOIN financial_metrics fm ON md.metric_name = fm.metric_name
                GROUP BY md.category
            """).df()
            
            stats['metrics_by_category'] = category_stats.to_dict('records')
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}
    
    def execute_custom_query(self, sql: str) -> pd.DataFrame:
        """Execute custom SQL query (read-only)"""
        # Security check - only allow SELECT statements
        sql_clean = sql.strip().upper()
        if not sql_clean.startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed")
        
        try:
            return self.connection.execute(sql).df()
        except Exception as e:
            self.logger.error(f"Error executing custom query: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.logger.info("DuckDB connection closed")
