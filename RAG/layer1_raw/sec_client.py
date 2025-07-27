"""
SEC EDGAR API Client
====================

Client for fetching S&P 500 company filings from SEC EDGAR database.
Handles rate limiting, retries, and proper SEC compliance.
"""

import time
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, date
import json
import re

from ..config import RAGConfig
from ..utils import setup_logging, safe_filename, get_current_timestamp

@dataclass
class Filing:
    """Represents a single SEC filing"""
    cik: str
    ticker: str
    company_name: str
    form_type: str  # 10-K, 10-Q, etc.
    filing_date: str
    accession_number: str
    document_url: str
    html_url: Optional[str] = None
    filing_year: int = None
    
    def __post_init__(self):
        if self.filing_year is None and self.filing_date:
            self.filing_year = int(self.filing_date.split('-')[0])

@dataclass
class CompanyInfo:
    """S&P 500 company information"""
    cik: str
    ticker: str
    company_name: str
    sic: Optional[str] = None
    industry: Optional[str] = None

class SECClient:
    """SEC EDGAR API client with rate limiting and compliance"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.USER_AGENT,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        
        # Cache for company info
        self._company_cache: Dict[str, CompanyInfo] = {}
        
    def _rate_limit(self):
        """Enforce SEC rate limiting (10 requests per second max)"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.config.REQUEST_DELAY:
            sleep_time = self.config.REQUEST_DELAY - time_since_last
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
        self.request_count += 1
        
        if self.request_count % 100 == 0:
            self.logger.info(f"Made {self.request_count} SEC API requests")
    
    def _make_request(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """Make a rate-limited request to SEC with retries"""
        self._rate_limit()
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                elif response.status_code == 404:
                    self.logger.warning(f"Document not found: {url}")
                    return None
                else:
                    self.logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.RequestException as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        return None
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for a company ticker"""
        # Try cache first
        if ticker in self._company_cache:
            return self._company_cache[ticker].cik
            
        # Fetch from SEC company tickers JSON
        url = "https://www.sec.gov/files/company_tickers.json"
        response = self._make_request(url)
        
        if not response:
            return None
            
        try:
            companies = response.json()
            for company_data in companies.values():
                if company_data.get('ticker', '').upper() == ticker.upper():
                    cik = str(company_data['cik_str']).zfill(10)
                    
                    # Cache the company info
                    self._company_cache[ticker] = CompanyInfo(
                        cik=cik,
                        ticker=ticker.upper(),
                        company_name=company_data.get('title', ''),
                        sic=str(company_data.get('sic', '')) if company_data.get('sic') else None
                    )
                    
                    return cik
                    
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Error parsing company tickers: {e}")
            
        return None
    
    def get_company_filings(self, cik: str, form_types: List[str] = None, 
                           start_date: str = None, end_date: str = None) -> List[Filing]:
        """Get filings for a company"""
        if form_types is None:
            form_types = self.config.FILING_TYPES
            
        filings = []
        
        # SEC submissions endpoint
        url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        response = self._make_request(url)
        
        if not response:
            return filings
            
        try:
            data = response.json()
            recent_filings = data.get('filings', {}).get('recent', {})
            
            if not recent_filings:
                return filings
                
            # Extract company info
            company_name = data.get('name', 'Unknown')
            ticker = data.get('tickers', [None])[0] if data.get('tickers') else None
            
            # Process filings
            forms = recent_filings.get('form', [])
            filing_dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            
            for i, form_type in enumerate(forms):
                if form_type in form_types:
                    filing_date = filing_dates[i]
                    accession = accession_numbers[i]
                    
                    # Date filtering
                    if start_date and filing_date < start_date:
                        continue
                    if end_date and filing_date > end_date:
                        continue
                        
                    # Build document URL
                    accession_clean = accession.replace('-', '')
                    document_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{accession}.txt"
                    html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{accession}-index.html"
                    
                    filing = Filing(
                        cik=cik,
                        ticker=ticker or 'UNKNOWN',
                        company_name=company_name,
                        form_type=form_type,
                        filing_date=filing_date,
                        accession_number=accession,
                        document_url=document_url,
                        html_url=html_url
                    )
                    
                    filings.append(filing)
                    
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            self.logger.error(f"Error parsing filings for CIK {cik}: {e}")
            
        return filings
    
    def download_filing_document(self, filing: Filing, save_path: Path) -> bool:
        """Download a filing document to local storage"""
        try:
            response = self._make_request(filing.document_url)
            if not response:
                return False
                
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save document
            with open(save_path, 'wb') as f:
                f.write(response.content)
                
            self.logger.debug(f"Downloaded {filing.form_type} for {filing.ticker}: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {filing.document_url}: {e}")
            return False
    
    def get_sp500_tickers(self) -> List[str]:
        """Get list of S&P 500 tickers (placeholder - you'd typically use a data source)"""
        # This is a sample - in production you'd fetch from a reliable source
        # like Wikipedia, Yahoo Finance, or a financial data provider
        sp500_sample = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM',
            'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'ADBE', 'CRM', 'NFLX',
            'KO', 'PFE', 'TMO', 'ABT', 'COST', 'AVGO', 'XOM', 'WMT', 'LLY',
            'CVX', 'MCD', 'ACN', 'ABBV', 'DHR', 'VZ', 'TXN', 'NEE', 'BMY',
            'CMCSA', 'HON', 'PM', 'ORCL', 'T', 'WFC', 'IBM', 'GE', 'AMD'
        ]
        
        self.logger.info(f"Using sample S&P 500 tickers: {len(sp500_sample)} companies")
        return sp500_sample
    
    def bulk_download_filings(self, tickers: List[str], years: List[int], 
                             form_types: List[str] = None) -> Dict[str, List[Filing]]:
        """Download filings for multiple companies and years"""
        if form_types is None:
            form_types = self.config.FILING_TYPES
            
        all_filings = {}
        total_tickers = len(tickers)
        
        self.logger.info(f"Starting bulk download for {total_tickers} tickers, years {years}")
        
        for i, ticker in enumerate(tickers):
            self.logger.info(f"Processing {ticker} ({i+1}/{total_tickers})")
            
            # Get CIK
            cik = self.get_company_cik(ticker)
            if not cik:
                self.logger.warning(f"Could not find CIK for {ticker}")
                continue
                
            # Get filings for each year
            ticker_filings = []
            for year in years:
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
                
                year_filings = self.get_company_filings(
                    cik, form_types, start_date, end_date
                )
                ticker_filings.extend(year_filings)
                
            all_filings[ticker] = ticker_filings
            
            # Progress update
            if (i + 1) % 10 == 0:
                total_filings = sum(len(f) for f in all_filings.values())
                self.logger.info(f"Progress: {i+1}/{total_tickers} tickers, {total_filings} filings found")
                
        total_filings = sum(len(f) for f in all_filings.values())
        self.logger.info(f"Bulk download complete: {total_filings} filings from {len(all_filings)} companies")
        
        return all_filings
