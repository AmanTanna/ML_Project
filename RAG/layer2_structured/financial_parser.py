"""
Financial Data Parser
====================

Extracts structured financial data from SEC filings (10-K, 10-Q).
Parses key financial metrics, ratios, and time series data.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, date
import pandas as pd

from ..config import RAGConfig
from ..utils import setup_logging, get_current_timestamp

@dataclass
class FinancialMetric:
    """Represents a single financial metric"""
    ticker: str
    company_name: str
    metric_name: str
    metric_value: Optional[float]
    metric_unit: str  # 'USD', 'shares', 'ratio', etc.
    period_type: str  # 'annual', 'quarterly'
    period_end_date: str
    filing_date: str
    form_type: str
    accession_number: str
    extraction_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)

@dataclass
class CompanyProfile:
    """Company profile and metadata"""
    ticker: str
    cik: str
    company_name: str
    industry: Optional[str] = None
    sector: Optional[str] = None
    sic_code: Optional[str] = None
    employees: Optional[int] = None
    headquarters: Optional[str] = None
    business_description: Optional[str] = None
    fiscal_year_end: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class FinancialDataParser:
    """Extracts structured financial data from SEC filings"""
    
    # Common financial statement patterns
    INCOME_STATEMENT_PATTERNS = {
        'revenue': [
            r'(?:total\s+)?(?:net\s+)?(?:revenues?|sales?|income\s+from\s+operations)',
            r'operating\s+revenues?',
            r'net\s+sales?'
        ],
        'gross_profit': [
            r'gross\s+profit',
            r'gross\s+margin'
        ],
        'operating_income': [
            r'(?:income|loss)\s+from\s+operations',
            r'operating\s+(?:income|loss)',
            r'earnings\s+from\s+operations'
        ],
        'net_income': [
            r'net\s+(?:income|loss|earnings)',
            r'net\s+(?:income|loss)\s+attributable\s+to',
            r'earnings\s+attributable\s+to'
        ],
        'ebitda': [
            r'earnings\s+before\s+interest,?\s+taxes?,?\s+depreciation\s+and\s+amortization',
            r'ebitda',
            r'adjusted\s+ebitda'
        ]
    }
    
    BALANCE_SHEET_PATTERNS = {
        'total_assets': [
            r'total\s+assets',
            r'total\s+consolidated\s+assets'
        ],
        'total_liabilities': [
            r'total\s+liabilities',
            r'total\s+consolidated\s+liabilities'
        ],
        'shareholders_equity': [
            r'(?:total\s+)?(?:shareholders?|stockholders?)\s+equity',
            r'total\s+equity'
        ],
        'cash_and_equivalents': [
            r'cash\s+and\s+cash\s+equivalents',
            r'cash\s+and\s+short.?term\s+investments'
        ],
        'total_debt': [
            r'total\s+debt',
            r'total\s+borrowings',
            r'long.?term\s+debt'
        ]
    }
    
    CASH_FLOW_PATTERNS = {
        'operating_cash_flow': [
            r'(?:net\s+)?cash\s+(?:provided\s+by|from)\s+operating\s+activities',
            r'operating\s+cash\s+flow'
        ],
        'investing_cash_flow': [
            r'(?:net\s+)?cash\s+(?:used\s+in|from)\s+investing\s+activities',
            r'investing\s+cash\s+flow'
        ],
        'financing_cash_flow': [
            r'(?:net\s+)?cash\s+(?:provided\s+by|used\s+in)\s+financing\s+activities',
            r'financing\s+cash\s+flow'
        ],
        'free_cash_flow': [
            r'free\s+cash\s+flow'
        ]
    }
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Combine all patterns
        self.all_patterns = {
            **self.INCOME_STATEMENT_PATTERNS,
            **self.BALANCE_SHEET_PATTERNS,
            **self.CASH_FLOW_PATTERNS
        }
        
        # Compile regex patterns for performance
        self._compiled_patterns = {}
        for metric, patterns in self.all_patterns.items():
            self._compiled_patterns[metric] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def parse_filing_text(self, filing_text: str, ticker: str, 
                         form_type: str, filing_date: str,
                         accession_number: str) -> List[FinancialMetric]:
        """Parse financial metrics from SEC filing text"""
        metrics = []
        
        # Clean and normalize text
        cleaned_text = self._clean_filing_text(filing_text)
        
        # Extract company information
        company_name = self._extract_company_name(cleaned_text, ticker)
        
        # Determine period type
        period_type = 'annual' if form_type == '10-K' else 'quarterly'
        
        # Extract period end date
        period_end_date = self._extract_period_end_date(cleaned_text, filing_date)
        
        # Extract each financial metric
        for metric_name, patterns in self._compiled_patterns.items():
            metric_value = self._extract_metric_value(cleaned_text, patterns, metric_name)
            
            if metric_value is not None:
                metric = FinancialMetric(
                    ticker=ticker,
                    company_name=company_name,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    metric_unit='USD',  # Assume USD for now
                    period_type=period_type,
                    period_end_date=period_end_date,
                    filing_date=filing_date,
                    form_type=form_type,
                    accession_number=accession_number,
                    extraction_timestamp=get_current_timestamp()
                )
                metrics.append(metric)
        
        self.logger.debug(f"Extracted {len(metrics)} metrics from {ticker} {form_type}")
        return metrics
    
    def _clean_filing_text(self, text: str) -> str:
        """Clean and normalize SEC filing text"""
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with parsing
        text = re.sub(r'[^\w\s\(\)\$\,\.\-\:]', '', text)
        
        return text.strip()
    
    def _extract_company_name(self, text: str, ticker: str) -> str:
        """Extract company name from filing text"""
        # Look for company name patterns
        patterns = [
            rf'({re.escape(ticker)})\s+([A-Z][A-Za-z\s&\.,]+?)(?:\s+(?:INC|CORP|LLC|LTD))',
            r'COMPANY\s+NAME:\s*([A-Z][A-Za-z\s&\.,]+)',
            r'REGISTRANT:\s*([A-Z][A-Za-z\s&\.,]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)  # Search in first 5KB
            if match:
                if len(match.groups()) > 1:
                    return match.group(2).strip()
                else:
                    return match.group(1).strip()
        
        return f"Company_{ticker}"
    
    def _extract_period_end_date(self, text: str, filing_date: str) -> str:
        """Extract period end date from filing"""
        # Look for period end date patterns
        patterns = [
            r'period\s+ended?\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:quarter|year)\s+ended?\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'as\s+of\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:10000], re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    # Try to parse and normalize date
                    parsed_date = pd.to_datetime(date_str)
                    return parsed_date.strftime('%Y-%m-%d')
                except:
                    continue
        
        # Default to filing date if can't extract period end
        return filing_date
    
    def _extract_metric_value(self, text: str, patterns: List[re.Pattern], 
                            metric_name: str) -> Optional[float]:
        """Extract numerical value for a financial metric"""
        # Search for pattern matches in the text
        for pattern in patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                # Look for numerical values near the match
                start_pos = max(0, match.start() - 500)
                end_pos = min(len(text), match.end() + 500)
                context = text[start_pos:end_pos]
                
                # Extract numerical values from context
                value = self._extract_number_from_context(context, metric_name)
                if value is not None:
                    return value
        
        return None
    
    def _extract_number_from_context(self, context: str, metric_name: str) -> Optional[float]:
        """Extract numerical value from text context"""
        # Look for various number formats
        number_patterns = [
            r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|mil|m)\b',  # Millions
            r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:billion|bil|b)\b',  # Billions
            r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:thousand|k)\b',     # Thousands
            r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b'                       # Raw numbers
        ]
        
        multipliers = {
            'million': 1_000_000, 'mil': 1_000_000, 'm': 1_000_000,
            'billion': 1_000_000_000, 'bil': 1_000_000_000, 'b': 1_000_000_000,
            'thousand': 1_000, 'k': 1_000
        }
        
        # Find all potential numbers
        candidates = []
        
        for pattern in number_patterns:
            matches = re.finditer(pattern, context, re.IGNORECASE)
            for match in matches:
                try:
                    number_str = match.group(1).replace(',', '')
                    base_value = float(number_str)
                    
                    # Determine multiplier
                    multiplier = 1
                    full_match = match.group(0).lower()
                    for unit, mult in multipliers.items():
                        if unit in full_match:
                            multiplier = mult
                            break
                    
                    final_value = base_value * multiplier
                    
                    # Add to candidates with position for scoring
                    candidates.append((final_value, match.start()))
                    
                except (ValueError, IndexError):
                    continue
        
        # Return the most likely candidate (first reasonable value found)
        if candidates:
            # Sort by position (closer to pattern match is better)
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]
        
        return None
    
    def parse_company_profile(self, filing_text: str, ticker: str, 
                            cik: str) -> CompanyProfile:
        """Extract company profile information from filing"""
        cleaned_text = self._clean_filing_text(filing_text)
        
        # Extract basic information
        company_name = self._extract_company_name(cleaned_text, ticker)
        
        # Extract business description (usually in Item 1)
        business_desc = self._extract_business_description(cleaned_text)
        
        # Extract industry/sector information
        industry, sector = self._extract_industry_sector(cleaned_text)
        
        # Extract employee count
        employees = self._extract_employee_count(cleaned_text)
        
        # Extract SIC code
        sic_code = self._extract_sic_code(cleaned_text)
        
        return CompanyProfile(
            ticker=ticker,
            cik=cik,
            company_name=company_name,
            industry=industry,
            sector=sector,
            sic_code=sic_code,
            employees=employees,
            business_description=business_desc[:1000] if business_desc else None  # Limit length
        )
    
    def _extract_business_description(self, text: str) -> Optional[str]:
        """Extract business description from Item 1"""
        # Look for Item 1 Business section
        patterns = [
            r'item\s+1\s*[\.\-]?\s*business\s*(.{500,2000}?)(?:item\s+\d|$)',
            r'business\s+overview\s*(.{500,2000}?)(?:item\s+\d|$)',
            r'description\s+of\s+business\s*(.{500,2000}?)(?:item\s+\d|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text[:50000], re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_industry_sector(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract industry and sector information"""
        # Common industry keywords
        industry_patterns = [
            r'industry:\s*([A-Za-z\s&\-]+)',
            r'operates\s+in\s+the\s+([A-Za-z\s&\-]+)\s+industry',
            r'engaged\s+in\s+([A-Za-z\s&\-]+)'
        ]
        
        industry = None
        for pattern in industry_patterns:
            match = re.search(pattern, text[:10000], re.IGNORECASE)
            if match:
                industry = match.group(1).strip()
                break
                
        # Sector is typically derived from industry
        sector = self._map_industry_to_sector(industry) if industry else None
        
        return industry, sector
    
    def _extract_employee_count(self, text: str) -> Optional[int]:
        """Extract employee count"""
        patterns = [
            r'approximately\s+(\d{1,3}(?:,\d{3})*)\s+employees',
            r'(\d{1,3}(?:,\d{3})*)\s+full.?time\s+employees',
            r'workforce\s+of\s+approximately\s+(\d{1,3}(?:,\d{3})*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1).replace(',', ''))
                except ValueError:
                    continue
        
        return None
    
    def _extract_sic_code(self, text: str) -> Optional[str]:
        """Extract SIC code"""
        pattern = r'sic\s+code:?\s*(\d{4})'
        match = re.search(pattern, text[:5000], re.IGNORECASE)
        return match.group(1) if match else None
    
    def _map_industry_to_sector(self, industry: str) -> Optional[str]:
        """Map industry to broader sector"""
        if not industry:
            return None
            
        industry_lower = industry.lower()
        
        sector_mapping = {
            'technology': ['software', 'hardware', 'semiconductor', 'internet', 'computer'],
            'healthcare': ['pharmaceutical', 'biotechnology', 'medical', 'healthcare'],
            'financial': ['banking', 'insurance', 'investment', 'financial'],
            'energy': ['oil', 'gas', 'energy', 'renewable', 'utilities'],
            'consumer': ['retail', 'consumer', 'apparel', 'food', 'beverage'],
            'industrial': ['manufacturing', 'aerospace', 'defense', 'construction'],
            'telecommunications': ['telecommunications', 'wireless', 'cable', 'media']
        }
        
        for sector, keywords in sector_mapping.items():
            if any(keyword in industry_lower for keyword in keywords):
                return sector.title()
        
        return 'Other'
