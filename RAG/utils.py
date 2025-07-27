"""
Utility functions for the RAG system
====================================

Common utilities used across all layers of the RAG system.
"""

import logging
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Set up logging configuration for the RAG system"""
    
    # Create logger
    logger = logging.getLogger("RAG_System")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def generate_uuid() -> str:
    """Generate a unique UUID string"""
    return str(uuid.uuid4())

def get_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def safe_filename(text: str) -> str:
    """Convert text to a safe filename"""
    # Remove or replace problematic characters
    safe_chars = []
    for char in text:
        if char.isalnum() or char in '-_.':
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    return ''.join(safe_chars)

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from a JSONL file"""
    data = []
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: Path) -> None:
    """Save data to a JSONL file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def append_jsonl(item: Dict[str, Any], file_path: Path) -> None:
    """Append a single item to a JSONL file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item) + '\n')

def validate_ticker(ticker: str) -> bool:
    """Validate that a ticker symbol is properly formatted"""
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Basic validation: 1-5 uppercase letters
    ticker = ticker.strip().upper()
    return len(ticker) >= 1 and len(ticker) <= 5 and ticker.isalpha()

def normalize_ticker(ticker: str) -> str:
    """Normalize a ticker symbol to standard format"""
    return ticker.strip().upper()

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()

def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"

def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def count_tokens_approximate(text: str) -> int:
    """Approximate token count (rough estimate: ~4 chars per token)"""
    return len(text) // 4

class ProgressTracker:
    """Simple progress tracking utility"""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger("RAG_System")
    
    def update(self, increment: int = 1) -> None:
        """Update progress"""
        self.current += increment
        if self.current % max(1, self.total // 20) == 0:  # Log every 5%
            percent = (self.current / self.total) * 100
            self.logger.info(f"{self.description}: {percent:.1f}% ({self.current}/{self.total})")
    
    def finish(self) -> None:
        """Mark as finished"""
        self.logger.info(f"{self.description}: Complete ({self.total}/{self.total})")

def progress_bar(current: int, total: int, description: str = "Progress") -> str:
    """Simple progress bar string"""
    percent = (current / total) * 100 if total > 0 else 0
    return f"{description}: {percent:.1f}% ({current}/{total})"

# Common file extensions
PDF_EXTENSIONS = {'.pdf'}
TEXT_EXTENSIONS = {'.txt', '.md', '.json', '.jsonl'}
PARQUET_EXTENSIONS = {'.parquet'}
INDEX_EXTENSIONS = {'.index', '.faiss'}
