"""
Text Chunking Manager
====================

Intelligent text segmentation for SEC filings with semantic awareness.
Handles overlapping chunks, section detection, and metadata preservation.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import uuid

from ..config import RAGConfig
from ..utils import setup_logging, safe_filename

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    chunk_id: str
    text: str
    start_pos: int
    end_pos: int
    source_document: str
    ticker: str
    form_type: str
    filing_date: str
    section: Optional[str] = None
    chunk_index: int = 0
    token_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'source_document': self.source_document,
            'ticker': self.ticker,
            'form_type': self.form_type,
            'filing_date': self.filing_date,
            'section': self.section,
            'chunk_index': self.chunk_index,
            'token_count': self.token_count
        }

class TextChunkingManager:
    """Manages intelligent text chunking for SEC filings"""
    
    # SEC filing section patterns
    SECTION_PATTERNS = {
        'business': [
            r'item\s+1\s*[\.\-]?\s*business',
            r'part\s+i\s*[\.\-]?\s*item\s+1\s*[\.\-]?\s*business'
        ],
        'risk_factors': [
            r'item\s+1a\s*[\.\-]?\s*risk\s+factors',
            r'risk\s+factors'
        ],
        'properties': [
            r'item\s+2\s*[\.\-]?\s*properties'
        ],
        'legal_proceedings': [
            r'item\s+3\s*[\.\-]?\s*legal\s+proceedings'
        ],
        'financial_statements': [
            r'item\s+8\s*[\.\-]?\s*financial\s+statements',
            r'consolidated\s+statements',
            r'balance\s+sheets?',
            r'income\s+statements?',
            r'cash\s+flows?'
        ],
        'md_and_a': [
            r'item\s+7\s*[\.\-]?\s*management.?s\s+discussion',
            r'management.?s\s+discussion\s+and\s+analysis'
        ],
        'controls': [
            r'item\s+9a\s*[\.\-]?\s*controls\s+and\s+procedures'
        ]
    }
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Chunking parameters
        self.chunk_size = config.CHUNK_SIZE  # Target tokens per chunk
        self.chunk_overlap = config.CHUNK_OVERLAP  # Overlap tokens
        self.max_chunk_size = config.MAX_CHUNK_SIZE  # Maximum tokens per chunk
        
        # Compile section patterns
        self._compiled_sections = {}
        for section_name, patterns in self.SECTION_PATTERNS.items():
            self._compiled_sections[section_name] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
        
        self.logger.info("Text Chunking Manager initialized")
    
    def chunk_document(self, text: str, source_document: str,
                      ticker: str, form_type: str, filing_date: str) -> List[TextChunk]:
        """Chunk a document into overlapping text segments"""
        if not text or not text.strip():
            return []
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Detect sections
        sections = self._detect_sections(cleaned_text)
        
        # Create chunks with section awareness
        chunks = self._create_chunks_with_sections(
            cleaned_text, sections, source_document, ticker, form_type, filing_date
        )
        
        self.logger.info(f"Created {len(chunks)} chunks from document {source_document}")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize SEC filing text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common SEC filing artifacts
        text = re.sub(r'table\s+of\s+contents', '', text, flags=re.IGNORECASE)
        text = re.sub(r'page\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove HTML-like tags if present
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\$\%]', '', text)
        
        # Remove lines with just numbers or symbols
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if len(line.strip()) > 10 and not re.match(r'^[\d\s\.\-\$\,]+$', line.strip()):
                cleaned_lines.append(line.strip())
        
        return '\n'.join(cleaned_lines)
    
    def _detect_sections(self, text: str) -> List[Tuple[str, int, int]]:
        """Detect sections in SEC filing text"""
        sections = []
        
        for section_name, patterns in self._compiled_sections.items():
            for pattern in patterns:
                matches = list(pattern.finditer(text))
                for match in matches:
                    sections.append((section_name, match.start(), match.end()))
        
        # Sort sections by position
        sections.sort(key=lambda x: x[1])
        
        # Merge overlapping sections of the same type
        merged_sections = []
        for section_name, start, end in sections:
            if merged_sections and merged_sections[-1][0] == section_name:
                # Extend previous section
                merged_sections[-1] = (section_name, merged_sections[-1][1], max(end, merged_sections[-1][2]))
            else:
                merged_sections.append((section_name, start, end))
        
        return merged_sections
    
    def _create_chunks_with_sections(self, text: str, sections: List[Tuple[str, int, int]],
                                   source_document: str, ticker: str, 
                                   form_type: str, filing_date: str) -> List[TextChunk]:
        """Create chunks with section awareness"""
        chunks = []
        text_length = len(text)
        
        # If no sections detected, chunk the entire document
        if not sections:
            return self._create_sliding_window_chunks(
                text, 0, text_length, None, source_document, ticker, form_type, filing_date
            )
        
        # Process each section
        for i, (section_name, section_start, section_end) in enumerate(sections):
            # Determine section text boundaries
            if i < len(sections) - 1:
                next_section_start = sections[i + 1][1]
                section_text_end = min(next_section_start, text_length)
            else:
                section_text_end = text_length
            
            # Extract section text (from pattern match to next section)
            section_text = text[section_start:section_text_end]
            
            # Create chunks for this section
            section_chunks = self._create_sliding_window_chunks(
                section_text, section_start, section_text_end, section_name,
                source_document, ticker, form_type, filing_date
            )
            
            chunks.extend(section_chunks)
        
        # Handle text before first section
        if sections and sections[0][1] > 0:
            pre_section_text = text[:sections[0][1]]
            pre_section_chunks = self._create_sliding_window_chunks(
                pre_section_text, 0, sections[0][1], 'header',
                source_document, ticker, form_type, filing_date
            )
            chunks = pre_section_chunks + chunks
        
        return chunks
    
    def _create_sliding_window_chunks(self, text: str, global_start: int, global_end: int,
                                    section: Optional[str], source_document: str,
                                    ticker: str, form_type: str, filing_date: str) -> List[TextChunk]:
        """Create overlapping chunks using sliding window approach"""
        chunks = []
        
        if not text.strip():
            return chunks
        
        # Approximate token count (rough estimate: ~4 chars per token)
        char_per_token = 4
        target_chars = self.chunk_size * char_per_token
        overlap_chars = self.chunk_overlap * char_per_token
        max_chars = self.max_chunk_size * char_per_token
        
        # Split into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return chunks
        
        current_chunk_text = ""
        current_chunk_start = global_start
        chunk_index = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            
            # Check if adding this sentence would exceed max chunk size
            potential_text = current_chunk_text + " " + sentence if current_chunk_text else sentence
            
            if len(potential_text) <= max_chars:
                current_chunk_text = potential_text
                i += 1
            else:
                # Create chunk if we have text
                if current_chunk_text.strip():
                    chunk = self._create_chunk(
                        current_chunk_text.strip(),
                        current_chunk_start,
                        current_chunk_start + len(current_chunk_text),
                        section, source_document, ticker, form_type, filing_date, chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                if len(sentence) <= max_chars:
                    # Start new chunk with this sentence
                    current_chunk_text = sentence
                    current_chunk_start = global_start + text.find(sentence, current_chunk_start - global_start)
                    i += 1
                else:
                    # Sentence is too long, split it
                    words = sentence.split()
                    partial_sentence = ""
                    for word in words:
                        if len(partial_sentence + " " + word) <= max_chars:
                            partial_sentence = partial_sentence + " " + word if partial_sentence else word
                        else:
                            if partial_sentence:
                                chunk = self._create_chunk(
                                    partial_sentence.strip(),
                                    current_chunk_start,
                                    current_chunk_start + len(partial_sentence),
                                    section, source_document, ticker, form_type, filing_date, chunk_index
                                )
                                chunks.append(chunk)
                                chunk_index += 1
                            partial_sentence = word
                            current_chunk_start += len(partial_sentence) + 1
                    
                    current_chunk_text = partial_sentence
                    i += 1
            
            # Check if we've reached target chunk size
            if len(current_chunk_text) >= target_chars and current_chunk_text.strip():
                chunk = self._create_chunk(
                    current_chunk_text.strip(),
                    current_chunk_start,
                    current_chunk_start + len(current_chunk_text),
                    section, source_document, ticker, form_type, filing_date, chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk_text, overlap_chars)
                current_chunk_text = overlap_text
                current_chunk_start += len(current_chunk_text) - len(overlap_text)
        
        # Add final chunk if we have remaining text
        if current_chunk_text.strip():
            chunk = self._create_chunk(
                current_chunk_text.strip(),
                current_chunk_start,
                current_chunk_start + len(current_chunk_text),
                section, source_document, ticker, form_type, filing_date, chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics"""
        # Split on sentence endings, but be careful with abbreviations and numbers
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_endings.split(text)
        
        # Filter out very short sentences (likely artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return sentences
    
    def _get_overlap_text(self, text: str, overlap_chars: int) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= overlap_chars:
            return text
            
        # Try to break at sentence boundary
        overlap_text = text[-overlap_chars:]
        
        # Find the first sentence start in the overlap
        sentences = self._split_into_sentences(overlap_text)
        if sentences:
            return sentences[0]
        else:
            return overlap_text
    
    def _create_chunk(self, text: str, start_pos: int, end_pos: int,
                     section: Optional[str], source_document: str,
                     ticker: str, form_type: str, filing_date: str, chunk_index: int) -> TextChunk:
        """Create a TextChunk object"""
        chunk_id = str(uuid.uuid4())
        token_count = self._estimate_token_count(text)
        
        return TextChunk(
            chunk_id=chunk_id,
            text=text,
            start_pos=start_pos,
            end_pos=end_pos,
            source_document=source_document,
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            section=section,
            chunk_index=chunk_index,
            token_count=token_count
        )
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple approximation: ~4 characters per token
        return len(text) // 4
    
    def chunk_multiple_documents(self, documents: List[Dict[str, Any]]) -> List[TextChunk]:
        """Chunk multiple documents efficiently"""
        all_chunks = []
        
        for doc in documents:
            doc_chunks = self.chunk_document(
                text=doc['text'],
                source_document=doc['source_document'],
                ticker=doc['ticker'],
                form_type=doc['form_type'],
                filing_date=doc['filing_date']
            )
            all_chunks.extend(doc_chunks)
        
        self.logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks
    
    def get_chunks_by_section(self, chunks: List[TextChunk], section: str) -> List[TextChunk]:
        """Filter chunks by section type"""
        return [chunk for chunk in chunks if chunk.section == section]
    
    def get_chunks_by_ticker(self, chunks: List[TextChunk], ticker: str) -> List[TextChunk]:
        """Filter chunks by company ticker"""
        return [chunk for chunk in chunks if chunk.ticker.upper() == ticker.upper()]
    
    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_length': 0,
                'sections': {},
                'tickers': {},
                'form_types': {}
            }
        
        # Basic stats
        total_chunks = len(chunks)
        avg_length = sum(len(chunk.text) for chunk in chunks) / total_chunks
        
        # Group by section
        sections = {}
        for chunk in chunks:
            section = chunk.section or 'unknown'
            sections[section] = sections.get(section, 0) + 1
        
        # Group by ticker
        tickers = {}
        for chunk in chunks:
            tickers[chunk.ticker] = tickers.get(chunk.ticker, 0) + 1
        
        # Group by form type
        form_types = {}
        for chunk in chunks:
            form_types[chunk.form_type] = form_types.get(chunk.form_type, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'avg_chunk_length': avg_length,
            'sections': sections,
            'tickers': tickers,
            'form_types': form_types,
            'avg_tokens_per_chunk': sum(chunk.token_count for chunk in chunks) / total_chunks,
            'total_tokens': sum(chunk.token_count for chunk in chunks)
        }
