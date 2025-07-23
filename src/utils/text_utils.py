"""
Text Processing Utilities

This module contains utilities for text cleaning, chunking, and preprocessing
optimized for RAG systems.
"""

import re
import string
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """Advanced text cleaning for optimal RAG performance."""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s\.,!?;:()\-\'""]')
        self.bullet_pattern = re.compile(r'^[\s]*[•\-\*\+]\s*', re.MULTILINE)
        self.page_break_pattern = re.compile(r'\f')
        self.multiple_dots_pattern = re.compile(r'\.{3,}')
        
    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning for RAG optimization.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text suitable for chunking and embedding
        """
        if not text:
            return ""
        
        # Remove page breaks
        text = self.page_break_pattern.sub('\n', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Clean up bullet points
        text = self.bullet_pattern.sub('• ', text)
        
        # Handle multiple dots (ellipsis, etc.)
        text = self.multiple_dots_pattern.sub('...', text)
        
        # Remove excessive special characters but keep punctuation
        text = self._clean_special_chars(text)
        
        # Fix common OCR errors
        text = self._fix_ocr_errors(text)
        
        # Normalize paragraph breaks
        text = self._normalize_paragraphs(text)
        
        return text.strip()
    
    def _clean_special_chars(self, text: str) -> str:
        """Remove problematic special characters while preserving structure."""
        # Keep important punctuation and structure
        allowed_special = set('.,!?;:()\-\'""\n\t ')
        
        cleaned_chars = []
        for char in text:
            if char.isalnum() or char in allowed_special:
                cleaned_chars.append(char)
            elif char in '•◦▪▫':  # Convert bullet variants
                cleaned_chars.append('•')
            elif char in '\u201c\u201d\u2018\u2019':  # Normalize smart quotes
                if char in '\u201c\u201d':  # Double quotes
                    cleaned_chars.append('"')
                else:  # Single quotes/apostrophes
                    cleaned_chars.append("'")
            else:
                cleaned_chars.append(' ')  # Replace unknown chars with space
        
        return ''.join(cleaned_chars)
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors."""
        fixes = {
            # Common OCR substitutions
            r'\b0\b': 'O',  # Zero to O
            r'\bl\b': 'I',  # lowercase l to I
            r'\brn\b': 'm',  # rn to m
            r'\bvv\b': 'w',  # vv to w
            # Fix broken words
            r'\b(\w+)-\s*\n\s*(\w+)\b': r'\1\2',  # Hyphenated line breaks
        }
        
        for pattern, replacement in fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_paragraphs(self, text: str) -> str:
        """Normalize paragraph breaks and spacing."""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Ensure sentences end properly
        text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1\n\n\2', text)
        
        return text


class TextChunker:
    """Advanced text chunking with semantic awareness."""
    
    def __init__(self):
        self.sentence_endings = re.compile(r'[.!?]+\s*')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        document_id: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Create semantically-aware text chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            document_id: Document identifier
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text:
            return []
        
        # Try semantic chunking first
        chunks = self._semantic_chunk(text, chunk_size, chunk_overlap)
        
        # Fallback to simple chunking if semantic fails
        if not chunks:
            chunks = self._simple_chunk(text, chunk_size, chunk_overlap)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                "text": chunk,
                "chunk_size": len(chunk),
                "word_count": len(chunk.split()),
                "document_id": document_id,
                "chunk_index": i
            }
            chunks[i] = chunk_dict
        
        return chunks
    
    def _semantic_chunk(
        self, 
        text: str, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> List[str]:
        """Create chunks respecting semantic boundaries."""
        try:
            # Split by paragraphs first
            paragraphs = self.paragraph_breaks.split(text)
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # If paragraph fits in current chunk, add it
                if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Save current chunk if it has content
                    if current_chunk:
                        chunks.append(current_chunk)
                        
                        # Start new chunk with overlap
                        overlap_text = self._get_overlap_text(current_chunk, chunk_overlap)
                        current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                    else:
                        # Paragraph is too long, split by sentences
                        sentence_chunks = self._split_by_sentences(paragraph, chunk_size, chunk_overlap)
                        chunks.extend(sentence_chunks)
                        current_chunk = ""
            
            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}, falling back to simple chunking")
            return []
    
    def _split_by_sentences(
        self, 
        text: str, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> List[str]:
        """Split text by sentences when paragraphs are too long."""
        sentences = self.sentence_endings.split(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add sentence ending back
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap_text = self._get_overlap_text(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Single sentence is too long, force split
                    chunks.append(sentence[:chunk_size])
                    current_chunk = sentence[chunk_size-chunk_overlap:] if len(sentence) > chunk_size else ""
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _simple_chunk(
        self, 
        text: str, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> List[str]:
        """Simple character-based chunking as fallback."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                # Look back for a space
                space_pos = text.rfind(' ', start, end)
                if space_pos > start + chunk_size // 2:  # Only if space is not too far back
                    end = space_pos
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + chunk_size - chunk_overlap, start + 1)
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= overlap_size:
            return text
        
        overlap_text = text[-overlap_size:]
        
        # Try to start at word boundary
        space_pos = overlap_text.find(' ')
        if space_pos > 0:
            overlap_text = overlap_text[space_pos:].strip()
        
        return overlap_text
