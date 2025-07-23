"""
Document Processing Module

Handles PDF processing, text extraction, chunking, and preparation
for embedding generation. Uses multiple strategies for robust text extraction.
"""

import logging
import asyncio
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import os

import PyPDF2
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import numpy as np

from ..config import PROCESSING_CONFIG, DATA_PATHS
from ..utils.text_utils import TextCleaner, TextChunker
from ..utils.file_utils import ensure_directory
from ..utils.performance_integration import monitor_performance, performance_context

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Advanced document processor with multiple extraction strategies and
    intelligent chunking for optimal RAG performance.
    """
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.text_chunker = TextChunker()
        self.processed_docs_path = Path(DATA_PATHS["processed"])
        ensure_directory(self.processed_docs_path)
        
    async def health_check(self) -> bool:
        """Check if the document processor is healthy."""
        try:
            # Test basic functionality
            test_text = "This is a test document for health check."
            chunks = await self._chunk_text(test_text, "test_doc")
            return len(chunks) > 0
        except Exception as e:
            logger.error(f"Document processor health check failed: {e}")
            return False
    
    @monitor_performance("document_processing", include_args=True)
    async def process_document(
        self, 
        content: bytes, 
        filename: str, 
        document_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Process a PDF document with multiple extraction strategies.
        
        Args:
            content: PDF file content as bytes
            filename: Original filename
            document_id: Unique document identifier
            session_id: Session identifier
            
        Returns:
            Processing results with metadata
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting processing for document {document_id}")
        
        try:
            # Extract text using multiple strategies
            extracted_text = await self._extract_text_multi_strategy(content, filename)
            
            if not extracted_text.strip():
                raise ValueError("No text could be extracted from the document")
            
            # Clean and preprocess text
            cleaned_text = self.text_cleaner.clean_text(extracted_text)
            
            # Generate chunks
            chunks = await self._chunk_text(cleaned_text, document_id)
            
            # Create document metadata
            metadata = self._create_document_metadata(
                document_id, filename, session_id, content, 
                cleaned_text, chunks, start_time
            )
            
            # Save processed document
            await self._save_processed_document(
                document_id, metadata, chunks, session_id
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(
                f"Successfully processed document {document_id}: "
                f"{len(chunks)} chunks in {processing_time:.2f}s"
            )
            
            return {
                "chunk_count": len(chunks),
                "processing_time": processing_time,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            raise
    
    async def _extract_text_multi_strategy(self, content: bytes, filename: str) -> str:
        """
        Extract text using multiple strategies for robustness.
        
        Args:
            content: PDF content as bytes
            filename: Original filename for logging
            
        Returns:
            Extracted text string
        """
        strategies = [
            ("pdfplumber", self._extract_with_pdfplumber),
            ("pypdf2", self._extract_with_pypdf2),
            ("ocr", self._extract_with_ocr)
        ]
        
        best_text = ""
        best_score = 0
        
        for strategy_name, extract_func in strategies:
            try:
                logger.debug(f"Trying {strategy_name} for {filename}")
                text = await extract_func(content)
                
                if text:
                    # Score text quality (length, character diversity, etc.)
                    score = self._score_extracted_text(text)
                    logger.debug(f"{strategy_name} score: {score}")
                    
                    if score > best_score:
                        best_text = text
                        best_score = score
                        
            except Exception as e:
                logger.warning(f"{strategy_name} extraction failed for {filename}: {e}")
                continue
        
        if not best_text:
            raise ValueError("All text extraction strategies failed")
            
        logger.info(f"Best extraction method had score: {best_score}")
        return best_text
    
    async def _extract_with_pdfplumber(self, content: bytes) -> str:
        """Extract text using pdfplumber (best for most PDFs)."""
        import io
        text_parts = []
        
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue
        
        return "\n\n".join(text_parts)
    
    async def _extract_with_pypdf2(self, content: bytes) -> str:
        """Extract text using PyPDF2 (fallback method)."""
        import io
        text_parts = []
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"PyPDF2 failed on page {page_num}: {e}")
                continue
        
        return "\n\n".join(text_parts)
    
    async def _extract_with_ocr(self, content: bytes) -> str:
        """Extract text using OCR (for image-based PDFs)."""
        try:
            # Convert PDF to images
            images = convert_from_bytes(content, dpi=300)
            text_parts = []
            
            for i, image in enumerate(images):
                try:
                    # Use Tesseract OCR
                    page_text = pytesseract.image_to_string(
                        image, 
                        config='--psm 6 -l eng'
                    )
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"OCR failed on page {i}: {e}")
                    continue
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _score_extracted_text(self, text: str) -> float:
        """
        Score the quality of extracted text.
        
        Args:
            text: Extracted text to score
            
        Returns:
            Quality score (higher is better)
        """
        if not text:
            return 0
        
        # Basic metrics
        length_score = min(len(text) / 1000, 10)  # Length up to 1000 chars
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        diversity_score = min(unique_chars / 26, 5)  # Up to 26 letters
        
        # Word count
        words = text.split()
        word_score = min(len(words) / 100, 5)  # Up to 100 words
        
        # Penalty for too many special characters (indicates poor extraction)
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:')
        special_penalty = min(special_chars / len(text) * 10, 5)
        
        total_score = length_score + diversity_score + word_score - special_penalty
        return max(total_score, 0)
    
    async def _chunk_text(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Chunk text into optimal segments for RAG.
        
        Args:
            text: Cleaned text to chunk
            document_id: Document identifier
            
        Returns:
            List of text chunks with metadata
        """
        chunks = self.text_chunker.chunk_text(
            text=text,
            chunk_size=PROCESSING_CONFIG["chunk_size"],
            chunk_overlap=PROCESSING_CONFIG["chunk_overlap"],
            document_id=document_id
        )
        
        # Add additional metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.update({
                "chunk_index": i,
                "document_id": document_id,
                "chunk_id": f"{document_id}_chunk_{i}",
                "created_at": datetime.utcnow().isoformat()
            })
        
        return chunks
    
    def _create_document_metadata(
        self, 
        document_id: str, 
        filename: str, 
        session_id: str,
        content: bytes, 
        text: str, 
        chunks: List[Dict], 
        start_time: datetime
    ) -> Dict[str, Any]:
        """Create comprehensive document metadata."""
        
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content).hexdigest()
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        metadata = {
            "document_id": document_id,
            "filename": filename,
            "session_id": session_id,
            "content_hash": content_hash,
            "text_hash": text_hash,
            "file_size": len(content),
            "text_length": len(text),
            "chunk_count": len(chunks),
            "processed_at": datetime.utcnow().isoformat(),
            "processing_config": {
                "chunk_size": PROCESSING_CONFIG["chunk_size"],
                "chunk_overlap": PROCESSING_CONFIG["chunk_overlap"]
            },
            "extraction_metadata": {
                "word_count": len(text.split()),
                "character_count": len(text),
                "estimated_pages": len(text) // 2000  # Rough estimate
            }
        }
        
        return metadata
    
    async def _save_processed_document(
        self, 
        document_id: str, 
        metadata: Dict[str, Any],
        chunks: List[Dict[str, Any]], 
        session_id: str
    ):
        """Save processed document data to disk."""
        
        # Create session directory
        session_dir = self.processed_docs_path / session_id
        ensure_directory(session_dir)
        
        # Save metadata
        metadata_path = session_dir / f"{document_id}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Save chunks
        chunks_path = session_dir / f"{document_id}_chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved processed document {document_id} to {session_dir}")
    
    async def list_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """List all documents in a session."""
        session_dir = self.processed_docs_path / session_id
        
        if not session_dir.exists():
            return []
        
        documents = []
        for metadata_file in session_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    documents.append({
                        "document_id": metadata["document_id"],
                        "filename": metadata["filename"],
                        "processed_at": metadata["processed_at"],
                        "chunk_count": metadata["chunk_count"],
                        "file_size": metadata["file_size"]
                    })
            except Exception as e:
                logger.error(f"Failed to load metadata from {metadata_file}: {e}")
                continue
        
        return documents
    
    async def delete_document(self, document_id: str, session_id: str):
        """Delete a document and its associated files."""
        session_dir = self.processed_docs_path / session_id
        
        if not session_dir.exists():
            return
        
        # Delete metadata and chunks files
        for pattern in [f"{document_id}_metadata.json", f"{document_id}_chunks.json"]:
            file_path = session_dir / pattern
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted {file_path}")
    
    async def get_document_chunks(self, document_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        session_dir = self.processed_docs_path / session_id
        chunks_path = session_dir / f"{document_id}_chunks.json"
        
        if not chunks_path.exists():
            return []
        
        try:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load chunks for document {document_id}: {e}")
            return []
