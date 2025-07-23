"""
Embedding Generation and Management

This module handles text embedding generation using various models
and provides utilities for embedding storage and retrieval.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import json
from pathlib import Path

# Import embedding models
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("openai not available")

from ..config import EMBEDDING_CONFIG, DATA_PATHS
from ..utils.file_utils import ensure_directory, save_json, load_json
from ..utils.performance_integration import monitor_performance, performance_context

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Multi-model embedding generator with fallback strategies.
    Supports local models (sentence-transformers) and API-based models (OpenAI).
    """
    
    def __init__(self):
        self.local_model = None
        self.embedding_dim = None
        self.model_name = EMBEDDING_CONFIG["model_name"]
        self.batch_size = EMBEDDING_CONFIG["batch_size"]
        
        # Initialize the best available model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model with fallback strategy."""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._initialize_local_model()
            elif OPENAI_AVAILABLE:
                self._initialize_openai_model()
            else:
                raise RuntimeError("No embedding models available")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def _initialize_local_model(self):
        """Initialize local sentence-transformer model."""
        try:
            logger.info(f"Loading local embedding model: {self.model_name}")
            self.local_model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.local_model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
            
            logger.info(f"Local model loaded successfully. Dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            if "all-MiniLM-L6-v2" not in self.model_name:
                # Fallback to smaller model
                logger.info("Trying fallback model: all-MiniLM-L6-v2")
                self.model_name = "all-MiniLM-L6-v2"
                self.local_model = SentenceTransformer(self.model_name)
                test_embedding = self.local_model.encode(["test"])
                self.embedding_dim = test_embedding.shape[1]
            else:
                raise
    
    def _initialize_openai_model(self):
        """Initialize OpenAI embedding model."""
        # OpenAI embeddings will be handled in generate_embeddings method
        self.embedding_dim = 1536  # Default for text-embedding-ada-002
        logger.info("Using OpenAI embeddings")
    
    @monitor_performance("embedding_generation", include_args=True)
    async def generate_embeddings(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress logging
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            if self.local_model is not None:
                return await self._generate_local_embeddings(texts, show_progress)
            elif OPENAI_AVAILABLE:
                return await self._generate_openai_embeddings(texts, show_progress)
            else:
                raise RuntimeError("No embedding model available")
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def _generate_local_embeddings(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings using local model."""
        if show_progress:
            logger.info(f"Generating embeddings for {len(texts)} texts using local model")
        
        # Process in batches for memory efficiency
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Run in thread pool to avoid blocking
            batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self.local_model.encode, batch
            )
            
            embeddings.append(batch_embeddings)
            
            if show_progress and len(texts) > self.batch_size:
                logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
        
        return np.vstack(embeddings)
    
    async def _generate_openai_embeddings(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        if show_progress:
            logger.info(f"Generating embeddings for {len(texts)} texts using OpenAI")
        
        embeddings = []
        
        # Process in smaller batches for API
        api_batch_size = min(self.batch_size, 100)  # OpenAI limit
        
        for i in range(0, len(texts), api_batch_size):
            batch = texts[i:i + api_batch_size]
            
            try:
                response = await openai.Embedding.acreate(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
                
                if show_progress and len(texts) > api_batch_size:
                    logger.info(f"Processed batch {i//api_batch_size + 1}/{(len(texts)-1)//api_batch_size + 1}")
                    
            except Exception as e:
                logger.error(f"OpenAI embedding failed for batch {i}: {e}")
                # Fill with zeros for failed batch
                embeddings.extend([[0.0] * self.embedding_dim] * len(batch))
        
        return np.array(embeddings)
    
    @monitor_performance("single_embedding_generation")
    async def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string
            
        Returns:
            numpy array embedding
        """
        embeddings = await self.generate_embeddings([text], show_progress=False)
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "model_type": "local" if self.local_model else "api",
            "batch_size": self.batch_size
        }


class EmbeddingStorage:
    """
    Storage and retrieval system for embeddings with metadata.
    Supports both file-based and in-memory storage.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path(DATA_PATHS["embeddings"])
        ensure_directory(self.storage_path)
        
        # In-memory cache for active session
        self.embedding_cache: Dict[str, Dict[str, Any]] = {}
        
    async def store_embeddings(
        self,
        session_id: str,
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store embeddings with associated metadata.
        
        Args:
            session_id: Session identifier
            document_id: Document identifier
            chunks: List of text chunks with metadata
            embeddings: numpy array of embeddings
            metadata: Additional metadata
        """
        try:
            # Prepare storage data
            storage_data = {
                "session_id": session_id,
                "document_id": document_id,
                "created_at": datetime.utcnow().isoformat(),
                "embedding_dimension": embeddings.shape[1],
                "chunk_count": len(chunks),
                "metadata": metadata or {},
                "chunks": chunks
            }
            
            # Create session directory
            session_dir = self.storage_path / session_id
            ensure_directory(session_dir)
            
            # Save metadata
            metadata_path = session_dir / f"{document_id}_embedding_metadata.json"
            save_json(storage_data, metadata_path)
            
            # Save embeddings as numpy array
            embeddings_path = session_dir / f"{document_id}_embeddings.npy"
            np.save(embeddings_path, embeddings)
            
            # Cache in memory
            cache_key = f"{session_id}_{document_id}"
            self.embedding_cache[cache_key] = {
                "embeddings": embeddings,
                "chunks": chunks,
                "metadata": storage_data
            }
            
            logger.info(f"Stored embeddings for document {document_id} in session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise
    
    async def load_embeddings(
        self, 
        session_id: str, 
        document_id: str
    ) -> Optional[Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Load embeddings and associated data.
        
        Args:
            session_id: Session identifier
            document_id: Document identifier
            
        Returns:
            Tuple of (embeddings, chunks, metadata) or None if not found
        """
        cache_key = f"{session_id}_{document_id}"
        
        # Check cache first
        if cache_key in self.embedding_cache:
            cached = self.embedding_cache[cache_key]
            return cached["embeddings"], cached["chunks"], cached["metadata"]
        
        try:
            session_dir = self.storage_path / session_id
            
            # Load metadata
            metadata_path = session_dir / f"{document_id}_embedding_metadata.json"
            if not metadata_path.exists():
                return None
                
            metadata = load_json(metadata_path)
            
            # Load embeddings
            embeddings_path = session_dir / f"{document_id}_embeddings.npy"
            if not embeddings_path.exists():
                return None
                
            embeddings = np.load(embeddings_path)
            chunks = metadata["chunks"]
            
            # Cache the loaded data
            self.embedding_cache[cache_key] = {
                "embeddings": embeddings,
                "chunks": chunks,
                "metadata": metadata
            }
            
            return embeddings, chunks, metadata
            
        except Exception as e:
            logger.error(f"Failed to load embeddings for {document_id}: {e}")
            return None
    
    async def load_session_embeddings(
        self, 
        session_id: str
    ) -> Dict[str, Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Load all embeddings for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary mapping document_id to (embeddings, chunks, metadata)
        """
        session_dir = self.storage_path / session_id
        if not session_dir.exists():
            return {}
        
        session_embeddings = {}
        
        # Find all embedding metadata files
        for metadata_file in session_dir.glob("*_embedding_metadata.json"):
            try:
                # Extract document ID from filename
                document_id = metadata_file.stem.replace("_embedding_metadata", "")
                
                # Load embeddings for this document
                result = await self.load_embeddings(session_id, document_id)
                if result:
                    session_embeddings[document_id] = result
                    
            except Exception as e:
                logger.error(f"Failed to load embeddings from {metadata_file}: {e}")
                continue
        
        logger.info(f"Loaded embeddings for {len(session_embeddings)} documents in session {session_id}")
        return session_embeddings
    
    async def delete_embeddings(self, session_id: str, document_id: str):
        """Delete embeddings for a specific document."""
        try:
            session_dir = self.storage_path / session_id
            
            # Delete files
            metadata_path = session_dir / f"{document_id}_embedding_metadata.json"
            embeddings_path = session_dir / f"{document_id}_embeddings.npy"
            
            for path in [metadata_path, embeddings_path]:
                if path.exists():
                    path.unlink()
            
            # Remove from cache
            cache_key = f"{session_id}_{document_id}"
            if cache_key in self.embedding_cache:
                del self.embedding_cache[cache_key]
            
            logger.info(f"Deleted embeddings for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise
    
    async def delete_session_embeddings(self, session_id: str):
        """Delete all embeddings for a session."""
        try:
            session_dir = self.storage_path / session_id
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
            
            # Clear cache for this session
            keys_to_remove = [key for key in self.embedding_cache.keys() 
                            if key.startswith(f"{session_id}_")]
            for key in keys_to_remove:
                del self.embedding_cache[key]
            
            logger.info(f"Deleted all embeddings for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete session embeddings: {e}")
            raise
    
    def clear_cache(self):
        """Clear the in-memory embedding cache."""
        self.embedding_cache.clear()
        logger.info("Cleared embedding cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        return {
            "cached_items": len(self.embedding_cache),
            "cache_keys": list(self.embedding_cache.keys())
        }
