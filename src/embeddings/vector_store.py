"""
FAISS Vector Store

This module provides FAISS-based vector storage and similarity search
for efficient retrieval in the RAG system.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
import pickle
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, falling back to simple similarity search")

from ..config import RETRIEVAL_CONFIG, DATA_PATHS
from ..utils.file_utils import ensure_directory
from .generator import EmbeddingGenerator, EmbeddingStorage

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store with fallback to simple cosine similarity.
    Supports multiple distance metrics and indexing strategies.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.id_counter = 0
        
        # Index configuration
        self.index_type = RETRIEVAL_CONFIG.get("index_type", "flat")
        self.distance_metric = RETRIEVAL_CONFIG.get("distance_metric", "cosine")
        
        # Storage paths
        self.vector_store_path = Path(DATA_PATHS["vector_store"])
        ensure_directory(self.vector_store_path)
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index or fallback similarity search."""
        try:
            if FAISS_AVAILABLE:
                self._initialize_faiss_index()
            else:
                self._initialize_simple_index()
            
            # Always initialize embeddings_matrix for compatibility
            self.embeddings_matrix = np.array([]).reshape(0, self.dimension)
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self._initialize_simple_index()
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index with specified configuration."""
        logger.info(f"Initializing FAISS index: type={self.index_type}, metric={self.distance_metric}")
        
        if self.distance_metric == "cosine":
            # For cosine similarity, we use inner product after normalization
            if self.index_type == "flat":
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                # IVF index for large datasets
                quantizer = faiss.IndexFlatIP(self.dimension)
                nlist = min(100, max(10, self.dimension // 4))  # Adaptive nlist
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                # Default to flat
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # L2 distance
            if self.index_type == "flat":
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dimension)
                nlist = min(100, max(10, self.dimension // 4))
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
        
        logger.info(f"FAISS index initialized successfully")
    
    def _initialize_simple_index(self):
        """Initialize simple numpy-based similarity search as fallback."""
        logger.info("Initializing simple vector store (FAISS not available)")
        self.embeddings_matrix = np.array([]).reshape(0, self.dimension)
        self.index = None
    
    async def add_vectors(
        self, 
        embeddings: np.ndarray, 
        metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add vectors to the index with associated metadata.
        
        Args:
            embeddings: numpy array of embeddings
            metadata_list: List of metadata dictionaries for each embedding
            
        Returns:
            List of assigned IDs
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata must have same length")
        
        try:
            # Normalize embeddings for cosine similarity
            if self.distance_metric == "cosine":
                embeddings = self._normalize_embeddings(embeddings)
            
            # Assign IDs
            ids = list(range(self.id_counter, self.id_counter + len(embeddings)))
            self.id_counter += len(embeddings)
            
            if FAISS_AVAILABLE and self.index is not None:
                await self._add_to_faiss_index(embeddings, ids, metadata_list)
            else:
                await self._add_to_simple_index(embeddings, ids, metadata_list)
            
            logger.info(f"Added {len(embeddings)} vectors to index")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add vectors to index: {e}")
            raise
    
    async def _add_to_faiss_index(
        self, 
        embeddings: np.ndarray, 
        ids: List[int], 
        metadata_list: List[Dict[str, Any]]
    ):
        """Add vectors to FAISS index."""
        # Train index if necessary (for IVF indices)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if self.index.ntotal + len(embeddings) >= 100:  # Minimum training size
                training_data = embeddings
                if self.index.ntotal > 0:
                    # Get some existing vectors for training
                    existing_vectors = self.index.reconstruct_n(0, min(self.index.ntotal, 1000))
                    training_data = np.vstack([existing_vectors, embeddings])
                
                logger.info("Training FAISS index...")
                self.index.train(training_data.astype(np.float32))
        
        # Add vectors
        self.index.add(embeddings.astype(np.float32))
        
        # Also update embeddings_matrix for compatibility
        if self.embeddings_matrix.shape[0] == 0:
            self.embeddings_matrix = embeddings
        else:
            self.embeddings_matrix = np.vstack([self.embeddings_matrix, embeddings])
        
        # Store metadata
        for i, metadata in enumerate(metadata_list):
            self.metadata_store[ids[i]] = metadata
    
    async def _add_to_simple_index(
        self, 
        embeddings: np.ndarray, 
        ids: List[int], 
        metadata_list: List[Dict[str, Any]]
    ):
        """Add vectors to simple numpy-based index."""
        # Append to embeddings matrix
        if self.embeddings_matrix.shape[0] == 0:
            self.embeddings_matrix = embeddings
        else:
            self.embeddings_matrix = np.vstack([self.embeddings_matrix, embeddings])
        
        # Store metadata
        for i, metadata in enumerate(metadata_list):
            self.metadata_store[ids[i]] = metadata
    
    async def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            if self.distance_metric == "cosine":
                query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))[0]
            
            if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
                results = await self._search_faiss(query_embedding, top_k)
            elif self.embeddings_matrix.shape[0] > 0:
                results = await self._search_simple(query_embedding, top_k)
            else:
                return []
            
            # Apply score threshold if specified
            if score_threshold is not None:
                results = [r for r in results if r["score"] >= score_threshold]
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _search_faiss(
        self, 
        query_embedding: np.ndarray, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search using FAISS index."""
        query_vector = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            metadata = self.metadata_store.get(idx, {})
            
            # Convert FAISS distance to similarity score
            if self.distance_metric == "cosine":
                similarity_score = float(score)  # Inner product after normalization
            else:
                # L2 distance -> similarity (higher is better)
                similarity_score = 1.0 / (1.0 + float(score))
            
            results.append({
                "id": int(idx),
                "score": similarity_score,
                "metadata": metadata,
                "rank": i + 1
            })
        
        return results
    
    async def _search_simple(
        self, 
        query_embedding: np.ndarray, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search using simple cosine similarity."""
        if self.embeddings_matrix.shape[0] == 0:
            return []
        
        # Compute similarities
        if self.distance_metric == "cosine":
            similarities = np.dot(self.embeddings_matrix, query_embedding)
        else:
            # L2 distance
            distances = np.linalg.norm(self.embeddings_matrix - query_embedding, axis=1)
            similarities = 1.0 / (1.0 + distances)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            metadata = self.metadata_store.get(int(idx), {})
            results.append({
                "id": int(idx),
                "score": float(similarities[idx]),
                "metadata": metadata,
                "rank": i + 1
            })
        
        return results
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    async def save_index(self, session_id: str):
        """Save the vector index and metadata to disk."""
        try:
            session_path = self.vector_store_path / session_id
            ensure_directory(session_path)
            
            # Save FAISS index
            if FAISS_AVAILABLE and self.index is not None:
                index_path = session_path / "faiss_index.bin"
                faiss.write_index(self.index, str(index_path))
            else:
                # Save simple embeddings matrix
                embeddings_path = session_path / "embeddings.npy"
                np.save(embeddings_path, self.embeddings_matrix)
            
            # Save metadata
            metadata_path = session_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            # Save index configuration
            config_path = session_path / "index_config.json"
            config = {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "distance_metric": self.distance_metric,
                "id_counter": self.id_counter,
                "saved_at": datetime.utcnow().isoformat(),
                "vector_count": self.index.ntotal if (FAISS_AVAILABLE and self.index) else len(self.embeddings_matrix)
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved vector index for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    async def load_index(self, session_id: str) -> bool:
        """Load vector index and metadata from disk."""
        try:
            session_path = self.vector_store_path / session_id
            if not session_path.exists():
                return False
            
            # Check for both direct path and faiss_index subdirectory
            config_path = session_path / "index_config.json"
            faiss_subdir_config_path = session_path / "faiss_index" / "index_config.json"
            
            if faiss_subdir_config_path.exists():
                # Use the faiss_index subdirectory structure
                config_path = faiss_subdir_config_path
                session_path = session_path / "faiss_index"
            elif not config_path.exists():
                return False
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.dimension = config["dimension"]
            self.index_type = config["index_type"]
            self.distance_metric = config["distance_metric"]
            self.id_counter = config["id_counter"]
            
            # Load index
            if FAISS_AVAILABLE:
                index_path = session_path / "faiss_index.bin"
                if index_path.exists():
                    self.index = faiss.read_index(str(index_path))
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {index_path}")
                else:
                    self._initialize_faiss_index()
                    
                # Always load embeddings_matrix for compatibility
                embeddings_path = session_path / "embeddings.npy"
                if embeddings_path.exists():
                    self.embeddings_matrix = np.load(embeddings_path)
                else:
                    self.embeddings_matrix = np.array([]).reshape(0, self.dimension)
            else:
                embeddings_path = session_path / "embeddings.npy"
                if embeddings_path.exists():
                    self.embeddings_matrix = np.load(embeddings_path)
                else:
                    self.embeddings_matrix = np.array([]).reshape(0, self.dimension)
            
            # Load metadata
            metadata_path = session_path / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
            else:
                self.metadata_store = {}
            
            logger.info(f"Loaded vector index for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    async def delete_vectors(self, vector_ids: List[int]):
        """Delete vectors from the index (not supported by all FAISS indices)."""
        # This is complex with FAISS indices, so we'll mark for rebuild
        logger.warning("Vector deletion requires index rebuild")
        
        # Remove from metadata
        for vid in vector_ids:
            if vid in self.metadata_store:
                del self.metadata_store[vid]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if FAISS_AVAILABLE and self.index is not None:
            vector_count = self.index.ntotal
        else:
            vector_count = len(self.embeddings_matrix)
        
        return {
            "vector_count": vector_count,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "distance_metric": self.distance_metric,
            "metadata_count": len(self.metadata_store),
            "using_faiss": FAISS_AVAILABLE and self.index is not None
        }
    
    def clear(self):
        """Clear the vector store."""
        self._initialize_index()
        self.metadata_store.clear()
        self.id_counter = 0
        logger.info("Cleared vector store")
