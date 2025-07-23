"""
NeuroDoc Configuration Settings

Contains configuration settings for the NeuroDoc RAG system.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Data paths
DATA_PATHS = {
    "raw": str(DATA_DIR / "raw"),
    "processed": str(DATA_DIR / "processed"),
    "embeddings": str(DATA_DIR / "embeddings"),
    "vector_store": str(DATA_DIR / "vector_store"),
    "sessions": str(DATA_DIR / "sessions"),
    "logs": str(PROJECT_ROOT / "logs")
}

# Document processing settings
PROCESSING_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "min_chunk_size": 100,
    "max_chunk_size": 2000,
    "separators": ["\n\n", "\n", ". ", " ", ""],
}

# Embedding model settings
EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",  # Sentence-BERT model
    "batch_size": 32,
    "max_length": 512,
    "normalize_embeddings": True,
}

# Alternative embedding models
EMBEDDING_MODELS = {
    "mini": "all-MiniLM-L6-v2",           # Fast, 384 dimensions
    "base": "all-mpnet-base-v2",          # Balanced, 768 dimensions
    "large": "all-MiniLM-L12-v2",         # Larger, 384 dimensions
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
}

# Retrieval settings
RETRIEVAL_CONFIG = {
    "index_type": "flat",                 # "flat" or "ivf"
    "distance_metric": "cosine",          # "cosine" or "l2"
    "top_k_retrieval": 10,
    "dense_weight": 0.7,                  # Weight for dense retrieval
    "sparse_weight": 0.3,                 # Weight for sparse (BM25) retrieval
    "min_score_threshold": 0.1,
}

# LLM settings - Updated for Local Gemma
LLM_CONFIG = {
    "model_name": "gemma3:latest",  # Ollama Gemma model name
    "temperature": 0.1,
    "max_tokens": 1500,
    "top_p": 0.9,
    "max_context_length": 4000,
    "max_retrieved_chunks": 5,
    "use_openai": False,  # Disabled OpenAI
    "use_local_model": False,  # Disabled HuggingFace local models
    "use_ollama": True,  # Use Ollama specifically
    "ollama_base_url": "http://localhost:11434",  # Default Ollama URL
    "local_model_name": "gemma3:latest",
    "alternative_models": {
        "small": "gemma3:latest",       # Available Gemma model
        "medium": "gemma3:latest",      # Available Gemma model
        "large": "gemma3:latest",       # Available Gemma model
        "instruct": "gemma3:latest",    # Available Gemma model
        "code": "gemma3:latest",        # Available Gemma model
        "fast": "gemma3:latest"         # Available Gemma model
    },
    "openai_api_key": os.getenv("OPENAI_API_KEY"),  # Keep for fallback if needed
}

# Memory settings
MEMORY_CONFIG = {
    "max_conversation_turns": 50,
    "session_timeout_hours": 24,
    "max_sessions_per_user": 10,
    "auto_save_interval_minutes": 5,
    "context_window_size": 4000,
}

# API settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info",
}

# Additional LLM settings for Phase 3
PHASE3_LLM_CONFIG = {
    "reasoning_model": "gemma3:latest",   # Model for multi-step reasoning
    "quality_model": "gemma3:latest",     # Model for quality assessment
    "citation_model": "gemma3:latest",    # Model for citation generation
    "fallback_model": "gemma3:latest",    # Fallback if primary model fails
    "model_timeout": 30,  # Timeout in seconds
    "max_retries": 3,
}

# Memory settings
MEMORY_CONFIG = {
    "max_conversation_length": 10,
    "memory_decay_factor": 0.95,
    "context_window_size": 4000,
}

# API settings
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info",
}

# Environment variables
def get_env_config() -> Dict[str, Optional[str]]:
    """Get environment configuration."""
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", EMBEDDING_CONFIG["model_name"]),
        "chunk_size": int(os.getenv("CHUNK_SIZE", PROCESSING_CONFIG["chunk_size"])),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", PROCESSING_CONFIG["chunk_overlap"])),
    }

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": DATA_PATHS["logs"] + "/neurodoc.log",
            "mode": "a",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["default", "file"],
    },
    "loggers": {
        "neurodoc": {
            "level": "INFO",
            "handlers": ["default", "file"],
            "propagate": False,
        },
    },
}

# Ensure directories exist
for path in DATA_PATHS.values():
    Path(path).mkdir(parents=True, exist_ok=True)

class Config:
    """
    Configuration class for NeuroDoc Phase 3 advanced components.
    Provides centralized access to all configuration settings.
    """
    
    def __init__(self):
        # Core configuration
        self.data_paths = DATA_PATHS
        self.processing_config = PROCESSING_CONFIG
        self.embedding_config = EMBEDDING_CONFIG
        self.embedding_models = EMBEDDING_MODELS
        self.retrieval_config = RETRIEVAL_CONFIG
        self.llm_config = LLM_CONFIG
        self.phase3_llm_config = PHASE3_LLM_CONFIG
        self.api_config = API_CONFIG
        self.logging_config = LOGGING_CONFIG
        
        # Phase 3 specific settings
        self.phase3_config = {
            "reasoning": {
                "max_steps": 5,
                "confidence_threshold": 0.7,
                "strategies": ["decomposition", "synthesis", "validation"]
            },
            "citation": {
                "confidence_threshold": 0.8,
                "max_citations_per_response": 10,
                "styles": ["apa", "mla", "chicago", "ieee"]
            },
            "quality": {
                "dimensions": ["relevance", "accuracy", "completeness", "clarity"],
                "score_threshold": 0.6,
                "assessment_timeout": 30  # seconds
            },
            "generation": {
                "model_selection_strategy": "adaptive",
                "context_window_size": 4000,
                "response_max_length": 2000,
                "use_local_llm": True,  # Force local LLM usage
                "ollama_models": LLM_CONFIG["alternative_models"]
            }
        }
    
    def get_phase3_setting(self, category: str, key: str, default=None):
        """Get a Phase 3 specific configuration setting."""
        return self.phase3_config.get(category, {}).get(key, default)
    
    def update_phase3_setting(self, category: str, key: str, value):
        """Update a Phase 3 specific configuration setting."""
        if category not in self.phase3_config:
            self.phase3_config[category] = {}
        self.phase3_config[category][key] = value
