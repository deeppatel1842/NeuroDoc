"""
NeuroDoc LLM Module

This module provides large language model integration, advanced response generation,
multi-step reasoning, citation management, and quality assessment capabilities
for the NeuroDoc RAG system.
"""

from .generator import ResponseGenerator
from .advanced_generator import AdvancedResponseGenerator, ResponseContext, GeneratedResponse
from .reasoning_engine import ReasoningEngine, ReasoningContext, ReasoningChain, ReasoningStrategy
from .citation_manager import CitationManager, Citation, Bibliography, CitationStyle
from .quality_assessor import QualityAssessor, QualityAssessment, QualityDimension

__all__ = [
    'ResponseGenerator',
    'AdvancedResponseGenerator',
    'ResponseContext',
    'GeneratedResponse',
    'ReasoningEngine',
    'ReasoningContext',
    'ReasoningChain',
    'ReasoningStrategy',
    'CitationManager',
    'Citation',
    'Bibliography',
    'CitationStyle',
    'QualityAssessor',
    'QualityAssessment',
    'QualityDimension'
]
