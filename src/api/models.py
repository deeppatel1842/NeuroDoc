"""
NeuroDoc API Models

This module defines Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class SessionStatus(str, Enum):
    """Session status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"


# Request Models
class QueryRequest(BaseModel):
    """Request model for document queries."""
    question: str = Field(..., description="The question to ask", min_length=1, max_length=1000)
    top_k: Optional[int] = Field(10, description="Number of chunks to retrieve", ge=1, le=50)
    context_limit: Optional[int] = Field(5, description="Number of previous conversations to include", ge=0, le=20)
    include_metadata: Optional[bool] = Field(True, description="Whether to include document metadata")
    prioritize_session_docs: Optional[bool] = Field(True, description="Whether to prioritize documents uploaded in this session")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the key benefits of retrieval-augmented generation?",
                "top_k": 10,
                "context_limit": 5,
                "include_metadata": True,
                "prioritize_session_docs": True
            }
        }


class SessionRequest(BaseModel):
    """Request model for creating a new session."""
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional session metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "metadata": {"purpose": "research", "domain": "AI"}
            }
        }


# Response Models
class Citation(BaseModel):
    """Citation information for a source."""
    document_id: str = Field(..., description="Document identifier")
    chunk_id: str = Field(..., description="Chunk identifier")
    text_snippet: str = Field(..., description="Relevant text snippet")
    relevance_score: float = Field(..., description="Relevance score for this citation", ge=0.0, le=1.0)
    # Optional fields that may not always be present
    document_name: Optional[str] = Field(None, description="Original document filename")
    page_number: Optional[int] = Field(None, description="Page number if available")
    confidence_score: Optional[float] = Field(None, description="Confidence score for this citation", ge=0.0, le=1.0)


class DocumentInfo(BaseModel):
    """Document information."""
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    upload_timestamp: datetime = Field(..., description="When the document was uploaded")
    chunk_count: int = Field(..., description="Number of chunks in the document")
    file_size: int = Field(..., description="File size in bytes")


class SourceDocument(BaseModel):
    """Source document information."""
    index: int = Field(..., description="Document index in results")
    document_id: str = Field(..., description="Document identifier")
    chunk_id: str = Field(..., description="Chunk identifier")
    relevance_score: float = Field(..., description="Relevance score", ge=0.0, le=1.0)
    text_preview: str = Field(..., description="Text preview")
    retrieval_type: str = Field(..., description="Type of retrieval used")


class QueryResponse(BaseModel):
    """Response model for document queries."""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(..., description="Source citations")
    source_documents: List[SourceDocument] = Field(..., description="List of source documents with details")
    confidence_score: float = Field(..., description="Overall confidence score", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(..., description="Response metadata including processing details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Retrieval-augmented generation combines the knowledge stored in a retrieval system...",
                "citations": [
                    {
                        "document_id": "doc_123",
                        "chunk_id": "chunk_456",
                        "text_snippet": "RAG models leverage external knowledge...",
                        "relevance_score": 0.95,
                        "document_name": "rag_paper.pdf",
                        "page_number": 3,
                        "confidence_score": 0.95
                    }
                ],
                "source_documents": [
                    {
                        "index": 1,
                        "document_id": "doc_123",
                        "chunk_id": "chunk_456",
                        "relevance_score": 0.95,
                        "text_preview": "RAG models leverage external knowledge...",
                        "retrieval_type": "semantic"
                    }
                ],
                "confidence_score": 0.87,
                "metadata": {
                    "model_used": "gemma3:latest",
                    "processing_time": 1.23,
                    "context_length": 1500,
                    "retrieved_chunks": 5,
                    "session_id": "session_789"
                }
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="Generated document identifier")
    filename: str = Field(..., description="Uploaded filename")
    status: DocumentStatus = Field(..., description="Processing status")
    chunk_count: int = Field(..., description="Number of chunks created")
    processing_time: float = Field(..., description="Processing time in seconds")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123456",
                "filename": "research_paper.pdf",
                "status": "processed",
                "chunk_count": 45,
                "processing_time": 3.21,
                "message": "Document processed successfully"
            }
        }


class SessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str = Field(..., description="Generated session identifier")
    created_at: datetime = Field(..., description="Session creation timestamp")
    status: SessionStatus = Field(..., description="Session status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123456",
                "created_at": "2025-07-22T10:30:00Z",
                "status": "active"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    components: Dict[str, str] = Field(..., description="Component health status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-07-22T10:30:00Z",
                "components": {
                    "document_processor": "healthy",
                    "retriever": "healthy",
                    "response_generator": "healthy"
                }
            }
        }


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    timestamp: datetime = Field(..., description="Turn timestamp")
    question: str = Field(..., description="User question")
    answer: str = Field(..., description="System answer")
    citations: List[Citation] = Field(..., description="Answer citations")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ProcessingError",
                "message": "Failed to process document",
                "details": {"file_type": "unsupported"},
                "timestamp": "2025-07-22T10:30:00Z"
            }
        }


# Phase 3 Advanced Request Models

class AdvancedQueryRequest(BaseModel):
    """Advanced request model for sophisticated queries with reasoning and citations."""
    question: str = Field(..., description="The question to ask", min_length=1, max_length=1000)
    response_type: Optional[str] = Field("direct_answer", description="Type of response needed")
    complexity_level: Optional[str] = Field("moderate", description="Complexity level for response")
    citation_style: Optional[str] = Field("academic", description="Citation style to use")
    max_length: Optional[int] = Field(1000, description="Maximum response length", ge=50, le=5000)
    include_sources: Optional[bool] = Field(True, description="Whether to include sources")
    reasoning_strategy: Optional[str] = Field(None, description="Specific reasoning strategy to use")
    reasoning_depth: Optional[int] = Field(3, description="Depth of reasoning", ge=1, le=10)
    quality_threshold: Optional[float] = Field(0.5, description="Quality threshold for filtering", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(10, description="Number of chunks to retrieve", ge=1, le=50)
    context_limit: Optional[int] = Field(5, description="Number of previous conversations to include", ge=0, le=20)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Compare machine learning and deep learning approaches for natural language processing",
                "response_type": "comparative",
                "complexity_level": "complex",
                "citation_style": "apa",
                "max_length": 1500,
                "include_sources": True,
                "reasoning_strategy": "comparative_reasoning",
                "reasoning_depth": 4,
                "quality_threshold": 0.7,
                "top_k": 15
            }
        }


class ReasoningRequest(BaseModel):
    """Request model for multi-step reasoning."""
    query: str = Field(..., description="Query for reasoning", min_length=1, max_length=1000)
    strategy: Optional[str] = Field(None, description="Reasoning strategy to use")
    depth: Optional[int] = Field(3, description="Reasoning depth", ge=1, le=10)
    confidence_threshold: Optional[float] = Field(0.7, description="Confidence threshold", ge=0.0, le=1.0)
    allow_speculation: Optional[bool] = Field(False, description="Allow speculative reasoning")
    domain_context: Optional[str] = Field(None, description="Domain context for reasoning")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the causal relationships between climate change and biodiversity loss?",
                "strategy": "causal_reasoning",
                "depth": 4,
                "confidence_threshold": 0.7,
                "domain_context": "environmental_science"
            }
        }


class CitationRequest(BaseModel):
    """Request model for citation generation."""
    query: str = Field(..., description="Query for citations")
    style: Optional[str] = Field("academic", description="Citation style")
    max_citations: Optional[int] = Field(10, description="Maximum number of citations", ge=1, le=20)
    quality_threshold: Optional[float] = Field(0.5, description="Quality threshold", ge=0.0, le=1.0)
    group_by_topic: Optional[bool] = Field(True, description="Group citations by topic")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "artificial intelligence in healthcare",
                "style": "apa",
                "max_citations": 8,
                "quality_threshold": 0.6,
                "group_by_topic": True
            }
        }


class QualityAssessmentRequest(BaseModel):
    """Request model for quality assessment."""
    content: str = Field(..., description="Content to assess", min_length=1)
    content_type: str = Field(..., description="Type of content being assessed")
    dimensions: Optional[List[str]] = Field(None, description="Specific quality dimensions to assess")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for assessment")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Artificial intelligence is transforming healthcare by enabling more accurate diagnoses and personalized treatments.",
                "content_type": "response",
                "dimensions": ["accuracy", "completeness", "clarity"],
                "context": {"query": "How is AI transforming healthcare?"}
            }
        }


# Phase 3 Response Models

class ReasoningStepResponse(BaseModel):
    """Response model for individual reasoning steps."""
    step_type: str = Field(..., description="Type of reasoning step")
    content: str = Field(..., description="Step content")
    confidence: float = Field(..., description="Confidence score")
    evidence_count: int = Field(..., description="Number of evidence pieces")
    assumptions: List[str] = Field(..., description="Assumptions made")


class ReasoningResponse(BaseModel):
    """Response model for reasoning results."""
    strategy: str = Field(..., description="Reasoning strategy used")
    steps: List[ReasoningStepResponse] = Field(..., description="Reasoning steps")
    final_conclusion: str = Field(..., description="Final conclusion")
    overall_confidence: float = Field(..., description="Overall confidence score")
    reasoning_path: List[str] = Field(..., description="Reasoning path description")
    alternative_paths: List[Dict[str, Any]] = Field(..., description="Alternative reasoning paths")
    validation_results: Dict[str, Any] = Field(..., description="Validation results")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")


class CitationResponse(BaseModel):
    """Response model for individual citations."""
    id: str = Field(..., description="Citation ID")
    citation_text: str = Field(..., description="Formatted citation")
    inline_citation: str = Field(..., description="Inline citation format")
    source_type: str = Field(..., description="Type of source")
    relevance_score: float = Field(..., description="Relevance score")
    quality_score: float = Field(..., description="Quality score")
    verification_status: str = Field(..., description="Verification status")
    supporting_quotes: List[str] = Field(..., description="Supporting quotes")
    page_reference: Optional[str] = Field(None, description="Page reference")


class BibliographyResponse(BaseModel):
    """Response model for bibliography."""
    citations: List[CitationResponse] = Field(..., description="List of citations")
    style: str = Field(..., description="Citation style used")
    total_citations: int = Field(..., description="Total number of citations")
    statistics: Dict[str, Any] = Field(..., description="Bibliography statistics")
    grouped_citations: List[Dict[str, Any]] = Field(..., description="Citations grouped by topic")


class QualityMetricResponse(BaseModel):
    """Response model for quality metrics."""
    dimension: str = Field(..., description="Quality dimension")
    score: float = Field(..., description="Quality score")
    confidence: float = Field(..., description="Assessment confidence")
    explanation: str = Field(..., description="Explanation of score")
    evidence: List[str] = Field(..., description="Evidence for score")
    suggestions: List[str] = Field(..., description="Improvement suggestions")


class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment."""
    content_id: str = Field(..., description="Content identifier")
    content_type: str = Field(..., description="Type of content assessed")
    metrics: List[QualityMetricResponse] = Field(..., description="Quality metrics")
    overall_score: float = Field(..., description="Overall quality score")
    overall_level: str = Field(..., description="Overall quality level")
    assessment_summary: str = Field(..., description="Assessment summary")
    strengths: List[str] = Field(..., description="Identified strengths")
    weaknesses: List[str] = Field(..., description="Identified weaknesses")
    improvement_suggestions: List[str] = Field(..., description="Improvement suggestions")
    timestamp: str = Field(..., description="Assessment timestamp")


class AdvancedQueryResponse(BaseModel):
    """Advanced response model with reasoning, citations, and quality assessment."""
    answer: str = Field(..., description="Generated answer")
    confidence_score: float = Field(..., description="Confidence in the answer")
    response_type: str = Field(..., description="Type of response generated")
    complexity_level: str = Field(..., description="Complexity level used")
    
    # Reasoning information
    reasoning_chain: Optional[ReasoningResponse] = Field(None, description="Reasoning chain used")
    reasoning_time: float = Field(..., description="Time spent on reasoning")
    
    # Citation information
    citations: List[CitationResponse] = Field(..., description="Citations for the response")
    citation_count: int = Field(..., description="Number of citations")
    
    # Quality assessment
    quality_assessment: QualityAssessmentResponse = Field(..., description="Quality assessment")
    
    # Source information
    sources_used: List[str] = Field(..., description="Sources used in generation")
    retrieved_chunks: List[Dict[str, Any]] = Field(..., description="Retrieved document chunks")
    
    # Metadata
    generation_time: float = Field(..., description="Total generation time")
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(..., description="Response timestamp")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
