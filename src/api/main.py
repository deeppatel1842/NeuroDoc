"""
NeuroDoc FastAPI Main Application

Contains the main FastAPI application with all API endpoints
for the NeuroDoc RAG system.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, Optional
import logging
import uuid
import os
import json
from datetime import datetime

from ..config import API_CONFIG
from .models import (
    QueryRequest, 
    QueryResponse, 
    DocumentUploadResponse,
    HealthResponse,
    SessionRequest,
    SessionResponse,
    AdvancedQueryRequest,
    AdvancedQueryResponse,
    ReasoningRequest,
    ReasoningResponse,
    ReasoningStepResponse,
    CitationRequest,
    BibliographyResponse,
    QualityAssessmentRequest,
    QualityAssessmentResponse,
    QualityMetricResponse,
    CitationResponse
)
from .dependencies import (
    get_current_session, 
    validate_file_type,
    get_advanced_generator,
    get_reasoning_engine,
    get_citation_manager,
    get_quality_assessor
)
from ..document_processing.processor import DocumentProcessor
from ..retrieval.hybrid_retriever import HybridRetriever
from ..memory.session_manager import SessionManager
from ..llm.generator import ResponseGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NeuroDoc API",
    description="Advanced RAG system for document-based question answering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files
static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Core components
document_processor = DocumentProcessor()
retriever = HybridRetriever()
session_manager = SessionManager()
response_generator = ResponseGenerator()

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    await retriever.initialize()

# Initialize Phase 3 advanced components
from ..llm import (
    AdvancedResponseGenerator,
    ReasoningEngine,
    CitationManager,
    QualityAssessor
)
from ..config import Config

# Create global config instance
config = Config()

# Initialize advanced components
advanced_generator = AdvancedResponseGenerator(config)
reasoning_engine = ReasoningEngine(config)
citation_manager = CitationManager(config)
quality_assessor = QualityAssessor(config)

# Set global instances in dependencies module for dependency injection
import sys
from . import dependencies
dependencies.advanced_generator = advanced_generator
dependencies.reasoning_engine = reasoning_engine
dependencies.citation_manager = citation_manager
dependencies.quality_assessor = quality_assessor

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint serving the chat interface."""
    static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static", "index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path)
    else:
        return {
            "message": "Welcome to NeuroDoc API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "chat": "Chat interface not found"
        }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring system status."""
    try:
        # Check core component health
        processor_status = await document_processor.health_check()
        retriever_status = await retriever.health_check()
        generator_status = await response_generator.health_check()
        
        overall_status = "healthy" if all([
            processor_status, retriever_status, generator_status
        ]) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            components={
                "document_processor": "healthy" if processor_status else "unhealthy",
                "retriever": "healthy" if retriever_status else "unhealthy",
                "response_generator": "healthy" if generator_status else "unhealthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            components={"error": str(e)}
        )

@app.post("/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Depends(get_current_session)
):
    """
    Upload and process a PDF document.
    
    Args:
        file: PDF file to upload
        session_id: Current session identifier
        
    Returns:
        DocumentUploadResponse with processing status and document ID
    """
    try:
        # Validate file type
        if not validate_file_type(file):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Read file content
        content = await file.read()
        
        # Process document
        processing_result = await document_processor.process_document(
            content=content,
            filename=file.filename,
            document_id=document_id,
            session_id=session_id
        )
        
        # Load the processed chunks and add to vector store
        session_dir = document_processor.processed_docs_path / session_id
        chunks_path = session_dir / f"{document_id}_chunks.json"
        
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Add document to retriever's vector store
            await retriever.index_documents(
                session_id=session_id,
                document_id=document_id,
                chunks=chunks
            )
            logger.info(f"Added document {document_id} to vector store for session {session_id}")
        
        # Add document to session tracking
        await session_manager.add_document_to_session(session_id, document_id)
        
        logger.info(f"Successfully processed document {document_id} for session {session_id}")
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="processed",
            chunk_count=processing_result["chunk_count"],
            processing_time=processing_result["processing_time"],
            message="Document processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(
    request: QueryRequest,
    session_id: str = Depends(get_current_session)
):
    """
    Query processed documents using RAG.
    
    Args:
        request: Query request containing question and optional parameters
        session_id: Current session identifier
        
    Returns:
        QueryResponse with answer, citations, and metadata
    """
    try:
        start_time = datetime.utcnow()
        
        # Get conversation history for context
        conversation_history = await session_manager.get_conversation_history(
            session_id, limit=request.context_limit or 5
        )
        
        # Get session information for document filtering
        session = await session_manager.get_session(session_id)
        session_document_ids = session.document_ids if session else []
        
        # Always prioritize user documents if they exist (dual-mode logic)
        use_user_docs = len(session_document_ids) > 0
        
        # Log the mode being used
        if use_user_docs:
            logger.info(f"Using USER MODE: {len(session_document_ids)} documents uploaded")
        else:
            logger.info("Using DEFAULT MODE: No user documents, will search pre-loaded knowledge base")
        
        # Extract just the questions from conversation history for context
        conversation_questions = []
        if conversation_history:
            for turn in conversation_history:
                if isinstance(turn, dict) and "question" in turn:
                    conversation_questions.append(turn["question"])
                elif isinstance(turn, str):
                    conversation_questions.append(turn)
        
        # Retrieve relevant documents with session document prioritization
        retrieval_results = await retriever.retrieve(
            query=request.question,
            session_id=session_id,
            top_k=request.top_k or 15,  # Increased from 10 to 15 for better coverage
            conversation_history=conversation_questions,
            session_document_ids=session_document_ids if use_user_docs else None
        )
        
        # Generate response
        response = await response_generator.generate_response(
            question=request.question,
            retrieved_docs=retrieval_results,
            conversation_history=conversation_history,
            session_id=session_id
        )
        
        # Store in conversation history
        await session_manager.add_to_conversation(
            session_id=session_id,
            question=request.question,
            answer=response["answer"],
            citations=response["citations"],
            metadata=response["metadata"]
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Query processed for session {session_id} in {processing_time:.2f}s")
        
        # Add processing details to metadata
        response["metadata"]["endpoint_processing_time"] = processing_time
        response["metadata"]["total_retrieved_chunks"] = len(retrieval_results)
        
        return QueryResponse(
            answer=response["answer"],
            citations=response["citations"],
            source_documents=response["source_documents"],
            confidence_score=response["confidence_score"],
            metadata=response["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/session", response_model=SessionResponse, tags=["Session"])
async def create_session(request: SessionRequest):
    """
    Create new session for document interaction.
    
    Args:
        request: Session creation request
        
    Returns:
        SessionResponse with new session ID
    """
    try:
        session_id = await session_manager.create_session(
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        logger.info(f"Created new session {session_id}")
        
        return SessionResponse(
            session_id=session_id,
            created_at=datetime.utcnow(),
            status="active"
        )
        
    except Exception as e:
        logger.error(f"Session creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@app.get("/session/{session_id}/history", tags=["Session"])
async def get_session_history(session_id: str, limit: int = 10):
    """
    Get conversation history for a session.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of conversation turns to return
        
    Returns:
        List of conversation turns
    """
    try:
        history = await session_manager.get_conversation_history(session_id, limit)
        return {"session_id": session_id, "history": history}
        
    except Exception as e:
        logger.error(f"Failed to get session history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session history: {str(e)}")

@app.delete("/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """
    Delete a session and its associated data.
    
    Args:
        session_id: Session identifier to delete
        
    Returns:
        Deletion confirmation
    """
    try:
        await session_manager.delete_session(session_id)
        logger.info(f"Deleted session {session_id}")
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Session deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Session deletion failed: {str(e)}")

@app.get("/documents", tags=["Documents"])
async def list_documents(session_id: str = Depends(get_current_session)):
    """
    List all documents in the current session.
    
    Args:
        session_id: Current session identifier
        
    Returns:
        List of documents with metadata
    """
    try:
        documents = await document_processor.list_documents(session_id)
        return {"session_id": session_id, "documents": documents}
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    session_id: str = Depends(get_current_session)
):
    """
    Delete a document and its associated embeddings.
    
    Args:
        document_id: Document identifier to delete
        session_id: Current session identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        await document_processor.delete_document(document_id, session_id)
        logger.info(f"Deleted document {document_id} from session {session_id}")
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Document deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document deletion failed: {str(e)}")


# Phase 3 Advanced LLM Endpoints

@app.post("/query/advanced", response_model=AdvancedQueryResponse, tags=["Advanced Query"])
async def advanced_query(
    request: AdvancedQueryRequest,
    session_id: str = Depends(get_current_session),
    advanced_gen: AdvancedResponseGenerator = Depends(get_advanced_generator),
    reasoning_eng: ReasoningEngine = Depends(get_reasoning_engine),
    citation_mgr: CitationManager = Depends(get_citation_manager),
    quality_assess: QualityAssessor = Depends(get_quality_assessor)
):
    """
    Process an advanced query with multi-step reasoning, high-quality citations, and quality assessment.
    
    Args:
        request: Advanced query request with parameters
        session_id: Current session identifier
        
    Returns:
        Comprehensive response with reasoning, citations, and quality metrics
    """
    try:
        start_time = datetime.utcnow()
        
        logger.info(f"🔍 Step 1: Starting retrieval for question: {request.question[:50]}...")
        # Retrieve relevant documents (without conversation history)
        try:
            retrieval_results = await retriever.retrieve(
                query=request.question,
                session_id=session_id,
                top_k=request.top_k,
                conversation_history=[]  # Empty list instead of None
            )
            logger.info(f"✅ Step 1 complete: Retrieved {len(retrieval_results)} documents")
        except Exception as e:
            logger.error(f"❌ Step 1 failed - Retrieval error: {e}")
            logger.error(f"❌ Retrieval error type: {type(e)}")
            import traceback
            logger.error(f"❌ Retrieval traceback: {traceback.format_exc()}")
            raise
        
        
        logger.info("🔍 Step 2: Preparing response context...")
        # Prepare response context
        try:
            from ..llm.advanced_generator import ResponseContext, ResponseType, ResponseComplexity
            
            response_type_map = {
                "direct_answer": ResponseType.DIRECT_ANSWER,
                "analytical": ResponseType.ANALYTICAL,
                "comparative": ResponseType.COMPARATIVE,
                "summary": ResponseType.SUMMARY,
                "explanatory": ResponseType.EXPLANATORY,
                "procedural": ResponseType.PROCEDURAL
            }
            
            complexity_map = {
                "simple": ResponseComplexity.SIMPLE,
                "moderate": ResponseComplexity.MODERATE,
                "complex": ResponseComplexity.COMPLEX,
                "expert": ResponseComplexity.EXPERT
            }
            
            response_context = ResponseContext(
                query=request.question,
                retrieved_chunks=retrieval_results,  # Direct list, not ["chunks"]
                conversation_history=[],
                response_type=response_type_map.get(request.response_type, ResponseType.DIRECT_ANSWER),
                complexity_level=complexity_map.get(request.complexity_level, ResponseComplexity.MODERATE),
                citation_style=request.citation_style,
                max_length=request.max_length,
                include_sources=request.include_sources
            )
            logger.info("✅ Step 2 complete: Response context created")
        except Exception as e:
            logger.error(f"❌ Step 2 failed - Response context error: {e}")
            logger.error(f"❌ Response context error type: {type(e)}")
            import traceback
            logger.error(f"❌ Response context traceback: {traceback.format_exc()}")
            raise
        
        logger.info("🔍 Step 3: Generating advanced response...")
        # Generate advanced response
        try:
            generated_response = await advanced_gen.generate_response(response_context)
            logger.info("✅ Step 3 complete: Advanced response generated")
        except Exception as e:
            logger.error(f"❌ Step 3 failed - Advanced response error: {e}")
            logger.error(f"❌ Advanced response error type: {type(e)}")
            import traceback
            logger.error(f"❌ Advanced response traceback: {traceback.format_exc()}")
            raise
        
        logger.info("🔍 Step 4: Generating citations...")
        # Generate citations
        try:
            from ..llm.citation_manager import CitationStyle
            citation_style_map = {
                "apa": CitationStyle.APA,
                "mla": CitationStyle.MLA,
                "chicago": CitationStyle.CHICAGO,
                "ieee": CitationStyle.IEEE,
                "harvard": CitationStyle.HARVARD,
                "vancouver": CitationStyle.VANCOUVER,
                "academic": CitationStyle.ACADEMIC,
                "simple": CitationStyle.SIMPLE
            }
            
            citations = await citation_mgr.generate_citations(
                retrieved_chunks=retrieval_results,  # Direct list, not ["chunks"]
                query=request.question,
                style=citation_style_map.get(request.citation_style, CitationStyle.ACADEMIC),
                max_citations=10,
                quality_threshold=request.quality_threshold
            )
            logger.info("✅ Step 4 complete: Citations generated")
        except Exception as e:
            logger.error(f"❌ Step 4 failed - Citation generation error: {e}")
            logger.error(f"❌ Citation error type: {type(e)}")
            import traceback
            logger.error(f"❌ Citation traceback: {traceback.format_exc()}")
            raise
        
        logger.info("🔍 Step 5: Assessing quality...")
        # Assess response quality
        try:
            from ..llm.quality_assessor import ContentType
            quality_assessment = await quality_assess.assess_content(
                content=generated_response.content,
                content_type=ContentType.RESPONSE,
                context={"query": request.question}
            )
            logger.info("✅ Step 5 complete: Quality assessment done")
        except Exception as e:
            logger.error(f"❌ Step 5 failed - Quality assessment error: {e}")
            logger.error(f"❌ Quality error type: {type(e)}")
            import traceback
            logger.error(f"❌ Quality traceback: {traceback.format_exc()}")
            raise
        
        # Save to conversation history
        logger.info("🔍 Step 6: Saving to conversation history...")
        try:
            # Convert Citation objects to dictionaries for serialization
            citations_dict = []
            for citation in citations:
                citation_dict = {
                    "id": citation.id,
                    "citation_text": citation.citation_text,
                    "inline_citation": citation.inline_citation,
                    "content_snippet": citation.content_snippet,
                    "relevance_score": citation.relevance_score,
                    "confidence_score": citation.confidence_score,
                    "quality_score": citation.quality_score,
                    "page_reference": citation.page_reference,
                    "section_reference": citation.section_reference,
                    "verification_status": citation.verification_status,
                    "metadata": citation.metadata
                }
                citations_dict.append(citation_dict)
            
            await session_manager.add_to_conversation(
                session_id=session_id,
                question=request.question,
                answer=generated_response.content,
                citations=citations_dict,
                metadata={
                    "confidence": generated_response.confidence_score,
                    "quality_score": quality_assessment.overall_score
                }
            )
            logger.info("✅ Step 6 complete: Saved to conversation history")
        except ValueError as e:
            logger.error(f"❌ Step 6 ValueError: {e}")
            if "not found" in str(e):
                # Create session if it doesn't exist
                logger.info(f"Creating missing session: {session_id}")
                await session_manager.create_session(session_id=session_id)
                # Try again
                await session_manager.add_to_conversation(
                    session_id=session_id,
                    question=request.question,
                    answer=generated_response.content,
                    citations=citations_dict,
                    metadata={
                        "confidence": generated_response.confidence_score,
                        "quality_score": quality_assessment.overall_score
                    }
                )
                logger.info("✅ Step 6 complete: Session created and saved to conversation history")
            else:
                raise
        except Exception as e:
            logger.error(f"❌ Step 6 failed - Session management error: {e}")
            logger.error(f"❌ Session error type: {type(e)}")
            import traceback
            logger.error(f"❌ Session traceback: {traceback.format_exc()}")
            raise
        
        # Convert objects to response format
        # Helper function to convert numpy types to Python types
        import numpy as np
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            return obj
        
        reasoning_response = None
        if hasattr(generated_response, 'reasoning_chain') and generated_response.reasoning_chain:
            reasoning_response = ReasoningResponse(
                strategy=generated_response.reasoning_chain.strategy.value,
                steps=[
                    ReasoningStepResponse(
                        step_type=step.step_type.value,
                        content=step.content,
                        confidence=convert_numpy_types(step.confidence),
                        evidence_count=len(step.evidence),
                        assumptions=step.assumptions
                    ) for step in generated_response.reasoning_chain.steps
                ],
                final_conclusion=generated_response.reasoning_chain.final_conclusion,
                overall_confidence=convert_numpy_types(generated_response.reasoning_chain.overall_confidence),
                reasoning_path=generated_response.reasoning_chain.reasoning_path,
                alternative_paths=generated_response.reasoning_chain.alternative_paths,
                validation_results=convert_numpy_types(generated_response.reasoning_chain.validation_results),
                metadata=convert_numpy_types(generated_response.reasoning_chain.metadata)
            )
        
        citation_responses = [
            CitationResponse(
                id=citation.id,
                citation_text=citation.citation_text,
                inline_citation=citation.inline_citation,
                source_type=citation.source_metadata.source_type.value,
                relevance_score=convert_numpy_types(citation.relevance_score),
                quality_score=convert_numpy_types(citation.quality_score),
                verification_status=citation.verification_status,
                supporting_quotes=citation.supporting_quotes,
                page_reference=citation.page_reference
            ) for citation in citations
        ]
        
        quality_response = QualityAssessmentResponse(
            content_id=quality_assessment.content_id,
            content_type=quality_assessment.content_type.value,
            metrics=[
                QualityMetricResponse(
                    dimension=metric.dimension.value,
                    score=convert_numpy_types(metric.score),
                    confidence=convert_numpy_types(metric.confidence),
                    explanation=metric.explanation,
                    evidence=convert_numpy_types(metric.evidence),
                    suggestions=convert_numpy_types(metric.suggestions)
                ) for metric in quality_assessment.metrics
            ],
            overall_score=convert_numpy_types(quality_assessment.overall_score),
            overall_level=quality_assessment.overall_level.value,
            assessment_summary=quality_assessment.assessment_summary,
            strengths=quality_assessment.strengths,
            weaknesses=quality_assessment.weaknesses,
            improvement_suggestions=quality_assessment.improvement_suggestions,
            timestamp=quality_assessment.timestamp
        )
        
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        # Wrap final return in try-except to isolate exact failure line
        import traceback
        
        try:
            return AdvancedQueryResponse(
                answer=generated_response.content,
                confidence_score=convert_numpy_types(generated_response.confidence_score),
                response_type=generated_response.response_type.value,
                complexity_level=generated_response.complexity_level.value,
                reasoning_chain=reasoning_response,
                reasoning_time=convert_numpy_types(generated_response.generation_time),
                citations=citation_responses,
                citation_count=len(citations),
                quality_assessment=quality_response,
                sources_used=convert_numpy_types(generated_response.sources_used),
                retrieved_chunks=convert_numpy_types(retrieval_results),  # Fixed: retrieval_results is already a list
                generation_time=convert_numpy_types(total_time),
                session_id=session_id,
                timestamp=end_time.isoformat(),
                metadata=convert_numpy_types({
                    "retrieval_count": len(retrieval_results),  # Fixed: retrieval_results is already a list
                    "context_messages": 0,
                    "reasoning_steps": (
                        len(getattr(getattr(generated_response, "reasoning_chain", None), "steps", []))
                        if getattr(generated_response, "reasoning_chain", None) and getattr(getattr(generated_response, "reasoning_chain", None), "steps", None)
                        else 0
                    )
                })
            )
        except Exception as e:
            logger.error("🔥 ERROR in return block:")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Failed to build AdvancedQueryResponse")
        
    except Exception as e:
        logger.error(f"Advanced query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced query processing failed: {str(e)}")


@app.post("/reasoning", response_model=ReasoningResponse, tags=["Reasoning"])
async def multi_step_reasoning(
    request: ReasoningRequest,
    session_id: str = Depends(get_current_session),
    reasoning_eng: ReasoningEngine = Depends(get_reasoning_engine)
):
    """
    Perform multi-step reasoning on a query using specified strategy.
    
    Args:
        request: Reasoning request with parameters
        session_id: Current session identifier
        
    Returns:
        Detailed reasoning chain with steps and conclusions
    """
    try:
        # Retrieve relevant information
        retrieval_results = await retriever.retrieve(
            query=request.query,
            session_id=session_id,
            top_k=10
        )
        
        # Prepare reasoning context
        from ..llm.reasoning_engine import ReasoningContext, ReasoningStrategy
        
        strategy_map = {
            "chain_of_thought": ReasoningStrategy.CHAIN_OF_THOUGHT,
            "causal_reasoning": ReasoningStrategy.CAUSAL_REASONING,
            "analogical_reasoning": ReasoningStrategy.ANALOGICAL_REASONING,
            "deductive_reasoning": ReasoningStrategy.DEDUCTIVE_REASONING,
            "inductive_reasoning": ReasoningStrategy.INDUCTIVE_REASONING,
            "abductive_reasoning": ReasoningStrategy.ABDUCTIVE_REASONING,
            "comparative_reasoning": ReasoningStrategy.COMPARATIVE_REASONING,
            "hierarchical_reasoning": ReasoningStrategy.HIERARCHICAL_REASONING,
            "meta_reasoning": ReasoningStrategy.META_REASONING
        }
        
        reasoning_context = ReasoningContext(
            query=request.query,
            retrieved_information=retrieval_results["chunks"],
            reasoning_depth=request.depth,
            confidence_threshold=request.confidence_threshold,
            allow_speculation=request.allow_speculation
        )
        
        if request.domain_context:
            reasoning_context.domain_knowledge["context"] = request.domain_context
        
        # Execute reasoning
        strategy = strategy_map.get(request.strategy) if request.strategy else None
        reasoning_chain = await reasoning_eng.reason(reasoning_context, strategy)
        
        return ReasoningResponse(
            strategy=reasoning_chain.strategy.value,
            steps=[
                ReasoningStepResponse(
                    step_type=step.step_type.value,
                    content=step.content,
                    confidence=step.confidence,
                    evidence_count=len(step.evidence),
                    assumptions=step.assumptions
                ) for step in reasoning_chain.steps
            ],
            final_conclusion=reasoning_chain.final_conclusion,
            overall_confidence=reasoning_chain.overall_confidence,
            reasoning_path=reasoning_chain.reasoning_path,
            alternative_paths=reasoning_chain.alternative_paths,
            validation_results=reasoning_chain.validation_results,
            metadata=reasoning_chain.metadata
        )
        
    except Exception as e:
        logger.error(f"Reasoning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")


@app.post("/citations", response_model=BibliographyResponse, tags=["Citations"])
async def generate_bibliography(
    request: CitationRequest,
    session_id: str = Depends(get_current_session),
    citation_mgr: CitationManager = Depends(get_citation_manager)
):
    """
    Generate a bibliography with high-quality citations for a query.
    
    Args:
        request: Citation request with parameters
        session_id: Current session identifier
        
    Returns:
        Formatted bibliography with citations
    """
    try:
        # Retrieve relevant documents
        retrieval_results = await retriever.retrieve(
            query=request.query,
            session_id=session_id,
            top_k=request.max_citations * 2  # Get extra for filtering
        )
        
        # Generate citations
        from ..llm.citation_manager import CitationStyle
        
        style_map = {
            "apa": CitationStyle.APA,
            "mla": CitationStyle.MLA,
            "chicago": CitationStyle.CHICAGO,
            "ieee": CitationStyle.IEEE,
            "harvard": CitationStyle.HARVARD,
            "vancouver": CitationStyle.VANCOUVER,
            "academic": CitationStyle.ACADEMIC,
            "simple": CitationStyle.SIMPLE
        }
        
        citations = await citation_mgr.generate_citations(
            retrieved_chunks=retrieval_results["chunks"],
            query=request.query,
            style=style_map.get(request.style, CitationStyle.ACADEMIC),
            max_citations=request.max_citations,
            quality_threshold=request.quality_threshold
        )
        
        # Create bibliography
        bibliography = await citation_mgr.create_bibliography(
            citations=citations,
            style=style_map.get(request.style, CitationStyle.ACADEMIC),
            group_by_topic=request.group_by_topic
        )
        
        citation_responses = [
            CitationResponse(
                id=citation.id,
                citation_text=citation.citation_text,
                inline_citation=citation.inline_citation,
                source_type=citation.source_metadata.source_type.value,
                relevance_score=citation.relevance_score,
                quality_score=citation.quality_score,
                verification_status=citation.verification_status,
                supporting_quotes=citation.supporting_quotes,
                page_reference=citation.page_reference
            ) for citation in citations
        ]
        
        return BibliographyResponse(
            citations=citation_responses,
            style=bibliography.style.value,
            total_citations=len(citations),
            statistics=bibliography.statistics,
            grouped_citations=[
                {
                    "topic": group.topic,
                    "citations": [c.id for c in group.citations],
                    "confidence": group.group_confidence
                } for group in bibliography.grouped_citations
            ]
        )
        
    except Exception as e:
        logger.error(f"Citation generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Citation generation failed: {str(e)}")


@app.post("/quality/assess", response_model=QualityAssessmentResponse, tags=["Quality"])
async def assess_content_quality(
    request: QualityAssessmentRequest,
    quality_assess: QualityAssessor = Depends(get_quality_assessor)
):
    """
    Assess the quality of content across multiple dimensions.
    
    Args:
        request: Quality assessment request with content and parameters
        
    Returns:
        Comprehensive quality assessment with scores and suggestions
    """
    try:
        from ..llm.quality_assessor import ContentType, QualityDimension
        
        content_type_map = {
            "response": ContentType.RESPONSE,
            "citation": ContentType.CITATION,
            "summary": ContentType.SUMMARY,
            "explanation": ContentType.EXPLANATION,
            "analysis": ContentType.ANALYSIS,
            "comparison": ContentType.COMPARISON
        }
        
        dimensions = None
        if request.dimensions:
            dimension_map = {
                "accuracy": QualityDimension.ACCURACY,
                "completeness": QualityDimension.COMPLETENESS,
                "relevance": QualityDimension.RELEVANCE,
                "clarity": QualityDimension.CLARITY,
                "coherence": QualityDimension.COHERENCE,
                "depth": QualityDimension.DEPTH,
                "objectivity": QualityDimension.OBJECTIVITY,
                "currency": QualityDimension.CURRENCY,
                "credibility": QualityDimension.CREDIBILITY,
                "usefulness": QualityDimension.USEFULNESS
            }
            dimensions = [dimension_map.get(d) for d in request.dimensions if d in dimension_map]
        
        assessment = await quality_assess.assess_content(
            content=request.content,
            content_type=content_type_map.get(request.content_type, ContentType.RESPONSE),
            context=request.context,
            dimensions=dimensions
        )
        
        return QualityAssessmentResponse(
            content_id=assessment.content_id,
            content_type=assessment.content_type.value,
            metrics=[
                QualityMetricResponse(
                    dimension=metric.dimension.value,
                    score=metric.score,
                    confidence=metric.confidence,
                    explanation=metric.explanation,
                    evidence=metric.evidence,
                    suggestions=metric.suggestions
                ) for metric in assessment.metrics
            ],
            overall_score=assessment.overall_score,
            overall_level=assessment.overall_level.value,
            assessment_summary=assessment.assessment_summary,
            strengths=assessment.strengths,
            weaknesses=assessment.weaknesses,
            improvement_suggestions=assessment.improvement_suggestions,
            timestamp=assessment.timestamp
        )
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")


@app.get("/quality/trends/{dimension}", tags=["Quality"])
async def get_quality_trends(
    dimension: str,
    time_window_hours: int = 24,
    quality_assess: QualityAssessor = Depends(get_quality_assessor)
):
    """
    Get quality trends for a specific dimension over time.
    
    Args:
        dimension: Quality dimension to analyze
        time_window_hours: Time window in hours for trend analysis
        
    Returns:
        Quality trend analysis with statistics
    """
    try:
        from ..llm.quality_assessor import QualityDimension
        
        dimension_map = {
            "accuracy": QualityDimension.ACCURACY,
            "completeness": QualityDimension.COMPLETENESS,
            "relevance": QualityDimension.RELEVANCE,
            "clarity": QualityDimension.CLARITY,
            "coherence": QualityDimension.COHERENCE,
            "depth": QualityDimension.DEPTH,
            "objectivity": QualityDimension.OBJECTIVITY,
            "currency": QualityDimension.CURRENCY,
            "credibility": QualityDimension.CREDIBILITY,
            "usefulness": QualityDimension.USEFULNESS
        }
        
        if dimension not in dimension_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dimension. Valid options: {list(dimension_map.keys())}"
            )
        
        trend = await quality_assess.analyze_quality_trends(
            dimension=dimension_map[dimension],
            time_window_hours=time_window_hours
        )
        
        return {
            "dimension": trend.dimension.value,
            "time_series": trend.time_series,
            "trend_direction": trend.trend_direction,
            "trend_strength": trend.trend_strength,
            "change_points": trend.change_points,
            "statistics": trend.statistics
        }
        
    except Exception as e:
        logger.error(f"Quality trend analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quality trend analysis failed: {str(e)}")


# Update health check to include new components
@app.get("/health/advanced", tags=["Health"])
async def advanced_health_check():
    """Advanced health check including Phase 3 components."""
    try:
        # Check all components
        processor_status = await document_processor.health_check()
        retriever_status = await retriever.health_check()
        generator_status = await response_generator.health_check()
        
        # Check Phase 3 components (basic health check)
        advanced_gen_status = advanced_generator is not None
        reasoning_eng_status = reasoning_engine is not None
        citation_mgr_status = citation_manager is not None
        quality_assess_status = quality_assessor is not None
        
        all_healthy = all([
            processor_status, retriever_status, generator_status,
            advanced_gen_status, reasoning_eng_status,
            citation_mgr_status, quality_assess_status
        ])
        
        overall_status = "healthy" if all_healthy else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "document_processor": "healthy" if processor_status else "unhealthy",
                "retriever": "healthy" if retriever_status else "unhealthy",
                "response_generator": "healthy" if generator_status else "unhealthy",
                "advanced_generator": "healthy" if advanced_gen_status else "unhealthy",
                "reasoning_engine": "healthy" if reasoning_eng_status else "unhealthy",
                "citation_manager": "healthy" if citation_mgr_status else "unhealthy",
                "quality_assessor": "healthy" if quality_assess_status else "unhealthy"
            },
            "capabilities": {
                "basic_query": True,
                "advanced_query": all_healthy,
                "multi_step_reasoning": reasoning_eng_status,
                "citation_generation": citation_mgr_status,
                "quality_assessment": quality_assess_status
            }
        }
        
    except Exception as e:
        logger.error(f"Advanced health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


# Server startup
if __name__ == "__main__":
    import sys
    import os
    import uvicorn
    
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Get host and port from environment or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting NeuroDoc server on {host}:{port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
