"""
NeuroDoc API Dependencies

This module contains FastAPI dependencies for request validation,
authentication, and session management.
"""

from fastapi import Header, HTTPException, UploadFile
from typing import Optional, List, Dict, Any
import uuid
import logging

from ..utils.file_utils import FileValidator
from ..llm import (
    AdvancedResponseGenerator, 
    ReasoningEngine, 
    CitationManager, 
    QualityAssessor,
    CitationStyle,
    QualityDimension
)
from ..config import Config

logger = logging.getLogger(__name__)


def get_current_session(
    x_session_id: Optional[str] = Header(None, description="Session ID header")
) -> str:
    """
    Get or create a session ID from request headers.
    
    Args:
        x_session_id: Session ID from X-Session-Id header
        
    Returns:
        Session ID string
        
    Raises:
        HTTPException: If session ID is invalid
    """
    if x_session_id:
        # Validate session ID format
        try:
            uuid.UUID(x_session_id)
            return x_session_id
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid session ID format. Must be a valid UUID."
            )
    else:
        # Generate new session ID if none provided
        new_session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID: {new_session_id}")
        return new_session_id


def validate_file_type(file: UploadFile) -> bool:
    """
    Validate uploaded file type and size.
    
    Args:
        file: Uploaded file to validate
        
    Returns:
        True if file is valid, False otherwise
        
    Raises:
        HTTPException: If file is invalid
    """
    # Check file type
    if not FileValidator.validate_file_type(file.filename or ""):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Only PDF files are supported."
        )
    
    # Check content type
    if file.content_type and not FileValidator.validate_content_type(file.content_type):
        raise HTTPException(
            status_code=415,
            detail=f"Invalid content type: {file.content_type}. Expected application/pdf."
        )
    
    # Check file size if available
    if hasattr(file, 'size') and file.size:
        if not FileValidator.validate_file_size(file.size):
            max_size_mb = FileValidator.MAX_FILE_SIZE // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_size_mb}MB"
            )
    
    return True


def validate_query_params(
    top_k: Optional[int] = None,
    context_limit: Optional[int] = None
) -> dict:
    """
    Validate query parameters.
    
    Args:
        top_k: Number of chunks to retrieve
        context_limit: Number of conversation turns to include
        
    Returns:
        Validated parameters dictionary
        
    Raises:
        HTTPException: If parameters are invalid
    """
    validated_params = {}
    
    if top_k is not None:
        if not (1 <= top_k <= 50):
            raise HTTPException(
                status_code=400,
                detail="top_k must be between 1 and 50"
            )
        validated_params["top_k"] = top_k
    
    if context_limit is not None:
        if not (0 <= context_limit <= 20):
            raise HTTPException(
                status_code=400,
                detail="context_limit must be between 0 and 20"
            )
        validated_params["context_limit"] = context_limit
    
    return validated_params


def get_user_agent(
    user_agent: Optional[str] = Header(None, description="User agent header")
) -> Optional[str]:
    """
    Extract user agent from headers for analytics.
    
    Args:
        user_agent: User agent string from headers
        
    Returns:
        User agent string or None
    """
    return user_agent


def get_client_ip(
    x_forwarded_for: Optional[str] = Header(None, description="X-Forwarded-For header"),
    x_real_ip: Optional[str] = Header(None, description="X-Real-IP header")
) -> Optional[str]:
    """
    Extract client IP address for logging and analytics.
    
    Args:
        x_forwarded_for: X-Forwarded-For header (for load balancers)
        x_real_ip: X-Real-IP header (for proxies)
        
    Returns:
        Client IP address or None
    """
    # Try X-Forwarded-For first (may contain multiple IPs)
    if x_forwarded_for:
        # Take the first IP in the chain
        return x_forwarded_for.split(',')[0].strip()
    
    # Fall back to X-Real-IP
    if x_real_ip:
        return x_real_ip.strip()
    
    return None


def require_session_id(session_id: str = Header(..., description="Required session ID")) -> str:
    """
    Require a valid session ID in the request headers.
    
    Args:
        session_id: Required session ID from headers
        
    Returns:
        Validated session ID
        
    Raises:
        HTTPException: If session ID is missing or invalid
    """
    try:
        uuid.UUID(session_id)
        return session_id
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid session ID format. Must be a valid UUID."
        )


def validate_json_content(content_type: Optional[str] = Header(None)) -> bool:
    """
    Validate that request contains JSON content.
    
    Args:
        content_type: Content-Type header
        
    Returns:
        True if valid JSON content type
        
    Raises:
        HTTPException: If content type is not JSON
    """
    if content_type and "application/json" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be application/json"
        )
    return True


# Global instances (will be initialized in main.py)
advanced_generator: Optional[AdvancedResponseGenerator] = None
reasoning_engine: Optional[ReasoningEngine] = None
citation_manager: Optional[CitationManager] = None
quality_assessor: Optional[QualityAssessor] = None

def get_advanced_generator() -> AdvancedResponseGenerator:
    """
    Get the advanced response generator instance.
    
    Returns:
        AdvancedResponseGenerator instance
        
    Raises:
        HTTPException: If generator not initialized
    """
    if advanced_generator is None:
        raise HTTPException(
            status_code=500,
            detail="Advanced response generator not initialized"
        )
    return advanced_generator


def get_reasoning_engine() -> ReasoningEngine:
    """
    Get the reasoning engine instance.
    
    Returns:
        ReasoningEngine instance
        
    Raises:
        HTTPException: If engine not initialized
    """
    if reasoning_engine is None:
        raise HTTPException(
            status_code=500,
            detail="Reasoning engine not initialized"
        )
    return reasoning_engine


def get_citation_manager() -> CitationManager:
    """
    Get the citation manager instance.
    
    Returns:
        CitationManager instance
        
    Raises:
        HTTPException: If manager not initialized
    """
    if citation_manager is None:
        raise HTTPException(
            status_code=500,
            detail="Citation manager not initialized"
        )
    return citation_manager


def get_quality_assessor() -> QualityAssessor:
    """
    Get the quality assessor instance.
    
    Returns:
        QualityAssessor instance
        
    Raises:
        HTTPException: If assessor not initialized
    """
    if quality_assessor is None:
        raise HTTPException(
            status_code=500,
            detail="Quality assessor not initialized"
        )
    return quality_assessor


def validate_citation_style(style: str) -> CitationStyle:
    """
    Validate and convert citation style string to enum.
    
    Args:
        style: Citation style string
        
    Returns:
        CitationStyle enum value
        
    Raises:
        HTTPException: If citation style is invalid
    """
    try:
        return CitationStyle(style.lower())
    except ValueError:
        valid_styles = [style.value for style in CitationStyle]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid citation style. Valid options: {valid_styles}"
        )


def validate_quality_dimensions(dimensions: Optional[List[str]]) -> Optional[List[QualityDimension]]:
    """
    Validate and convert quality dimension strings to enums.
    
    Args:
        dimensions: List of quality dimension strings
        
    Returns:
        List of QualityDimension enum values or None
        
    Raises:
        HTTPException: If any dimension is invalid
    """
    if not dimensions:
        return None
    
    try:
        return [QualityDimension(dim.lower()) for dim in dimensions]
    except ValueError as e:
        valid_dimensions = [dim.value for dim in QualityDimension]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid quality dimension. Valid options: {valid_dimensions}"
        )


def validate_reasoning_strategy(strategy: Optional[str]) -> Optional[str]:
    """
    Validate reasoning strategy parameter.
    
    Args:
        strategy: Reasoning strategy string
        
    Returns:
        Validated strategy string or None
        
    Raises:
        HTTPException: If strategy is invalid
    """
    if not strategy:
        return None
    
    from ..llm.reasoning_engine import ReasoningStrategy
    
    try:
        ReasoningStrategy(strategy.lower())
        return strategy.lower()
    except ValueError:
        valid_strategies = [s.value for s in ReasoningStrategy]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reasoning strategy. Valid options: {valid_strategies}"
        )


def validate_response_parameters(
    max_length: Optional[int] = None,
    quality_threshold: Optional[float] = None,
    reasoning_depth: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validate response generation parameters.
    
    Args:
        max_length: Maximum response length
        quality_threshold: Quality threshold for filtering
        reasoning_depth: Depth of reasoning
        
    Returns:
        Validated parameters dictionary
        
    Raises:
        HTTPException: If any parameter is invalid
    """
    params = {}
    
    if max_length is not None:
        if max_length < 50 or max_length > 5000:
            raise HTTPException(
                status_code=400,
                detail="max_length must be between 50 and 5000"
            )
        params["max_length"] = max_length
    
    if quality_threshold is not None:
        if quality_threshold < 0.0 or quality_threshold > 1.0:
            raise HTTPException(
                status_code=400,
                detail="quality_threshold must be between 0.0 and 1.0"
            )
        params["quality_threshold"] = quality_threshold
    
    if reasoning_depth is not None:
        if reasoning_depth < 1 or reasoning_depth > 10:
            raise HTTPException(
                status_code=400,
                detail="reasoning_depth must be between 1 and 10"
            )
        params["reasoning_depth"] = reasoning_depth
    
    return params
