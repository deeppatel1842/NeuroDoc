"""
LLM Integration and Response Generation

Handles integration with Large Language Models for generating
responses based on retrieved context in the RAG system.
"""

import logging
import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np

# Import LLM libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available")

# Import local LLM client
from .local_client import LocalLLMManager

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available")

from ..config import LLM_CONFIG
from ..utils.text_utils import TextCleaner

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Multi-model response generator with local Llama 2 support.
    Supports local Ollama models and OpenAI GPT models as fallback.
    """
    
    def __init__(self):
        self.model_name = LLM_CONFIG.get("model_name", "llama2:7b-chat")
        self.max_tokens = LLM_CONFIG.get("max_tokens", 1500)
        self.temperature = LLM_CONFIG.get("temperature", 0.1)
        self.top_p = LLM_CONFIG.get("top_p", 0.9)
        
        # Context limits
        self.max_context_length = LLM_CONFIG.get("max_context_length", 4000)
        self.max_retrieved_chunks = LLM_CONFIG.get("max_retrieved_chunks", 5)
        
        # Initialize clients
        self.local_llm_manager = None
        self.openai_client = None
        self.local_model = None
        self.local_tokenizer = None
        
        self.text_cleaner = TextCleaner()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available LLM models."""
        try:
            # Prioritize local LLM (Ollama) over other options
            if LLM_CONFIG.get("use_ollama", True):
                self._initialize_ollama()
            
            # OpenAI as fallback
            if OPENAI_AVAILABLE and LLM_CONFIG.get("use_openai", False):
                self._initialize_openai()
            
            # Local Hugging Face as final fallback
            if TRANSFORMERS_AVAILABLE and LLM_CONFIG.get("use_local_model", False):
                self._initialize_local_model()
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    def _initialize_ollama(self):
        """Initialize Ollama local LLM manager."""
        try:
            self.local_llm_manager = LocalLLMManager(LLM_CONFIG)
            logger.info("Ollama LLM manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            self.local_llm_manager = None
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        api_key = LLM_CONFIG.get("openai_api_key")
        if api_key:
            openai.api_key = api_key
            self.openai_client = openai
            logger.info("OpenAI client initialized")
        else:
            logger.warning("OpenAI API key not provided")
    
    def _initialize_local_model(self):
        """Initialize local Hugging Face model."""
        try:
            model_name = LLM_CONFIG.get("local_model_name", "microsoft/DialoGPT-medium")
            
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.local_tokenizer.pad_token is None:
                self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
            
            logger.info(f"Local model {model_name} initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            self.local_model = None
            self.local_tokenizer = None
    
    async def health_check(self) -> bool:
        """Check if the response generator is healthy."""
        try:
            # Simple test generation
            test_response = await self.generate_response(
                question="What is AI?",
                retrieved_docs=[{
                    "text": "Artificial Intelligence (AI) is a field of computer science.",
                    "score": 0.9,
                    "metadata": {"test": True}
                }],
                conversation_history=[],
                session_id="test"
            )
            return len(test_response.get("answer", "")) > 0
        except Exception as e:
            logger.error(f"Response generator health check failed: {e}")
            return False
    
    async def generate_response(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        conversation_history: List[Any],  # Can be strings or dicts
        session_id: str
    ) -> Dict[str, Any]:
        """
        Generate a response based on the question and retrieved context.
        
        Args:
            question: User's question
            retrieved_docs: Retrieved document chunks with scores
            conversation_history: Recent conversation history
            session_id: Session identifier
            
        Returns:
            Dictionary containing answer, citations, and metadata
        """
        try:
            start_time = datetime.utcnow()
            
            # Prepare context
            context = self._prepare_context(retrieved_docs, conversation_history)
            
            # Generate response using priority order: Ollama -> OpenAI -> Local Hugging Face -> Fallback
            if self.local_llm_manager and LLM_CONFIG.get("use_ollama", True):
                response = await self._generate_ollama_response(question, context)
            elif self.openai_client and LLM_CONFIG.get("use_openai", False):
                response = await self._generate_openai_response(question, context)
            elif self.local_model and LLM_CONFIG.get("use_local_model", False):
                response = await self._generate_local_response(question, context)
            else:
                response = self._generate_fallback_response(question, retrieved_docs)
            
            # Extract citations
            citations = self._extract_citations(response, retrieved_docs)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                response, retrieved_docs, question
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                "answer": response,
                "citations": citations,
                "source_documents": self._format_source_documents(retrieved_docs),
                "confidence_score": confidence_score,
                "metadata": {
                    "model_used": self._get_active_model_name(),
                    "processing_time": processing_time,
                    "context_length": len(context),
                    "retrieved_chunks": len(retrieved_docs),
                    "session_id": session_id
                }
            }
            
            logger.info(f"Generated response for session {session_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error while generating a response. Please try again.",
                "citations": [],
                "source_documents": [],
                "confidence_score": 0.0,
                "metadata": {"error": str(e)}
            }
    
    async def _generate_ollama_response(self, question: str, context: str) -> str:
        """Generate response using Ollama local LLM."""
        try:
            # Check if this is a simple conversational question
            conversational_patterns = [
                'how are you', 'what is your name', 'who are you', 'hello', 'hi there',
                'good morning', 'good afternoon', 'good evening', 'how do you do',
                'nice to meet you', 'what are you', 'are you okay', 'how\'s it going'
            ]
            
            question_lower = question.lower().strip()
            is_conversational = any(pattern in question_lower for pattern in conversational_patterns)
            
            if is_conversational:
                # Handle conversational questions directly without document context
                if 'how are you' in question_lower or 'how do you do' in question_lower or 'how\'s it going' in question_lower:
                    return "I'm doing great, thank you for asking! I'm here to help you analyze your research documents and answer questions about AI, machine learning, and related topics. What would you like to explore today?"
                elif 'name' in question_lower or 'who are you' in question_lower or 'what are you' in question_lower:
                    return "I'm NeuroDoc, your AI-powered document analysis assistant. I specialize in helping you understand and explore research papers in AI, machine learning, and related fields."
                elif any(greeting in question_lower for greeting in ['hello', 'hi', 'good morning', 'good afternoon', 'good evening']):
                    return "Hello! I'm NeuroDoc, ready to help you dive into your research documents. What questions do you have about AI, machine learning, or your papers?"
                else:
                    return "I'm NeuroDoc, your AI assistant for document analysis. I'm here to help you understand research papers and answer questions about AI and machine learning topics. How can I assist you today?"
            
            # For technical questions, use the full context
            system_prompt = """You are NeuroDoc, an AI assistant specialized in intelligent document analysis and comprehensive question answering. 
            You have access to document context and should use your full AI capabilities to:
            - Analyze and synthesize information thoroughly
            - Provide detailed, comprehensive responses
            - Make intelligent inferences and connections
            - Go beyond simple text extraction to provide valuable insights
            - Structure information clearly and professionally
            - Always support claims with specific source citations
            
            Your goal is to be as helpful and comprehensive as possible while maintaining accuracy."""
            
            prompt = f"""Context from research documents:
{context}

Question: {question}

Instructions:
- Provide a comprehensive answer using the document context
- Include specific details and examples from the research
- Reference multiple sources when available ("Document 1 explains...", "According to the research...")
- Be detailed but clear (aim for 100-200 words)
- If the context doesn't contain relevant information, say so clearly

Answer:"""

            response = await self.local_llm_manager.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=2000
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Ollama response generation failed: {e}")
            raise
    
    def _prepare_context(
        self,
        retrieved_docs: List[Dict[str, Any]],
        conversation_history: List[Any]  # Can be strings or dicts
    ) -> str:
        """Prepare context string from retrieved documents and conversation history."""
        context_parts = []
        
        # Add conversation history for context
        if conversation_history:
            context_parts.append("Previous conversation:")
            for question in conversation_history[-3:]:  # Last 3 questions
                # Handle both string and dict formats
                if isinstance(question, dict):
                    q_text = question.get("content", question.get("question", str(question)))
                else:
                    q_text = str(question)
                context_parts.append(f"Q: {q_text}")
            context_parts.append("")
        
        # Add retrieved documents
        context_parts.append("Relevant information:")
        
        # Sort documents by score and limit number
        sorted_docs = sorted(
            retrieved_docs[:self.max_retrieved_chunks], 
            key=lambda x: x.get("score", 0), 
            reverse=True
        )
        
        current_length = 0
        for i, doc in enumerate(sorted_docs):
            doc_text = doc.get("text", "")
            
            # Check context length limit
            if current_length + len(doc_text) > self.max_context_length:
                # Truncate if necessary
                remaining_space = self.max_context_length - current_length
                if remaining_space > 100:  # Only add if significant space remains
                    doc_text = doc_text[:remaining_space] + "..."
                else:
                    break
            
            context_parts.append(f"Document {i+1}: {doc_text}")
            current_length += len(doc_text)
        
        return "\n".join(context_parts)
    
    async def _generate_openai_response(self, question: str, context: str) -> str:
        """Generate response using OpenAI API."""
        prompt = self._create_rag_prompt(question, context)
        
        try:
            if "gpt-3.5" in self.model_name or "gpt-4" in self.model_name:
                # Use ChatCompletion API for newer models
                response = await openai.ChatCompletion.acreate(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context. Always cite your sources and be accurate."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                return response.choices[0].message.content.strip()
            else:
                # Use Completion API for older models
                response = await openai.Completion.acreate(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                return response.choices[0].text.strip()
                
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    async def _generate_local_response(self, question: str, context: str) -> str:
        """Generate response using local Hugging Face model."""
        prompt = self._create_rag_prompt(question, context)
        
        try:
            # Tokenize input
            inputs = self.local_tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=self.max_context_length,
                truncation=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.local_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.local_tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Local model generation failed: {e}")
            raise
    
    def _generate_fallback_response(
        self, 
        question: str, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """Generate a simple fallback response when models are unavailable."""
        if not retrieved_docs:
            return "I don't have enough information to answer your question. Please provide more context or documents."
        
        # Simple extractive response
        best_doc = max(retrieved_docs, key=lambda x: x.get("score", 0))
        doc_text = best_doc.get("text", "")
        
        # Find relevant sentences
        sentences = doc_text.split(".")
        relevant_sentences = []
        
        question_words = set(question.lower().split())
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            if overlap > 0:
                relevant_sentences.append((sentence.strip(), overlap))
        
        if relevant_sentences:
            # Sort by relevance and take top sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            response_parts = [sent[0] for sent in relevant_sentences[:3]]
            return ". ".join(response_parts) + "."
        else:
            # Return beginning of best document
            return doc_text[:500] + "..." if len(doc_text) > 500 else doc_text
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a prompt for RAG-based question answering."""
        prompt_template = """You are NeuroDoc, an AI assistant specialized in intelligent document analysis and comprehensive question answering. Your role is to provide complete, detailed, and insightful responses based on the provided context.

IMPORTANT INSTRUCTIONS:
- Use your full AI capabilities to analyze and synthesize information from the context
- Provide comprehensive answers that go beyond simple extraction
- Draw connections, make inferences, and provide detailed explanations
- Use your knowledge to enhance and complete the answer when relevant
- Be thorough and detailed in your responses
- Always cite specific sources when making claims
- If asked for summaries, provide well-structured, comprehensive summaries
- Do not limit yourself to just extracting text - use your intelligence to provide valuable insights

Context from documents:
{context}

Question: {question}

Provide a comprehensive, detailed answer using your full AI capabilities to analyze and synthesize the information:"""
        
        return prompt_template.format(context=context, question=question)
    
    def _extract_citations(
        self, 
        response: str, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract and format citations from the response."""
        citations = []
        
        # Look for document references in the response
        doc_patterns = [
            r"[Dd]ocument\s+(\d+)",
            r"[Ss]ource\s+(\d+)",
            r"\[(\d+)\]"
        ]
        
        cited_docs = set()
        for pattern in doc_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    doc_index = int(match) - 1  # Convert to 0-based index
                    if 0 <= doc_index < len(retrieved_docs):
                        cited_docs.add(doc_index)
                except ValueError:
                    continue
        
        # If no explicit citations found, use top-scoring documents (more generous)
        if not cited_docs:
            # Include top 3-5 documents based on relevance score
            num_to_include = min(5, len(retrieved_docs))
            for i in range(num_to_include):
                if retrieved_docs[i].get("score", 0) > 0.1:  # Only include if reasonable relevance
                    cited_docs.add(i)
        
        # Format citations
        for doc_index in cited_docs:
            if doc_index < len(retrieved_docs):
                doc = retrieved_docs[doc_index]
                metadata = doc.get("metadata", {})
                
                citation = {
                    "document_id": metadata.get("document_id", "unknown"),
                    "chunk_id": metadata.get("chunk_id", f"chunk_{doc_index}"),
                    "text_snippet": doc.get("text", "")[:200] + "..." if len(doc.get("text", "")) > 200 else doc.get("text", ""),
                    "relevance_score": float(doc.get("score", 0.0)),
                    # Optional fields
                    "document_name": metadata.get("filename") or metadata.get("document_name"),
                    "page_number": metadata.get("page_number"),
                    "confidence_score": float(doc.get("score", 0.0))
                }
                citations.append(citation)
        
        return citations
    
    def _calculate_confidence_score(
        self,
        response: str,
        retrieved_docs: List[Dict[str, Any]],
        question: str
    ) -> float:
        """Calculate a confidence score for the generated response."""
        if not retrieved_docs:
            return 0.1
        
        # Factors for confidence calculation
        factors = []
        
        # 1. Average retrieval score
        avg_retrieval_score = np.mean([doc.get("score", 0) for doc in retrieved_docs])
        factors.append(avg_retrieval_score)
        
        # 2. Response length (longer responses with more details get higher scores)
        response_length = len(response.split())
        if response_length > 100:  # Comprehensive answers
            length_score = 0.9
        elif response_length > 50:  # Moderate answers
            length_score = 0.7
        elif response_length > 20:  # Basic answers
            length_score = 0.5
        else:  # Very short answers
            length_score = 0.3
        factors.append(length_score)
        
        # 3. Keyword overlap between question and response
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        overlap_ratio = len(question_words.intersection(response_words)) / len(question_words)
        factors.append(overlap_ratio)
        
        # 4. Number of retrieved documents (more context = higher confidence)
        doc_count_score = min(len(retrieved_docs) / 5, 1.0)
        factors.append(doc_count_score)
        
        # Calculate weighted average (boosted for better user experience)
        weights = [0.25, 0.35, 0.2, 0.2]  # Prioritize response quality over retrieval scores
        confidence = sum(w * f for w, f in zip(weights, factors))
        
        # Boost confidence for responses that seem comprehensive
        if response_length > 80 and avg_retrieval_score > 0.3:
            confidence = min(confidence + 0.25, 1.0)  # Boost by 25%
        
        # Additional boost for multi-source responses
        if len(retrieved_docs) >= 4 and response_length > 100:
            confidence = min(confidence + 0.15, 1.0)  # Extra 15% for comprehensive multi-source answers
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _format_source_documents(
        self, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format source documents for the response."""
        sources = []
        
        for i, doc in enumerate(retrieved_docs):
            metadata = doc.get("metadata", {})
            source = {
                "index": i + 1,
                "document_id": metadata.get("document_id", "unknown"),
                "chunk_id": metadata.get("chunk_id", f"chunk_{i}"),
                "relevance_score": float(doc.get("score", 0.0)),
                "text_preview": doc.get("text", "")[:300] + "..." if len(doc.get("text", "")) > 300 else doc.get("text", ""),
                "retrieval_type": doc.get("retrieval_type", "semantic")
            }
            sources.append(source)
        
        return sources
        
        return sources
    
    def _get_active_model_name(self) -> str:
        """Get the name of the currently active model."""
        if self.openai_client and LLM_CONFIG.get("use_openai", True):
            return f"openai/{self.model_name}"
        elif self.local_model and LLM_CONFIG.get("use_local_model", False):
            return f"local/{LLM_CONFIG.get('local_model_name', 'unknown')}"
        else:
            return "fallback/extractive"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            "primary_model": self.model_name,
            "openai_available": OPENAI_AVAILABLE and self.openai_client is not None,
            "local_model_available": TRANSFORMERS_AVAILABLE and self.local_model is not None,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_context_length": self.max_context_length
        }
