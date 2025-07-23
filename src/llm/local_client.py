"""
Local LLM Client for NeuroDoc
Provides integration with local language models via Ollama
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class LocalLLMResponse:
    """Response from local LLM model."""
    content: str
    model: str
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaClient:
    """
    Client for interacting with Ollama local LLM server.
    Supports Llama 2, Mistral, CodeLlama, and other models.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self._available_models: Optional[List[str]] = None
        self._model_info: Dict[str, Dict] = {}
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available."""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def check_health(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    self._available_models = models
                    return models
                else:
                    logger.error(f"Failed to list models: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            await self._ensure_session()
            
            payload = {"name": model_name}
            
            logger.info(f"Pulling model {model_name}...")
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json=payload
            ) as response:
                if response.status == 200:
                    # Stream the pull progress
                    async for line in response.content:
                        try:
                            progress = json.loads(line)
                            if "status" in progress:
                                logger.info(f"Pull progress: {progress['status']}")
                        except json.JSONDecodeError:
                            continue
                    
                    logger.info(f"Successfully pulled model {model_name}")
                    return True
                else:
                    logger.error(f"Failed to pull model {model_name}: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        stream: bool = False
    ) -> LocalLLMResponse:
        """Generate text using a local model."""
        try:
            await self._ensure_session()
            
            # Prepare the request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            }
            
            if system:
                payload["system"] = system
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            start_time = time.time()
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    if stream:
                        return await self._handle_streaming_response(response, model)
                    else:
                        data = await response.json()
                        return LocalLLMResponse(
                            content=data.get("response", ""),
                            model=model,
                            total_duration=data.get("total_duration"),
                            load_duration=data.get("load_duration"),
                            prompt_eval_count=data.get("prompt_eval_count"),
                            eval_count=data.get("eval_count"),
                            eval_duration=data.get("eval_duration")
                        )
                else:
                    error_text = await response.text()
                    logger.error(f"Generation failed: HTTP {response.status} - {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error generating with model {model}: {e}")
            raise
    
    async def _handle_streaming_response(
        self, 
        response: aiohttp.ClientResponse, 
        model: str
    ) -> LocalLLMResponse:
        """Handle streaming response from Ollama."""
        content_parts = []
        final_data = {}
        
        async for line in response.content:
            try:
                data = json.loads(line)
                if "response" in data:
                    content_parts.append(data["response"])
                
                # Capture final statistics
                if data.get("done", False):
                    final_data = data
                    
            except json.JSONDecodeError:
                continue
        
        return LocalLLMResponse(
            content="".join(content_parts),
            model=model,
            total_duration=final_data.get("total_duration"),
            load_duration=final_data.get("load_duration"),
            prompt_eval_count=final_data.get("prompt_eval_count"),
            eval_count=final_data.get("eval_count"),
            eval_duration=final_data.get("eval_duration")
        )
    
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9
    ) -> LocalLLMResponse:
        """Chat interface for conversation-style interactions."""
        try:
            await self._ensure_session()
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    message = data.get("message", {})
                    return LocalLLMResponse(
                        content=message.get("content", ""),
                        model=model,
                        total_duration=data.get("total_duration"),
                        load_duration=data.get("load_duration"),
                        prompt_eval_count=data.get("prompt_eval_count"),
                        eval_count=data.get("eval_count"),
                        eval_duration=data.get("eval_duration")
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Chat failed: HTTP {response.status} - {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error in chat with model {model}: {e}")
            raise


class LocalLLMManager:
    """
    Manager for local LLM operations in NeuroDoc.
    Handles model selection, fallbacks, and optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama = OllamaClient(config.get("ollama_base_url", "http://localhost:11434"))
        self.primary_model = config.get("model_name", "llama2:7b-chat")
        self.fallback_model = config.get("fallback_model", "mistral:7b-instruct")
        self.available_models: List[str] = []
        self._initialized = False
    
    async def initialize(self):
        """Initialize the local LLM manager."""
        try:
            # Check Ollama health
            is_healthy = await self.ollama.check_health()
            if not is_healthy:
                raise Exception("Ollama server is not accessible")
            
            # Get available models
            self.available_models = await self.ollama.list_models()
            logger.info(f"Available models: {self.available_models}")
            
            # Ensure primary model is available
            if self.primary_model not in self.available_models:
                logger.info(f"Primary model {self.primary_model} not found, attempting to pull...")
                success = await self.ollama.pull_model(self.primary_model)
                if success:
                    self.available_models.append(self.primary_model)
                else:
                    logger.warning(f"Failed to pull primary model {self.primary_model}")
            
            self._initialized = True
            logger.info("Local LLM manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize local LLM manager: {e}")
            raise
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a response using local LLM."""
        if not self._initialized:
            await self.initialize()
        
        # Select model
        selected_model = model or self.primary_model
        if selected_model not in self.available_models:
            logger.warning(f"Model {selected_model} not available, using fallback")
            selected_model = self.fallback_model
        
        try:
            response = await self.ollama.generate(
                model=selected_model,
                prompt=prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.info(f"Generated response with {selected_model} in {response.total_duration or 0}ns")
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response with {selected_model}: {e}")
            
            # Try fallback model if primary failed
            if selected_model != self.fallback_model and self.fallback_model in self.available_models:
                logger.info(f"Retrying with fallback model {self.fallback_model}")
                try:
                    response = await self.ollama.generate(
                        model=self.fallback_model,
                        prompt=prompt,
                        system=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.content
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
            
            raise Exception(f"All local LLM models failed: {e}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            "available_models": self.available_models,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "ollama_healthy": await self.ollama.check_health()
        }
