#!/usr/bin/env python3
"""
Universal LLM Client supporting both OpenAI and Ollama APIs
Allows switching between providers for comparison and testing
Based on the ollama_chat.py example with additional OpenAI support
"""

import os
import sys
import json
import requests
import time
from typing import List, Dict, Any, Optional, Generator, Union
from dataclasses import dataclass

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OPENAI_API_KEY, 
    LLM_PROVIDER, 
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL
)
from utils.logger import logger

@dataclass
class LLMResponse:
    """Standardized response format for both providers"""
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = None

class UniversalLLMClient:
    """Universal client that can switch between OpenAI and Ollama"""
    
    def __init__(self, provider: str = None):
        self.provider = provider or LLM_PROVIDER
        self.openai_client = None
        
        logger.info(f"Initializing LLM client with provider: {self.provider}")
        
        if self.provider == "openai":
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI client configured")
            except ImportError:
                logger.error("OpenAI package not installed. Run: pip install openai")
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        elif self.provider == "ollama":
            self.ollama_base = OLLAMA_BASE_URL
            self.ollama_model = OLLAMA_MODEL
            logger.info(f"Ollama client configured with model: {self.ollama_model}")
            self._validate_ollama_connection()
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _validate_ollama_connection(self):
        """Validate that Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_base}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = [m["name"] for m in response.json().get("models", [])]
            if self.ollama_model not in models:
                available = ", ".join(models)
                logger.warning(f"Model {self.ollama_model} not found. Available: {available}")
                # Try to use the first available model if the specified one isn't found
                if models:
                    self.ollama_model = models[0]
                    logger.info(f"Using first available model: {self.ollama_model}")
            else:
                logger.info(f"Ollama model {self.ollama_model} is available")
                
            # Test a simple request to ensure Ollama is responsive
            test_response = requests.post(
                f"{self.ollama_base}/api/generate",
                json={"model": self.ollama_model, "prompt": "test", "stream": False},
                timeout=3
            )
            if test_response.status_code == 200:
                logger.info("Ollama is responsive")
            else:
                logger.warning(f"Ollama test failed: {test_response.status_code}")
                
        except requests.RequestException as e:
            logger.warning(f"Could not connect to Ollama at {self.ollama_base}: {e}")
            logger.warning("Consider switching to OpenAI provider if Ollama issues persist")
    
    def _supports_chat_api(self) -> bool:
        """Check if Ollama supports the /api/chat endpoint"""
        try:
            response = requests.get(f"{self.ollama_base}/api/chat", timeout=10)
            return response.status_code in (400, 405)  # 405 is typical for GET on POST endpoint
        except requests.RequestException:
            return False
    
    def _http_request(self, method: str, url: str, **kwargs):
        """Helper function for HTTP requests with proper error handling"""
        timeout = kwargs.pop("timeout", 300)
        response = requests.request(method, url, timeout=timeout, **kwargs)
        try:
            response.raise_for_status()
        except requests.HTTPError:
            logger.error(f"HTTP {response.status_code} for {url}\nBody: {response.text}")
            raise
        return response
    
    def _ollama_chat(self, messages: List[Dict[str, str]], 
                     temperature: float = 0.7, 
                     max_tokens: int = 2000,
                     stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Send request to Ollama using /api/chat endpoint"""
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": stream
        }
        
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        
        if options:
            payload["options"] = options
        
        if not stream:
            response = self._http_request("POST", f"{self.ollama_base}/api/chat", json=payload)
            data = response.json()
            return data["message"]["content"]
        else:
            with requests.post(f"{self.ollama_base}/api/chat", json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        return
    
    def _ollama_generate(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.7, 
                        max_tokens: int = 2000,
                        stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """Send request to Ollama using /api/generate endpoint (fallback)"""
        # Convert messages to prompt format using the same logic as the example
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.upper()}: {content}")
        prompt_parts.append("ASSISTANT:")
        prompt = "\n".join(prompt_parts)
        
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": stream
        }
        
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        
        if options:
            payload["options"] = options
        
        if not stream:
            response = self._http_request("POST", f"{self.ollama_base}/api/generate", json=payload)
            data = response.json()
            return data.get("response", "")
        else:
            with requests.post(f"{self.ollama_base}/api/generate", json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("response", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        return
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = None,
                       temperature: float = 0.7, 
                       max_tokens: int = 2000,
                       stream: bool = False) -> Union[LLMResponse, str]:
        """
        Universal chat completion method that works with both providers
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model name (optional, uses default from config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
        
        Returns:
            LLMResponse object with standardized format (or just content string for compatibility)
        """
        start_time = time.time()
        
        if self.provider == "openai":
            try:
                response = self.openai_client.chat.completions.create(
                    model=model or "gpt-3.5-turbo",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
                
                if stream:
                    # Handle streaming response
                    content_parts = []
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content_parts.append(chunk.choices[0].delta.content)
                    content = "".join(content_parts)
                else:
                    content = response.choices[0].message.content
                    
                return LLMResponse(
                    content=content,
                    provider="openai",
                    model=model or "gpt-3.5-turbo",
                    usage=response.usage.__dict__ if hasattr(response, 'usage') and response.usage else None,
                    response_time=time.time() - start_time
                )
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                raise
        
        elif self.provider == "ollama":
            try:
                # Try chat API first, fallback to generate
                if self._supports_chat_api():
                    content = self._ollama_chat(messages, temperature, max_tokens, stream)
                else:
                    content = self._ollama_generate(messages, temperature, max_tokens, stream)
                
                if stream:
                    # For streaming, we need to collect all pieces
                    if hasattr(content, '__iter__'):
                        content_parts = list(content)
                        content = "".join(content_parts)
                
                # Ensure content is a string
                if not isinstance(content, str):
                    content = str(content)
                
                return LLMResponse(
                    content=content,
                    provider="ollama",
                    model=self.ollama_model,
                    response_time=time.time() - start_time
                )
                
            except requests.Timeout:
                logger.error("Ollama request timed out. The model might be busy or unresponsive.")
                raise Exception("Ollama request timed out")
            except requests.RequestException as e:
                logger.error(f"Ollama connection error: {e}")
                raise Exception(f"Ollama connection error: {e}")
            except Exception as e:
                logger.error(f"Ollama API error: {e}")
                raise Exception(f"Ollama API error: {e}")
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def simple_completion(self, prompt: str, **kwargs) -> str:
        """Simple completion for backward compatibility"""
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(messages, **kwargs)
        if isinstance(response, LLMResponse):
            return response.content
        return str(response)
    
    def get_embedding(self, text: str, model: str = None) -> List[float]:
        """Get text embeddings (OpenAI only for now)"""
        if self.provider == "openai":
            response = self.openai_client.embeddings.create(
                model=model or "text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        else:
            logger.warning("Embeddings not yet implemented for Ollama")
            raise NotImplementedError("Embeddings not yet implemented for Ollama")
    
    def switch_provider(self, provider: str):
        """Switch between providers at runtime"""
        old_provider = self.provider
        self.provider = provider
        
        if provider == "openai" and not self.openai_client:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("Switched to OpenAI provider")
            except ImportError:
                self.provider = old_provider
                logger.error("OpenAI package not installed")
                raise ImportError("OpenAI package not installed")
        
        elif provider == "ollama":
            self._validate_ollama_connection()
            logger.info("Switched to Ollama provider")
        
        logger.info(f"Switched from {old_provider} to {provider}")
    
    def test_connection(self):
        """Test the connection to the LLM provider"""
        try:
            test_messages = [
                {"role": "user", "content": "Hello, please respond with 'Connection successful'"}
            ]
            
            response = self.chat_completion(test_messages)
            content = response.content if isinstance(response, LLMResponse) else response
            logger.info(f"LLM connection test successful: {content}")
            return True
            
        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")
            return False

# Create a global instance for backward compatibility
llm_client = UniversalLLMClient()

# Factory function for easy use
def get_llm_client(provider: str = None) -> UniversalLLMClient:
    """Factory function to get LLM client instance"""
    return UniversalLLMClient(provider)

# Backward compatibility functions
def get_llm_response(messages: List[Dict[str, str]], model: str = None, temperature: float = 0.7, max_tokens: int = None) -> str:
    """
    Get response from the configured LLM provider (backward compatibility)
    
    Args:
        messages: List of message dictionaries
        model: Model name (optional)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Returns:
        String response from the LLM
    """
    response = llm_client.chat_completion(messages, model, temperature, max_tokens or 2000)
    if isinstance(response, LLMResponse):
        return response.content
    return response

def llm_completion(prompt: str, provider: str = None, **kwargs) -> str:
    """Simple completion function for backward compatibility"""
    client = get_llm_client(provider)
    return client.simple_completion(prompt, **kwargs)

def llm_chat(messages: List[Dict[str, str]], provider: str = None, **kwargs) -> str:
    """Chat completion function for backward compatibility"""
    client = get_llm_client(provider)
    response = client.chat_completion(messages, **kwargs)
    if isinstance(response, LLMResponse):
        return response.content
    return response

# Test function
def test_both_providers():
    """Test both providers with the same prompt"""
    test_prompt = "What is the capital of France? Answer in one sentence."
    
    print("Testing OpenAI...")
    try:
        openai_client = get_llm_client("openai")
        openai_response = openai_client.simple_completion(test_prompt)
        print(f"OpenAI Response: {openai_response}")
    except Exception as e:
        print(f"OpenAI Error: {e}")
    
    print("\nTesting Ollama...")
    try:
        ollama_client = get_llm_client("ollama")
        ollama_response = ollama_client.simple_completion(test_prompt)
        print(f"Ollama Response: {ollama_response}")
    except Exception as e:
        print(f"Ollama Error: {e}")

if __name__ == "__main__":
    test_both_providers()
