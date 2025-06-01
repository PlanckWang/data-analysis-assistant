"""LLM Provider implementations for ChatGPT, Claude, and DeepSeek"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Send chat messages and get response"""
        pass
    
    @abstractmethod
    def format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Format messages for the specific provider"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI (ChatGPT) provider"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.base_url = "https://api.openai.com/v1"
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    async def chat(
        self,
        messages: List[Message],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": self.format_messages(messages),
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        async with httpx.AsyncClient() as client:
            if stream:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60.0
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data_str)
                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
            else:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and result["choices"]:
                    yield result["choices"][0]["message"]["content"]


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) provider"""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229")
        self.base_url = "https://api.anthropic.com/v1"
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    def format_messages(self, messages: List[Message]) -> str:
        # Claude uses a different format - combine into a single prompt
        formatted = []
        for msg in messages:
            if msg.role == "user":
                formatted.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                formatted.append(f"Assistant: {msg.content}")
            elif msg.role == "system":
                # Add system message at the beginning
                formatted.insert(0, msg.content)
        
        # Ensure the conversation ends with Human: for Claude to respond
        if not formatted[-1].startswith("Human:"):
            formatted.append("Human: Please respond to the above.")
        
        formatted.append("Assistant:")
        
        return "\n\n".join(formatted)
    
    async def chat(
        self,
        messages: List[Message],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        # Extract system message if present
        system_message = None
        chat_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                chat_messages.append(msg)
        
        # Format messages for Claude API v2
        formatted_messages = []
        for msg in chat_messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        data = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "stream": stream
        }
        
        if system_message:
            data["system"] = system_message
        
        async with httpx.AsyncClient() as client:
            if stream:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=data,
                    timeout=60.0
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            
                            try:
                                chunk = json.loads(data_str)
                                if chunk.get("type") == "content_block_delta":
                                    if "delta" in chunk and "text" in chunk["delta"]:
                                        yield chunk["delta"]["text"]
                            except json.JSONDecodeError:
                                continue
            else:
                response = await client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=data,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                
                if "content" in result and result["content"]:
                    yield result["content"][0]["text"]


class DeepSeekProvider(LLMProvider):
    """DeepSeek provider"""
    
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    async def chat(
        self,
        messages: List[Message],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": self.format_messages(messages),
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        async with httpx.AsyncClient() as client:
            if stream:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60.0
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data_str)
                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
            else:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and result["choices"]:
                    yield result["choices"][0]["message"]["content"]


class LLMProviderFactory:
    """Factory to create LLM providers"""
    
    @staticmethod
    def create(provider_name: str) -> LLMProvider:
        """Create an LLM provider instance"""
        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "deepseek": DeepSeekProvider
        }
        
        provider_class = providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        return provider_class()
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of providers that have API keys configured"""
        available = []
        
        if os.getenv("OPENAI_API_KEY"):
            available.append("openai")
        if os.getenv("ANTHROPIC_API_KEY"):
            available.append("anthropic")
        if os.getenv("DEEPSEEK_API_KEY"):
            available.append("deepseek")
        
        return available