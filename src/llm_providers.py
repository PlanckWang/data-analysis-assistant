"""
LLM Provider implementations for various services like OpenAI, Anthropic, and DeepSeek.

This module defines an abstract base class `LLMProvider` and concrete implementations
for different LLM APIs. It also includes a factory class `LLMProviderFactory`
to instantiate providers.
"""

import os
import json
import structlog
from typing import Dict, Any, List, Optional, AsyncGenerator, Union # Added Union for format_messages
from abc import ABC, abstractmethod
from dataclasses import dataclass, field # Added field for dataclass docstrings

import httpx
from dotenv import load_dotenv

from . import exceptions # Added import

# Load environment variables
load_dotenv()

logger = structlog.get_logger(__name__) # Changed


@dataclass
class Message:
    """
    Represents a single message in a chat conversation.
    """
    role: str = field(metadata={"description": "The role of the message sender (e.g., 'user', 'assistant', 'system')."})
    content: str = field(metadata={"description": "The content of the message."})


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Sends chat messages to the LLM and yields response chunks.

        Args:
            messages: A list of Message objects representing the conversation history.
            stream: Whether to stream the response or return it as a single string.
            temperature: The sampling temperature for generation.
            max_tokens: The maximum number of tokens to generate.

        Yields:
            str: Chunks of the LLM's response if streaming, or the full response if not.
        """
        pass # pragma: no cover
    
    @abstractmethod
    def format_messages(self, messages: List[Message]) -> Union[List[Dict[str, str]], str]:
        """
        Formats messages for the specific LLM provider's API.

        Args:
            messages: A list of Message objects.

        Returns:
            The formatted messages, type depends on the provider (e.g., list of dicts or a single string).
        """
        pass # pragma: no cover


class OpenAIProvider(LLMProvider):
    """LLM Provider for OpenAI (ChatGPT) API."""
    
    def __init__(self) -> None:
        """
        Initializes the OpenAIProvider.

        Raises:
            exceptions.LLMProviderError: If the OpenAI API key is not found in environment variables.
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.base_url = "https://api.openai.com/v1"
        
        if not self.api_key:
            raise exceptions.LLMProviderError(self.__class__.__name__, "OPENAI_API_KEY not found in environment variables")
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """
        Formats messages for the OpenAI API.

        Args:
            messages: A list of Message objects.

        Returns:
            A list of dictionaries, each with "role" and "content" keys.
        """
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    async def chat(
        self,
        messages: List[Message],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Sends chat messages to the OpenAI API and yields response chunks.

        Args:
            messages: A list of Message objects representing the conversation history.
            stream: Whether to stream the response.
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate.

        Yields:
            str: Chunks of the LLM's response if streaming, or the full response.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data: Dict[str, Any] = {
            "model": self.model,
            "messages": self.format_messages(messages),
            "temperature": temperature,
            "stream": stream
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
    """LLM Provider for Anthropic (Claude) API."""
    
    def __init__(self) -> None:
        """
        Initializes the AnthropicProvider.

        Raises:
            exceptions.LLMProviderError: If the Anthropic API key is not found in environment variables.
        """
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229")
        self.base_url = "https://api.anthropic.com/v1"
        
        if not self.api_key:
            raise exceptions.LLMProviderError(self.__class__.__name__, "ANTHROPIC_API_KEY not found in environment variables")
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, str]]: # Corrected based on usage in chat()
        """
        Formats messages for the Anthropic Claude API (v2 messages API).

        Args:
            messages: A list of Message objects. System messages are handled separately.

        Returns:
            A list of dictionaries for user/assistant messages.
        """
        # System message is handled separately in the API call for v2
        formatted_messages: List[Dict[str, str]] = []
        for msg in messages:
            if msg.role != "system": # System messages are passed in a dedicated 'system' parameter
                formatted_messages.append({"role": msg.role, "content": msg.content})
        return formatted_messages

    async def chat(
        self,
        messages: List[Message],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Sends chat messages to the Anthropic API and yields response chunks.

        Args:
            messages: A list of Message objects representing the conversation history.
            stream: Whether to stream the response.
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate.

        Yields:
            str: Chunks of the LLM's response if streaming, or the full response.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        headers: Dict[str, str] = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01", # Recommended version
            "Content-Type": "application/json"
        }

        # Extract system message if present
        system_message: Optional[str] = None
        chat_messages: List[Message] = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                chat_messages.append(msg)

        # Format messages for Claude API v2
        formatted_api_messages: List[Dict[str, str]] = self.format_messages(chat_messages) # Use the corrected format_messages

        data: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_api_messages, # Use the list of dicts
            "temperature": temperature,
            "max_tokens": max_tokens or 4096, # Anthropic requires max_tokens
            "stream": stream
        }

        if system_message:
            data["system"] = system_message

        async with httpx.AsyncClient() as client:
            if stream:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages", # Correct endpoint for Claude v2
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
                                # Log this or handle as needed
                                logger.warn("json_decode_error_in_stream", data_str=data_str)
                                continue
            else:
                response = await client.post(
                    f"{self.base_url}/messages", # Correct endpoint for Claude v2
                    headers=headers,
                    json=data,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()

                if "content" in result and result["content"]:
                    # Assuming the primary content is in the first block if multiple exist
                    yield result["content"][0]["text"]


class DeepSeekProvider(LLMProvider):
class DeepSeekProvider(LLMProvider):
    """DeepSeek provider"""
    
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        
        if not self.api_key:
            raise exceptions.LLMProviderError(self.__class__.__name__, "DEEPSEEK_API_KEY not found in environment variables")
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """
        Formats messages for the DeepSeek API.

        Args:
            messages: A list of Message objects.

        Returns:
            A list of dictionaries, each with "role" and "content" keys.
        """
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    async def chat(
        self,
        messages: List[Message],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Sends chat messages to the DeepSeek API and yields response chunks.

        Args:
            messages: A list of Message objects representing the conversation history.
            stream: Whether to stream the response.
            temperature: The sampling temperature.
            max_tokens: The maximum number of tokens to generate.

        Yields:
            str: Chunks of the LLM's response if streaming, or the full response.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Corrected data initialization - remove the duplicate
        data: Dict[str, Any] = {
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
                response_non_stream = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60.0
                )
                response_non_stream.raise_for_status()
                result = response_non_stream.json()
                
                if "choices" in result and result["choices"]:
                    yield result["choices"][0]["message"]["content"]


class LLMProviderFactory:
    """Factory to create LLM provider instances."""
    
    @staticmethod
    def create(provider_name: str) -> LLMProvider:
        """
        Creates an LLM provider instance based on the provider name.

        Args:
            provider_name: The name of the LLM provider (e.g., 'openai', 'anthropic').

        Returns:
            An instance of the requested LLMProvider.

        Raises:
            exceptions.LLMProviderError: If the provider_name is unknown or unsupported.
        """
        providers: Dict[str, type[LLMProvider]] = { # Added type hint for providers dict
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "deepseek": DeepSeekProvider
        }
        
        provider_class = providers.get(provider_name.lower())
        if not provider_class:
            raise exceptions.LLMProviderError(provider_name, "Provider not recognized or supported.")
        
        return provider_class() # type: ignore # Pylance might complain, but this is fine
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """
        Gets a list of LLM providers that have their API keys configured in environment variables.

        Returns:
            A list of strings, where each string is the name of an available provider.
        """
        available: List[str] = []
        
        if os.getenv("OPENAI_API_KEY"):
            available.append("openai")
        if os.getenv("ANTHROPIC_API_KEY"):
            available.append("anthropic")
        if os.getenv("DEEPSEEK_API_KEY"):
            available.append("deepseek")
        
        return available