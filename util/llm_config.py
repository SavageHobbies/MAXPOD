"""Configuration for LLM providers and models."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for an LLM provider and model."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    

# Default configurations for different providers
OLLAMA_CONFIGS = {
    "nemotron-mini": LLMConfig(
        provider="ollama",
        model="nemotron-mini:4b-instruct-q4_K_M",
        base_url="http://localhost:11434"
    ),
    "llama2": LLMConfig(
        provider="ollama",
        model="llama2",
        base_url="http://localhost:11434"
    ),
    # Add more Ollama models as needed
}

OPENAI_CONFIGS = {
    "gpt-4": LLMConfig(
        provider="openai",
        model="gpt-4"
    ),
    "gpt-3.5-turbo": LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo"
    ),
}

ANTHROPIC_CONFIGS = {
    "claude-3-opus": LLMConfig(
        provider="anthropic",
        model="claude-3-opus"
    ),
    "claude-3-sonnet": LLMConfig(
        provider="anthropic",
        model="claude-3-sonnet"
    ),
}

GEMINI_CONFIGS = {
    "gemini-pro": LLMConfig(
        provider="gemini",
        model="gemini-pro"
    ),
}

HUGGINGFACE_CONFIGS = {
    "mistral": LLMConfig(
        provider="huggingface",
        model="mistralai/Mistral-7B-v0.1"
    ),
}

# Combine all configurations
ALL_CONFIGS = {
    **OLLAMA_CONFIGS,
    **OPENAI_CONFIGS,
    **ANTHROPIC_CONFIGS,
    **GEMINI_CONFIGS,
    **HUGGINGFACE_CONFIGS,
}