"""This is a utility class for interacting with various AI models through different providers."""
import requests
import subprocess
import json
from os import getenv
from typing import Optional, Type
from enum import Enum
from pydantic import BaseModel


class AIProvider(Enum):
    """Enum for supported AI providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"


class AiUtil:
    """
    AiUtil is a utility class for interacting with various AI models.
    Supports multiple providers including Ollama, OpenAI, Anthropic, Gemini, and Hugging Face.
    
    Attributes:
        provider (AIProvider): The AI provider to use
        model (str): The model to use for generating completions
        temperature (float): The sampling temperature to use
        max_response_len (int): The maximum length of the response in tokens
        frequency_penalty (float): The penalty for repeated tokens
    """

    def __init__(
            self,
            provider: AIProvider = AIProvider.OLLAMA,
            model: str = "nemotron-mini:4b-instruct-q4_K_M",
            temperature: float = 0.7,
            max_response_len: Optional[int] = None,
            frequency_penalty: float = 0,
            api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_response_len = max_response_len
        self.frequency_penalty = frequency_penalty
        self.api_key = api_key
        
        # Provider-specific setup
        if provider == AIProvider.OLLAMA:
            self.base_url = "http://localhost:11434"
            if not self._check_ollama_running():
                print(f"Ollama not running. Starting {model}...")
                self._start_ollama_model()
                
        elif provider == AIProvider.OPENAI:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key or getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("OpenAI package not installed. Run 'pip install openai'")
                
        elif provider == AIProvider.ANTHROPIC:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key or getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                raise ImportError("Anthropic package not installed. Run 'pip install anthropic'")
                
        elif provider == AIProvider.GEMINI:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key or getenv("GOOGLE_API_KEY"))
                self.client = genai
            except ImportError:
                raise ImportError("Google GenerativeAI package not installed. Run 'pip install google-generativeai'")
                
        elif provider == AIProvider.HUGGINGFACE:
            try:
                from huggingface_hub import HfApi
                self.client = HfApi(token=api_key or getenv("HUGGINGFACE_API_KEY"))
            except ImportError:
                raise ImportError("Hugging Face package not installed. Run 'pip install huggingface-hub'")

    def _check_ollama_running(self) -> bool:
        """Check if Ollama is running and the specified model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model.get("name") == self.model for model in models)
            return False
        except requests.exceptions.RequestException:
            return False

    def _start_ollama_model(self):
        """Start Ollama with the specified model"""
        try:
            subprocess.run(["ollama", "run", self.model], shell=True)
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to start Ollama model: {e}")

    def chat(self, messages: list, output_model: Type[BaseModel]) -> str:
        """
        Send messages to the AI model and get a response.
        Handles different providers' APIs appropriately.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            output_model (Type[BaseModel]): Pydantic model for response format
        
        Returns:
            str: The model's response content
        """
        # Add JSON schema instruction to system message
        schema_instruction = f"Always respond with valid JSON that matches the following structure:\n{output_model.model_json_schema()}"
        formatted_messages = [{"role": "system", "content": schema_instruction}]
        formatted_messages.extend(msg for msg in messages if msg["role"] != "system")

        if self.provider == AIProvider.OLLAMA:
            return self._chat_ollama(formatted_messages)
            
        elif self.provider == AIProvider.OPENAI:
            return self._chat_openai(formatted_messages)
            
        elif self.provider == AIProvider.ANTHROPIC:
            return self._chat_anthropic(formatted_messages)
            
        elif self.provider == AIProvider.GEMINI:
            return self._chat_gemini(formatted_messages)
            
        elif self.provider == AIProvider.HUGGINGFACE:
            return self._chat_huggingface(formatted_messages)
            
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _chat_ollama(self, messages: list) -> str:
        """Handle chat with Ollama"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def _chat_openai(self, messages: list) -> str:
        """Handle chat with OpenAI"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_response_len,
                frequency_penalty=self.frequency_penalty
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def _chat_anthropic(self, messages: list) -> str:
        """Handle chat with Anthropic"""
        try:
            # Convert messages to Anthropic format
            prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            completion = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_response_len,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def _chat_gemini(self, messages: list) -> str:
        """Handle chat with Google's Gemini"""
        try:
            model = self.client.GenerativeModel(self.model)
            chat = model.start_chat(temperature=self.temperature)
            for msg in messages:
                response = chat.send_message(msg["content"])
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

    def _chat_huggingface(self, messages: list) -> str:
        """Handle chat with Hugging Face"""
        try:
            # Format messages for Hugging Face
            prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            response = self.client.text_generation(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_length=self.max_response_len
            )
            return response[0]["generated_text"]
        except Exception as e:
            raise RuntimeError(f"Hugging Face API error: {e}")


if __name__ == "__main__":
    # Example usage:
    ai = AiUtil(provider=AIProvider.OLLAMA, model="nemotron-mini:4b-instruct-q4_K_M")

    class Joke(BaseModel):
        """Pydantic model for a joke."""
        setup: str
        punchline: str

    class JokeList(BaseModel):
        """Pydantic model for a list of jokes."""
        jokes: list[Joke]

    response = ai.chat(
        messages=[
            {"role": "user", "content": "Give me 10 funny jokes"}
        ],
        output_model=JokeList
    )
    print(response)
