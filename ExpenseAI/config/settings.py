from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class LLMProviderSettings(BaseSettings):
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = int(os.getenv("MAX_RETRIES", 3))

class OpenAISettings(LLMProviderSettings):
    api_key: str = os.getenv("OPENAI_API_KEY")
    default_model: str = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")

class ClaudeSettings(LLMProviderSettings):
    api_key: str = os.getenv("ANTHROPIC_API_KEY")
    default_model: str = os.getenv("CLAUDE_MODEL_ID", "claude-sonnet-4-6")

class BedrockSettings(LLMProviderSettings):
    default_model: str = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")
    max_tokens: int = int(os.getenv("MAX_TOKENS", 1024))

class Settings(BaseSettings):
    app_name: str = "expense-ai"
    openai: OpenAISettings = OpenAISettings()
    claude: ClaudeSettings = ClaudeSettings()
    bedrock: BedrockSettings = BedrockSettings()

@lru_cache
def get_settings():
    return Settings()