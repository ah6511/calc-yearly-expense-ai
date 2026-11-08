"""Instructor-backed LLM client factory supporting OpenAI, Claude, and AWS Bedrock."""

from typing import Any, TypeVar

import anthropic
import boto3
import instructor
from openai import OpenAI
from pydantic import BaseModel

from expense_ai.config import Settings, get_settings
from expense_ai.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class InstructorLLMClient(LLMClient):
    """Structured LLM client backed by the `instructor` library."""

    def __init__(self, settings: Settings | None = None, provider: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider = (provider or self.settings.llm_provider).lower()
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        match self.provider:
            case "openai":
                api_key = (
                    self.settings.openai_api_key.get_secret_value()
                    if self.settings.openai_api_key
                    else None
                )
                return instructor.from_openai(OpenAI(api_key=api_key))
            case "claude":
                api_key = (
                    self.settings.claude_api_key.get_secret_value()
                    if self.settings.claude_api_key
                    else None
                )
                return instructor.from_anthropic(anthropic.Anthropic(api_key=api_key))
            case "bedrock":
                client_kwargs: dict[str, Any] = {
                    "service_name": "bedrock-runtime",
                    "region_name": self.settings.aws_region,
                }
                if self.settings.aws_access_key_id:
                    client_kwargs["aws_access_key_id"] = self.settings.aws_access_key_id
                if self.settings.aws_secret_access_key:
                    client_kwargs["aws_secret_access_key"] = (
                        self.settings.aws_secret_access_key.get_secret_value()
                    )
                if self.settings.aws_session_token:
                    client_kwargs["aws_session_token"] = (
                        self.settings.aws_session_token.get_secret_value()
                    )
                return instructor.from_bedrock(boto3.client(**client_kwargs))
            case _:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")

    @property
    def default_model(self) -> str:
        match self.provider:
            case "openai":
                return self.settings.openai_model_id
            case "claude":
                return self.settings.claude_model_id
            case "bedrock":
                return self.settings.bedrock_model_id
            case _:
                return self.settings.active_model_id

    def create_completion(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> T:
        model = str(kwargs.get("model", self.default_model))
        max_tokens = kwargs.get("max_tokens", self.settings.max_tokens)

        completion_params: dict[str, Any] = {
            "model": model,
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            # "temperature": kwargs.get("temperature", self.settings.temperature),
            "response_model": response_model,
            "messages": messages,
        }

        if model.startswith("gpt-5"):
            completion_params["max_completion_tokens"] = max_tokens
        else:
            completion_params["max_tokens"] = max_tokens

        result: T = self.client.chat.completions.create(**completion_params)
        return result


def create_llm_client(
    settings: Settings | None = None,
    provider: str | None = None,
) -> InstructorLLMClient:
    """Instantiate an InstructorLLMClient from application settings."""
    return InstructorLLMClient(settings=settings, provider=provider)
