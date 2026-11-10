"""Strands Agents SDK-backed LLM client factory supporting OpenAI, Claude, and AWS Bedrock."""

from typing import Any, TypeVar, cast

import boto3
from pydantic import BaseModel
from strands import Agent
from strands.models import BedrockModel
from strands.models.anthropic import AnthropicModel
from strands.models.model import Model
from strands.models.openai import OpenAIModel
from strands.types.agent import Limits

from expense_ai.config import Settings, get_settings
from expense_ai.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class StrandsLLMClient(LLMClient):
    """Structured LLM client backed by the `strands-agents` SDK."""

    def __init__(self, settings: Settings | None = None, provider: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider = (provider or self.settings.llm_provider).lower()
        self.model = self._build_model(
            model_id=self.default_model,
            max_tokens=self.settings.max_tokens,
        )

    def _build_model(self, model_id: str, max_tokens: int) -> Model:
        match self.provider:
            case "openai":
                api_key = (
                    self.settings.openai_api_key.get_secret_value()
                    if self.settings.openai_api_key
                    else None
                )
                client_args: dict[str, Any] = {}
                if api_key:
                    client_args["api_key"] = api_key
                token_key = (
                    "max_completion_tokens" if model_id.startswith("gpt-5") else "max_tokens"
                )
                return OpenAIModel(
                    client_args=client_args,
                    model_id=model_id,
                    params={token_key: max_tokens},
                )
            case "claude":
                api_key = (
                    self.settings.claude_api_key.get_secret_value()
                    if self.settings.claude_api_key
                    else None
                )
                anthropic_client_args: dict[str, Any] = {}
                if api_key:
                    anthropic_client_args["api_key"] = api_key
                return AnthropicModel(
                    client_args=anthropic_client_args,
                    model_id=model_id,
                    max_tokens=max_tokens,
                )
            case "bedrock":
                session_kwargs: dict[str, Any] = {
                    "region_name": self.settings.aws_region,
                }
                if self.settings.aws_access_key_id:
                    session_kwargs["aws_access_key_id"] = self.settings.aws_access_key_id
                if self.settings.aws_secret_access_key:
                    session_kwargs["aws_secret_access_key"] = (
                        self.settings.aws_secret_access_key.get_secret_value()
                    )
                if self.settings.aws_session_token:
                    session_kwargs["aws_session_token"] = (
                        self.settings.aws_session_token.get_secret_value()
                    )
                boto_session = boto3.Session(**session_kwargs)
                return BedrockModel(
                    boto_session=boto_session,
                    model_id=model_id,
                    max_tokens=max_tokens,
                )
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

    @staticmethod
    def _split_messages(messages: list[dict[str, str]]) -> tuple[str | None, str]:
        """Extract system prompt and user prompt from a list of chat messages."""
        system_parts: list[str] = []
        user_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
        system_prompt = "\n\n".join(system_parts) if system_parts else None
        user_prompt = "\n\n".join(user_parts)
        return system_prompt, user_prompt

    def create_completion(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> T:
        model_id = str(kwargs.get("model", self.default_model))
        max_tokens = int(kwargs.get("max_tokens", self.settings.max_tokens))
        max_retries = int(kwargs.get("max_retries", self.settings.max_retries))

        model = (
            self.model
            if model_id == self.default_model and max_tokens == self.settings.max_tokens
            else self._build_model(model_id=model_id, max_tokens=max_tokens)
        )

        system_prompt, user_prompt = self._split_messages(messages)

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            callback_handler=None,
        )
        limits: Limits | None = (
            Limits(structured_output_attempts=max_retries) if max_retries > 0 else None
        )
        result = agent(
            user_prompt,
            structured_output_model=response_model,
            limits=limits,
        )
        if result.structured_output is None:
            raise RuntimeError(
                f"Strands agent did not return structured output (stop_reason={result.stop_reason!r})"
            )
        return cast(T, result.structured_output)


def create_llm_client(
    settings: Settings | None = None,
    provider: str | None = None,
) -> StrandsLLMClient:
    """Instantiate a StrandsLLMClient from application settings."""
    return StrandsLLMClient(settings=settings, provider=provider)
