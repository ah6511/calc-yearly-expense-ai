"""Unit tests for StrandsLLMClient and create_llm_client."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from strands.models import BedrockModel
from strands.models.anthropic import AnthropicModel
from strands.models.openai import OpenAIModel

from expense_ai.config import Settings
from expense_ai.llm import StrandsLLMClient, create_llm_client
from expense_ai.models import Transaction, Transactions


def test_strands_llm_client_provider_models(test_settings: Settings) -> None:
    claude_client = create_llm_client(settings=test_settings, provider="claude")
    assert isinstance(claude_client.model, AnthropicModel)
    assert claude_client.default_model == test_settings.claude_model_id

    openai_client = create_llm_client(settings=test_settings, provider="openai")
    assert isinstance(openai_client.model, OpenAIModel)
    assert openai_client.default_model == test_settings.openai_model_id

    bedrock_client = create_llm_client(settings=test_settings, provider="bedrock")
    assert isinstance(bedrock_client.model, BedrockModel)
    assert bedrock_client.default_model == test_settings.bedrock_model_id


def test_strands_llm_client_unsupported_provider(test_settings: Settings) -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        StrandsLLMClient(settings=test_settings, provider="unknown")


def test_strands_llm_client_create_completion(test_settings: Settings) -> None:
    expected = Transactions(
        transactions=[
            Transaction(date="10/24", description="City of Napervil UT Bill", amount=289.78)
        ]
    )
    fake_agent_instance = MagicMock()
    fake_agent_instance.return_value = SimpleNamespace(
        structured_output=expected,
        stop_reason="end_turn",
    )

    with patch("expense_ai.llm.factory.Agent", return_value=fake_agent_instance) as mock_agent_cls:
        client = StrandsLLMClient(settings=test_settings, provider="claude")
        result = client.create_completion(
            response_model=Transactions,
            messages=[
                {"role": "system", "content": "Extract withdrawals."},
                {"role": "user", "content": "10/24 City of Napervil UT Bill 289.78"},
            ],
        )

    assert result == expected
    mock_agent_cls.assert_called_once_with(
        model=client.model,
        system_prompt="Extract withdrawals.",
        callback_handler=None,
    )
    call_args: Any = fake_agent_instance.call_args
    assert call_args.args == ("10/24 City of Napervil UT Bill 289.78",)
    assert call_args.kwargs["structured_output_model"] is Transactions

