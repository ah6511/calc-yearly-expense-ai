"""Unit tests for Gemini configuration, InstructorLLMClient, and CLI provider parsing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import instructor
import pytest
from google.genai import types as genai_types

from expense_ai.cli import build_parser
from expense_ai.config import Settings
from expense_ai.llm import InstructorLLMClient, create_llm_client
from expense_ai.models import Transaction, Transactions


def test_gemini_settings_defaults_and_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(_env_file=None, llm_provider="gemini", base_dir=tmp_path)
    assert settings.gemini_model_id == "gemini-4-neon"
    assert settings.active_model_id == "gemini-4-neon"

    monkeypatch.setenv("GEMINI_API_KEY", "gem-key-1")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    s_gemini = Settings(_env_file=None, llm_provider="gemini", base_dir=tmp_path)
    assert s_gemini.gemini_api_key is not None
    assert s_gemini.gemini_api_key.get_secret_value() == "gem-key-1"

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-key-2")
    s_google = Settings(_env_file=None, llm_provider="gemini", base_dir=tmp_path)
    assert s_google.gemini_api_key is not None
    assert s_google.gemini_api_key.get_secret_value() == "goog-key-2"
    assert s_google.gemini_thinking_level is None

    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "medium")
    s_thinking = Settings(_env_file=None, llm_provider="gemini", base_dir=tmp_path)
    assert s_thinking.gemini_thinking_level == "MEDIUM"

    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "  ")
    s_blank = Settings(_env_file=None, llm_provider="gemini", base_dir=tmp_path)
    assert s_blank.gemini_thinking_level is None


def test_instructor_llm_client_gemini_initialization_and_completion(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        gemini_api_key="gem-test-key",  # type: ignore[arg-type]
        base_dir=tmp_path,
    )
    expected = Transactions(
        transactions=[
            Transaction(date="10/24", description="City of Napervil UT Bill", amount=289.78)
        ]
    )
    fake_genai_client = MagicMock()
    fake_instructor_client = MagicMock()
    fake_instructor_client.chat.completions.create.return_value = expected

    with (
        patch("expense_ai.llm.factory.genai.Client", return_value=fake_genai_client) as mock_genai,
        patch(
            "expense_ai.llm.factory.instructor.from_genai",
            return_value=fake_instructor_client,
        ) as mock_from_genai,
    ):
        client = create_llm_client(settings=settings, provider="gemini")
        assert isinstance(client, InstructorLLMClient)
        assert client.default_model == "gemini-4-neon"

        messages = [
            {"role": "system", "content": "Extract withdrawals."},
            {"role": "user", "content": "10/24 City of Napervil UT Bill 289.78"},
        ]
        result = client.create_completion(response_model=Transactions, messages=messages)

    assert result == expected
    mock_genai.assert_called_once_with(api_key="gem-test-key")
    mock_from_genai.assert_called_once_with(fake_genai_client)
    fake_instructor_client.chat.completions.create.assert_called_once_with(
        model="gemini-4-neon",
        max_retries=settings.max_retries,
        response_model=Transactions,
        messages=messages,
        max_tokens=settings.max_tokens,
    )


def test_instructor_llm_client_gemini_thinking_level(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        gemini_api_key="gem-test-key",  # type: ignore[arg-type]
        gemini_thinking_level="MEDIUM",
        base_dir=tmp_path,
    )
    expected = Transactions(transactions=[])
    fake_genai_client = MagicMock()
    fake_instructor_client = MagicMock()
    fake_instructor_client.chat.completions.create.return_value = expected

    with (
        patch("expense_ai.llm.factory.genai.Client", return_value=fake_genai_client),
        patch("expense_ai.llm.factory.instructor.from_genai", return_value=fake_instructor_client),
    ):
        client = create_llm_client(settings=settings, provider="gemini")
        messages = [{"role": "user", "content": "Extract withdrawals."}]
        client.create_completion(response_model=Transactions, messages=messages)

    fake_instructor_client.chat.completions.create.assert_called_once_with(
        model="gemini-4-neon",
        max_retries=settings.max_retries,
        response_model=Transactions,
        messages=messages,
        max_tokens=settings.max_tokens,
        thinking_config=genai_types.ThinkingConfig(thinking_level="MEDIUM"),
    )


def test_openai_and_claude_effort_settings_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = Settings(_env_file=None, base_dir=tmp_path)
    assert defaults.openai_reasoning_effort is None
    assert defaults.claude_thinking_effort is None

    monkeypatch.setenv("OPENAI_REASONING_EFFORT", " HIGH ")
    monkeypatch.setenv("CLAUDE_THINKING_EFFORT", "MEDIUM")
    s_set = Settings(_env_file=None, base_dir=tmp_path)
    assert s_set.openai_reasoning_effort == "high"
    assert s_set.claude_thinking_effort == "medium"

    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "  ")
    monkeypatch.setenv("CLAUDE_THINKING_EFFORT", "")
    s_blank = Settings(_env_file=None, base_dir=tmp_path)
    assert s_blank.openai_reasoning_effort is None
    assert s_blank.claude_thinking_effort is None


def test_instructor_llm_client_openai_reasoning_effort(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        openai_model_id="gpt-5-mini",
        openai_reasoning_effort="medium",
        base_dir=tmp_path,
    )
    expected = Transactions(transactions=[])
    fake_openai_client = MagicMock()
    fake_instructor_client = MagicMock()
    fake_instructor_client.chat.completions.create.return_value = expected

    with (
        patch("expense_ai.llm.factory.OpenAI", return_value=fake_openai_client),
        patch(
            "expense_ai.llm.factory.instructor.from_openai",
            return_value=fake_instructor_client,
        ) as mock_from_openai,
    ):
        client = create_llm_client(settings=settings, provider="openai")
        messages = [{"role": "user", "content": "Extract withdrawals."}]
        client.create_completion(response_model=Transactions, messages=messages)

    mock_from_openai.assert_called_once_with(
        fake_openai_client,
        mode=instructor.Mode.JSON_SCHEMA,
    )

    fake_instructor_client.chat.completions.create.assert_called_once_with(
        model="gpt-5-mini",
        max_retries=settings.max_retries,
        response_model=Transactions,
        messages=messages,
        max_completion_tokens=settings.max_tokens,
        reasoning_effort="medium",
    )


def test_instructor_llm_client_claude_thinking_effort(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="claude",
        claude_api_key="sk-ant-test",  # type: ignore[arg-type]
        claude_thinking_effort="low",
        base_dir=tmp_path,
    )
    expected = Transactions(transactions=[])
    fake_anthropic_client = MagicMock()
    fake_instructor_client = MagicMock()
    fake_instructor_client.chat.completions.create.return_value = expected

    with (
        patch("expense_ai.llm.factory.anthropic.Anthropic", return_value=fake_anthropic_client),
        patch(
            "expense_ai.llm.factory.instructor.from_anthropic",
            return_value=fake_instructor_client,
        ),
    ):
        client = create_llm_client(settings=settings, provider="claude")
        messages = [{"role": "user", "content": "Extract withdrawals."}]
        client.create_completion(response_model=Transactions, messages=messages)

    fake_instructor_client.chat.completions.create.assert_called_once_with(
        model="claude-sonnet-4-6",
        max_retries=settings.max_retries,
        response_model=Transactions,
        messages=messages,
        max_tokens=settings.max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        tool_choice={"type": "auto"},
    )


def test_instructor_llm_client_omits_effort_when_unset(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="claude",
        claude_api_key="sk-ant-test",  # type: ignore[arg-type]
        base_dir=tmp_path,
    )
    expected = Transactions(transactions=[])
    fake_anthropic_client = MagicMock()
    fake_instructor_client = MagicMock()
    fake_instructor_client.chat.completions.create.return_value = expected

    with (
        patch("expense_ai.llm.factory.anthropic.Anthropic", return_value=fake_anthropic_client),
        patch(
            "expense_ai.llm.factory.instructor.from_anthropic",
            return_value=fake_instructor_client,
        ),
    ):
        client = create_llm_client(settings=settings, provider="claude")
        messages = [{"role": "user", "content": "Extract withdrawals."}]
        client.create_completion(response_model=Transactions, messages=messages)

    fake_instructor_client.chat.completions.create.assert_called_once_with(
        model="claude-sonnet-4-6",
        max_retries=settings.max_retries,
        response_model=Transactions,
        messages=messages,
        max_tokens=settings.max_tokens,
    )


def test_cli_accepts_gemini_provider() -> None:
    args = build_parser().parse_args(["--provider", "gemini"])
    assert args.provider == "gemini"
