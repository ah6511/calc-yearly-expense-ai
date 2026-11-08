"""Pytest fixtures providing fake LLM, fake TypeSafe classifier, and temporary local storage."""

from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from expense_ai.classification import TransactionClassifier
from expense_ai.config import Settings
from expense_ai.llm import LLMClient
from expense_ai.models import ClassificationResult, Transaction, Transactions
from expense_ai.storage import LocalStorage

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient(LLMClient):
    """Deterministic in-memory LLM client returning preconfigured transactions."""

    def __init__(self, transactions: list[Transaction]) -> None:
        self.transactions = transactions
        self.calls: list[dict[str, Any]] = []

    def create_completion(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> T:
        self.calls.append({"response_model": response_model, "messages": messages, "kwargs": kwargs})
        return Transactions(transactions=self.transactions)  # type: ignore[return-value]


class FakeClassifier(TransactionClassifier):
    """Deterministic in-memory transaction classifier driven by rules."""

    def __init__(self, mapping: dict[str, ClassificationResult]) -> None:
        self.mapping = mapping
        self.default = ClassificationResult(
            exclusion_choice="other",
            category_choice="other",
        )

    def classify(self, description: str) -> ClassificationResult:
        return self.mapping.get(description, self.default)

    async def classify_many(self, descriptions: list[str]) -> list[ClassificationResult]:
        return [self.classify(d) for d in descriptions]


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Return isolated Settings rooted in a temporary directory."""
    return Settings(
        _env_file=None,
        llm_provider="claude",
        storage_backend="local",
        base_dir=tmp_path,
        typesafe_api_key="ts_test_key",  # type: ignore[arg-type]
        claude_api_key="sk-ant-test",  # type: ignore[arg-type]
    )


@pytest.fixture
def local_storage(test_settings: Settings, tmp_path: Path) -> LocalStorage:
    """Return a LocalStorage instance rooted in `tmp_path`."""
    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return LocalStorage(settings=test_settings, base_dir=tmp_path)
