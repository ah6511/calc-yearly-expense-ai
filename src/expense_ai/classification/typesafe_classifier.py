"""TypeSafe System One classifier for transaction exclusion and categorization."""

import asyncio
import logging
from typing import Protocol

from typesafe_sdk import AsyncTypeSafeClient, Choice, TypeSafeClient

from expense_ai.classification.criteria import (
    CATEGORY_CRITERIA,
    CATEGORY_INSTRUCTIONS,
    EXCLUSION_CRITERIA,
    EXCLUSION_INSTRUCTIONS,
)
from expense_ai.config import Settings, get_settings
from expense_ai.models import ClassificationResult

logger = logging.getLogger(__name__)


class TransactionClassifier(Protocol):
    """Protocol for classifying transaction descriptions."""

    def classify(self, description: str) -> ClassificationResult:
        """Classify a single transaction description synchronously."""
        ...

    async def classify_many(self, descriptions: list[str]) -> list[ClassificationResult]:
        """Classify a batch of transaction descriptions asynchronously."""
        ...


class TypeSafeClassifier(TransactionClassifier):
    """Transaction classifier powered by TypeSafe AI System One."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.typesafe_model
        resolved_key = api_key or (
            self.settings.typesafe_api_key.get_secret_value()
            if self.settings.typesafe_api_key
            else None
        )
        self._api_key = resolved_key
        self._sync_client: TypeSafeClient | None = None

    @property
    def sync_client(self) -> TypeSafeClient:
        if self._sync_client is None:
            self._sync_client = TypeSafeClient(api_key=self._api_key, model=self.model)
        return self._sync_client

    @staticmethod
    def _build_questions() -> dict[str, Choice]:
        return {
            "exclude": Choice(
                instructions=EXCLUSION_INSTRUCTIONS,
                criteria=EXCLUSION_CRITERIA,
            ),
            "category": Choice(
                instructions=CATEGORY_INSTRUCTIONS,
                criteria=CATEGORY_CRITERIA,
            ),
        }

    @staticmethod
    def _parse_response(description: str, response: object) -> ClassificationResult:
        choices = getattr(response, "choices", {})
        exclusion = choices["exclude"]
        expense_category = choices["category"]

        result = ClassificationResult(
            exclusion_choice=str(exclusion.choice),
            exclusion_confidence=float(exclusion.confidence),
            exclusion_probabilities=dict(exclusion.probabilities or {}),
            category_choice=str(expense_category.choice),
            category_confidence=float(expense_category.confidence),
            category_probabilities=dict(expense_category.probabilities or {}),
        )
        if result.should_exclude:
            logger.info(
                "Excluded transaction %r -> reason=%s (conf=%.3f), category=%s (conf=%.3f)",
                description,
                result.exclusion_choice,
                result.exclusion_confidence,
                result.category_choice,
                result.category_confidence,
            )
        return result

    def classify(self, description: str) -> ClassificationResult:
        """Classify a single transaction description synchronously."""
        response = self.sync_client.system_one(
            state=description,
            questions=self._build_questions(),
        )
        return self._parse_response(description, response)

    async def classify_many(self, descriptions: list[str]) -> list[ClassificationResult]:
        """Classify multiple transaction descriptions concurrently with bounded concurrency."""
        if not descriptions:
            return []

        semaphore = asyncio.Semaphore(self.settings.typesafe_max_concurrency)
        questions = self._build_questions()

        async with AsyncTypeSafeClient(api_key=self._api_key, model=self.model) as client:

            async def _one(desc: str) -> ClassificationResult:
                async with semaphore:
                    response = await client.system_one(state=desc, questions=questions)
                    return self._parse_response(desc, response)

            return list(await asyncio.gather(*(_one(d) for d in descriptions)))


if __name__ == "__main__":
    from expense_ai.logging_config import configure_logging

    configure_logging()
    classifier = TypeSafeClassifier()
    result = classifier.classify("U IL Tuition Pmt ACH Pay 383906943 Web ID: 9376000511")
    assert result.exclusion_choice == "tuition payment"
    logger.info(result)
