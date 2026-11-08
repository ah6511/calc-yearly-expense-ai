"""End-to-end statement extraction, classification, and aggregation pipeline."""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from expense_ai.classification import TransactionClassifier, TypeSafeClassifier
from expense_ai.config import Settings, get_settings
from expense_ai.llm import LLMClient, create_llm_client
from expense_ai.models import (
    CategorizedTransaction,
    StatementResult,
    Transaction,
    Transactions,
)
from expense_ai.pdf import extract_text_from_pdf
from expense_ai.prompts import load_prompt
from expense_ai.storage import StorageBackend, create_storage

logger = logging.getLogger(__name__)


def parse_statement_id(pdf_path: str | Path) -> str:
    """Derive statement identifier (e.g. '20251125') from a statement PDF path."""
    filename = Path(pdf_path).stem
    return filename.split("-")[0]


class ExpensePipeline:
    """Orchestrates PDF extraction, TypeSafe classification, and storage persistence."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
        classifier: TransactionClassifier | None = None,
        storage: StorageBackend | None = None,
        pdf_extractor: Callable[[str | Path], str] = extract_text_from_pdf,
        system_prompt: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm_client = llm_client
        self._classifier = classifier
        self._storage = storage
        self.pdf_extractor = pdf_extractor
        self.system_prompt = system_prompt or load_prompt("extract_transactions.md")

    @property
    def llm_client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = create_llm_client(self.settings)
        return self._llm_client

    @property
    def classifier(self) -> TransactionClassifier:
        if self._classifier is None:
            self._classifier = TypeSafeClassifier(self.settings)
        return self._classifier

    @property
    def storage(self) -> StorageBackend:
        if self._storage is None:
            self._storage = create_storage(self.settings)
        return self._storage

    def extract_transactions(self, pdf_path: str | Path) -> list[Transaction]:
        """Extract raw withdrawal transactions from a statement PDF via structured LLM call."""
        local_pdf = self.storage.resolve_local_pdf(str(pdf_path))
        doc_text = self.pdf_extractor(local_pdf)
        completion = self.llm_client.create_completion(
            response_model=Transactions,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": doc_text},
            ],
        )
        transactions = list(completion.transactions)
        transactions.sort(key=lambda tx: tx.date)
        return transactions

    def filter_and_categorize(
        self, transactions: list[Transaction]
    ) -> list[CategorizedTransaction]:
        """Classify transactions synchronously and keep non-excluded ones."""
        kept: list[CategorizedTransaction] = []
        for tx in transactions:
            result = self.classifier.classify(tx.description)
            if not result.should_exclude:
                kept.append(
                    CategorizedTransaction(
                        date=tx.date,
                        description=tx.description,
                        amount=tx.amount,
                        category=result.category_choice,
                    )
                )
            else:
                logger.info(
                    "Excluding transaction: %s, reason is %s",
                    tx.model_dump(),
                    result.exclusion_choice,
                )
        return kept

    async def filter_and_categorize_async(
        self, transactions: list[Transaction]
    ) -> list[CategorizedTransaction]:
        """Classify transactions concurrently via AsyncTypeSafeClient and keep non-excluded ones."""
        descriptions = [tx.description for tx in transactions]
        results = await self.classifier.classify_many(descriptions)
        kept: list[CategorizedTransaction] = []
        for tx, result in zip(transactions, results, strict=True):
            if not result.should_exclude:
                kept.append(
                    CategorizedTransaction(
                        date=tx.date,
                        description=tx.description,
                        amount=tx.amount,
                        category=result.category_choice,
                    )
                )
            else:
                logger.info(
                    "Excluding transaction: %s, reason is %s",
                    tx.model_dump(),
                    result.exclusion_choice,
                )
        return kept

    @staticmethod
    def sum_transactions(transactions: list[CategorizedTransaction]) -> float:
        """Return total sum of transaction amounts rounded to 2 decimal places."""
        return round(sum(tx.amount for tx in transactions), 2)

    def process_statement(self, pdf_ref: str | Path) -> StatementResult:
        """Process a single PDF statement synchronously and persist JSON results."""
        statement_id = parse_statement_id(pdf_ref)
        logger.info("Processing statement %s (%s)", statement_id, pdf_ref)
        raw_transactions = self.extract_transactions(pdf_ref)
        categorized = self.filter_and_categorize(raw_transactions)
        total = self.sum_transactions(categorized)

        self.storage.save_transactions(statement_id, categorized)
        self.storage.save_year_total(statement_id, total)

        month = statement_id[4:6] if len(statement_id) >= 6 else statement_id
        logger.info("Expense for month %s is %.2f.", month, total)
        return StatementResult(
            statement_id=statement_id,
            transactions=categorized,
            transactions_sum=total,
        )

    async def process_statement_async(self, pdf_ref: str | Path) -> StatementResult:
        """Process a single PDF statement asynchronously."""
        statement_id = parse_statement_id(pdf_ref)
        logger.info("Processing statement %s (%s)", statement_id, pdf_ref)
        raw_transactions = await asyncio.to_thread(self.extract_transactions, pdf_ref)
        categorized = await self.filter_and_categorize_async(raw_transactions)
        total = self.sum_transactions(categorized)

        await asyncio.to_thread(self.storage.save_transactions, statement_id, categorized)
        await asyncio.to_thread(self.storage.save_year_total, statement_id, total)

        month = statement_id[4:6] if len(statement_id) >= 6 else statement_id
        logger.info("Expense for month %s is %.2f.", month, total)
        return StatementResult(
            statement_id=statement_id,
            transactions=categorized,
            transactions_sum=total,
        )

    async def process_all_statements(
        self, pdf_refs: list[str] | None = None
    ) -> tuple[float, list[CategorizedTransaction]]:
        """Process all discovered PDF statements and write `total_withdrawals.csv`."""
        refs = pdf_refs if pdf_refs is not None else self.storage.list_pdfs()
        if not refs:
            logger.warning("No PDF statements found to process.")
            return 0.0, []

        results = await asyncio.gather(*(self.process_statement_async(ref) for ref in refs))
        total_sum = round(sum(r.transactions_sum for r in results), 2)
        withdrawals = [tx for r in results for tx in r.transactions]
        withdrawals.sort(key=lambda tx: tx.date)

        logger.info("The number of withdrawals for the year: %d", len(withdrawals))
        logger.info("Total sum of withdrawals for the year: %.2f", total_sum)

        if withdrawals:
            self.storage.save_total_withdrawals_csv(withdrawals)

        return total_sum, withdrawals
