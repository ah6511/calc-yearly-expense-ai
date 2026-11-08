"""Storage backend protocol for statement inputs and transaction outputs."""

from pathlib import Path
from typing import Protocol

from expense_ai.models import CategorizedTransaction


class StorageBackend(Protocol):
    """Unified interface for local filesystem or S3 storage."""

    def list_pdfs(self) -> list[str]:
        """Return sorted list of PDF identifiers (local paths or S3 keys)."""
        ...

    def resolve_local_pdf(self, pdf_ref: str) -> Path:
        """Return a local filesystem Path to the PDF (downloading if necessary)."""
        ...

    def save_transactions(
        self, statement_id: str, transactions: list[CategorizedTransaction]
    ) -> str:
        """Save categorized transactions JSON for a statement."""
        ...

    def save_year_total(self, statement_id: str, transactions_sum: float) -> str:
        """Save statement total JSON (`{"year_total": sum}`)."""
        ...

    def save_total_withdrawals_csv(self, transactions: list[CategorizedTransaction]) -> str:
        """Save aggregated yearly withdrawals CSV (`total_withdrawals.csv`)."""
        ...
