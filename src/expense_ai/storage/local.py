"""Local filesystem storage backend."""

import csv
import io
import json
from pathlib import Path

from expense_ai.config import Settings, get_settings
from expense_ai.models import CategorizedTransaction
from expense_ai.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    """Reads PDFs and writes JSON/CSV artifacts on the local filesystem."""

    def __init__(
        self,
        settings: Settings | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_dir = Path(base_dir) if base_dir is not None else Path(self.settings.base_dir)
        self.pdf_dir = self.base_dir / self.settings.pdf_dir
        self.transactions_dir = self.base_dir / self.settings.transactions_dir
        self.year_total_dir = self.base_dir / self.settings.year_total_dir

    def list_pdfs(self) -> list[str]:
        if not self.pdf_dir.exists() or not self.pdf_dir.is_dir():
            return []
        pdfs = [
            str(p)
            for p in self.pdf_dir.rglob("*")
            if p.is_file() and p.suffix.lower() == ".pdf"
        ]
        pdfs.sort()
        return pdfs

    def resolve_local_pdf(self, pdf_ref: str) -> Path:
        path = Path(pdf_ref)
        if not path.is_absolute() and not path.exists():
            candidate = self.pdf_dir / pdf_ref
            if candidate.exists():
                return candidate
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_ref}")
        return path

    def save_transactions(
        self, statement_id: str, transactions: list[CategorizedTransaction]
    ) -> str:
        self.transactions_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.transactions_dir / f"{statement_id}.json"
        payload = [tx.model_dump() for tx in transactions]
        out_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
        return str(out_path.resolve())

    def save_year_total(self, statement_id: str, transactions_sum: float) -> str:
        self.year_total_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.year_total_dir / f"{statement_id}.json"
        out_path.write_text(
            json.dumps({"year_total": transactions_sum}, indent=4), encoding="utf-8"
        )
        return str(out_path.resolve())

    def save_total_withdrawals_csv(self, transactions: list[CategorizedTransaction]) -> str:
        self.transactions_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.transactions_dir / "total_withdrawals.csv"
        out_path.write_text(serialize_transactions_csv(transactions), encoding="utf-8")
        return str(out_path.resolve())


def serialize_transactions_csv(transactions: list[CategorizedTransaction]) -> str:
    """Serialize categorized transactions to CSV text with standard headers."""
    fieldnames = ["date", "description", "amount", "category"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for tx in transactions:
        writer.writerow(tx.model_dump())
    return buf.getvalue()
