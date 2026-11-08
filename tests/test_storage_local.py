"""Unit tests for LocalStorage."""

import json
from pathlib import Path

from expense_ai.models import CategorizedTransaction
from expense_ai.storage.local import LocalStorage


def test_local_storage_roundtrip(local_storage: LocalStorage, tmp_path: Path) -> None:
    pdf_file = tmp_path / "pdf" / "20251125-statements-8506-.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy")

    discovered = local_storage.list_pdfs()
    assert len(discovered) == 1
    assert discovered[0].endswith("20251125-statements-8506-.pdf")
    assert local_storage.resolve_local_pdf("20251125-statements-8506-.pdf") == pdf_file

    txs = [
        CategorizedTransaction(
            date="10/24",
            description="City of Napervil UT Bill",
            amount=289.78,
            category="utility_payment",
        )
    ]
    tx_path = Path(local_storage.save_transactions("20251125", txs))
    assert tx_path.exists()
    assert json.loads(tx_path.read_text(encoding="utf-8")) == [
        {
            "date": "10/24",
            "description": "City of Napervil UT Bill",
            "amount": 289.78,
            "category": "utility_payment",
        }
    ]

    total_path = Path(local_storage.save_year_total("20251125", 289.78))
    assert json.loads(total_path.read_text(encoding="utf-8")) == {"year_total": 289.78}

    csv_path = Path(local_storage.save_total_withdrawals_csv(txs))
    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0] == "date,description,amount,category"
    assert csv_lines[1] == "10/24,City of Napervil UT Bill,289.78,utility_payment"

