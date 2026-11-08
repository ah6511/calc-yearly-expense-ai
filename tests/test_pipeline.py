"""End-to-end tests for ExpensePipeline using in-memory fakes (no network calls)."""

import json
from pathlib import Path

from tests.conftest import FakeClassifier, FakeLLMClient

from expense_ai.config import Settings
from expense_ai.models import ClassificationResult, Transaction
from expense_ai.pipeline import ExpensePipeline, parse_statement_id
from expense_ai.storage.local import LocalStorage


def test_parse_statement_id() -> None:
    assert parse_statement_id("/path/to/20251125-statements-8506-.pdf") == "20251125"
    assert parse_statement_id("20240223-statements-8506-.pdf") == "20240223"


async def test_pipeline_filters_excluded_and_saves_outputs(
    test_settings: Settings,
    local_storage: LocalStorage,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "pdf" / "20251125-statements-8506-.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    raw_transactions = [
        Transaction(date="10/29", description="Chase Credit Crd Autopay", amount=39.58),
        Transaction(date="10/24", description="City of Napervil UT Bill", amount=-289.78),
        Transaction(date="10/25", description="Fid Bkg Svc LLC Moneyline", amount=1000.00),
    ]
    fake_llm = FakeLLMClient(transactions=raw_transactions)
    fake_classifier = FakeClassifier(
        mapping={
            "City of Napervil UT Bill": ClassificationResult(
                exclusion_choice="other",
                category_choice="utility_payment",
            ),
            "Chase Credit Crd Autopay": ClassificationResult(
                exclusion_choice="other",
                category_choice="credit_card_payment",
            ),
            "Fid Bkg Svc LLC Moneyline": ClassificationResult(
                exclusion_choice="investment",
                category_choice="other",
            ),
        }
    )

    pipeline = ExpensePipeline(
        settings=test_settings,
        llm_client=fake_llm,
        classifier=fake_classifier,
        storage=local_storage,
        pdf_extractor=lambda _: "dummy statement text",
    )

    total_sum, withdrawals = await pipeline.process_all_statements()

    assert len(withdrawals) == 2
    assert [w.date for w in withdrawals] == ["10/24", "10/29"]
    assert withdrawals[0].category == "utility_payment"
    assert withdrawals[1].category == "credit_card_payment"
    assert total_sum == 329.36

    tx_json = json.loads(
        (tmp_path / "transactions" / "20251125.json").read_text(encoding="utf-8")
    )
    assert len(tx_json) == 2

    year_total_json = json.loads(
        (tmp_path / "year_total" / "20251125.json").read_text(encoding="utf-8")
    )
    assert year_total_json == {"year_total": 329.36}

    csv_file = tmp_path / "transactions" / "total_withdrawals.csv"
    assert csv_file.exists()
