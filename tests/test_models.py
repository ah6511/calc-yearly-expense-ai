"""Unit tests for domain models and settings."""

from pathlib import Path

from expense_ai.config import Settings
from expense_ai.models import (
    CategorizedTransaction,
    ClassificationResult,
    Transaction,
)


def test_transaction_amount_always_positive() -> None:
    tx = Transaction(date="10/24", description="City of Napervil UT Bill", amount=-289.78)
    assert tx.amount == 289.78

    cat_tx = CategorizedTransaction(
        date="10/29",
        description="Chase Credit Crd Autopay",
        amount=-39.58,
        category="credit_card_payment",
    )
    assert cat_tx.amount == 39.58


def test_classification_result_should_exclude() -> None:
    keep = ClassificationResult(exclusion_choice="other", category_choice="utility_payment")
    assert keep.should_exclude is False

    exclude = ClassificationResult(exclusion_choice="investment", category_choice="other")
    assert exclude.should_exclude is True


def test_settings_resolves_storage_backend(tmp_path: Path) -> None:
    s1 = Settings(_env_file=None, local_mode=True, base_dir=tmp_path)
    assert s1.storage_backend == "local"

    s2 = Settings(
        _env_file=None,
        local_mode=False,
        use_local=False,
        bucket_name="my-bucket",
        base_dir=tmp_path,
    )
    assert s2.storage_backend == "s3"

    s3 = Settings(_env_file=None, storage_backend="s3", bucket_name="b", base_dir=tmp_path)
    assert s3.storage_backend == "s3"

