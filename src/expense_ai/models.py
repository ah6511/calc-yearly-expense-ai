"""Domain models and schemas for bank transactions and classifications."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class ExclusionReason(StrEnum):
    """Reason for excluding or keeping a transaction."""

    PAYMENT_FROM = "payment from"
    PAYROLL = "payroll"
    COLLEGE_SAVINGS = "college savings"
    INVESTMENT = "investment"
    RENT_PAYMENT = "rent payment"
    TUITON_PAYMENT = "tuition payment"
    TAX_REFUND = "tax refund"
    DEPOSIT = "deposit"
    PI_PAYMENT = "pi payment"
    OTHER = "other"


class ExpenseCategory(StrEnum):
    """Transaction expense category returned by classifier."""

    WITHDRAWAL = "withdrawal"
    CREDIT_CARD_PAYMENT = "credit card payment"
    ZELLE_PAYMENT = "zelle payment"
    UTILITY_PAYMENT = "utility expense"
    COMMUNICATION_PAYMENT = "communication expense"
    MORTGAGE_PAYMENT = "mortgage expense"
    HOA_PAYMENT = "hoa expense"
    PROPERTY_TAX_PAYMENT = "property tax expense"
    OTHER = "other"


class Transaction(BaseModel):
    """Bank transaction extracted from a statement."""

    date: str = Field(description="Bank transaction date")
    description: str = Field(description="Bank transaction description")
    amount: float = Field(
        description="Bank transaction amount as a positive number (no negative sign)"
    )

    @field_validator("amount")
    @classmethod
    def make_positive(cls, v: float) -> float:
        return abs(v)


class Transactions(BaseModel):
    """List of bank transactions returned by LLM extraction."""

    transactions: list[Transaction] = Field(description="List of bank transactions")


class CategorizedTransaction(Transaction):
    """Bank transaction enriched with an expense category."""

    category: str = Field(description="Transaction category")


class ClassificationResult(BaseModel):
    """Typed result from transaction exclusion and category classification."""

    exclusion_choice: str
    exclusion_confidence: float = 1.0
    exclusion_probabilities: dict[str, float] = Field(default_factory=dict)
    category_choice: str
    category_confidence: float = 1.0
    category_probabilities: dict[str, float] = Field(default_factory=dict)

    @property
    def should_exclude(self) -> bool:
        """Return True if the transaction should be excluded from expenses."""
        return self.exclusion_choice != ExclusionReason.OTHER.value


class StatementResult(BaseModel):
    """Processed result for a single bank statement PDF."""

    statement_id: str
    transactions: list[CategorizedTransaction]
    transactions_sum: float
