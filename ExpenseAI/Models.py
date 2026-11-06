from typing import List
from pydantic import BaseModel, Field, field_validator

class Transaction(BaseModel):
    """Bank transaction."""

    date: str = Field(description="Bank transaction date")
    description: str = Field(description="Bank transaction description")
    amount: float = Field(description="Bank transaction amount as a positive number (no negative sign)")

    @field_validator("amount")
    @classmethod
    def make_positive(cls, v: float) -> float:
        return abs(v)


class Transactions(BaseModel):
    """List of bank transaction."""

    transactions: List[Transaction] = Field(description="List of bank transactions")

class CategorizedTransaction(Transaction):
    """Bank transaction with category."""

    category: str = Field(description="Transaction category")
