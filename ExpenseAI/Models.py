from typing import List
from enum import Enum
from pydantic import BaseModel, Field

class ExpenseCategory(str, Enum):
    ATM_WITHDRAWAL = "ATM withdrawal"
    OTHER_WITHDRAWAL = "Other withdrawal"
    CREDIT_CARD_PAYMENT = "Credit card payment"
    ZELLE_PAYMENT = "Zelle payment"
    HOA_PAYMENT = "Home Owners Association (HOA) payment"
    UTILITY_PAYMENT = "Utility payment"
    HOME_LOAN_MORTGAGE_PAYMENT = "Home loan mortgage payment"
    PROPERTY_TAX_PAYMENT = "Property tax (Williltax) payment"
    OTHER_PAYMENT = "Other payment"

class Transaction(BaseModel):
    """Bank transaction."""

    date: str = Field(description="Bank transaction date")
    description: str = Field(description="Bank transaction description")
    amount: float = Field(description="Bank transaction amount")
    category: ExpenseCategory = Field(description="Transaction category")


class Transactions(BaseModel):
    """List of bank transaction."""

    transactions: List[Transaction] = Field(description="List of bank transactions")
