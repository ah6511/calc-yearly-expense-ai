"""Declarative criteria for transaction exclusion and categorization."""


from expense_ai.models import ExclusionReason, ExpenseCategory

EXCLUSION_INSTRUCTIONS = (
    "Classify whether this bank transaction matches one of the excluded transaction types, "
    "or 'other' if it does not match any of them."
)

EXCLUSION_CRITERIA: dict[str, str] = {
    ExclusionReason.PAYMENT_FROM.value: "Payment from someone else",
    ExclusionReason.PAYROLL.value: "Payroll transaction",
    ExclusionReason.COLLEGE_SAVINGS.value: "Payment to or from College Savings 529 account",
    ExclusionReason.INVESTMENT.value: "Fidelity brokerage (Fid Bkg) or Vanguard investment",
    ExclusionReason.RENT_PAYMENT.value: "Rent or lease payment (college campus or landlord)",
    ExclusionReason.TUITON_PAYMENT.value: ("College or university tuition payment"),
    ExclusionReason.TAX_REFUND.value: "Tax refund transaction",
    ExclusionReason.DEPOSIT.value: "Deposit transaction",
    ExclusionReason.PI_PAYMENT.value: "Payment related to PI*",
    ExclusionReason.OTHER.value: "Any other regular expense or payment not listed above",
}

CATEGORY_INSTRUCTIONS = "Categorize the transaction"

CATEGORY_CRITERIA: dict[str, str] = {
    ExpenseCategory.WITHDRAWAL.value: "ATM or other withdrawal",
    ExpenseCategory.CREDIT_CARD_PAYMENT.value: "Credit card payment",
    ExpenseCategory.ZELLE_PAYMENT.value: "Zelle payment to somebody",
    ExpenseCategory.UTILITY_PAYMENT.value: "Utility payment (e.g. electricity, gas, water)",
    ExpenseCategory.COMMUNICATION_PAYMENT.value: ("Communication payment to telecom or ISP (e.g. att, verizon, comcast, etc)"),
    ExpenseCategory.MORTGAGE_PAYMENT.value: "Mortgage payment",
    ExpenseCategory.HOA_PAYMENT.value: "Stillwater Homeowners Association (HOA) payment",
    ExpenseCategory.PROPERTY_TAX_PAYMENT.value: "Property tax payment",
    ExpenseCategory.OTHER.value: "Other withdrawal or payment",
}
