from typesafe_sdk import Choice, TypeSafeClient

class TypeSafeClientWrapper:
    def __init__(self, model="jev-latest") -> None:
        self.client = TypeSafeClient(model=model)
    
    def exclude_transaction(self, transaction_description: str) -> tuple[str, str]:
        response = self.client.system_one(
            state=transaction_description,
            questions={
                "exclude": Choice(
                    instructions="Is this a transaction that should be excluded?",
                    criteria={
                        "payment_from": "Payment from someone else",
                        "payroll": "Payroll transaction",
                        "college_savings": "Payment to or from College Savings 529 account",
                        "investment": "Fidelity brokerage (Fid Bkg) or Vanguard investment",
                        "rent_payment": "Rent or lease payment (college campus or landlord)",
                        "tax_refund": "Tax refund transaction",
                        "deposit": "Deposit transaction",
                        "pi_payment": "Payment related to PI*",
                        "other": "Other"
                    },
                ),
                "category": Choice(
                    instructions="Categorize the transaction",
                    criteria={
                        "withdrawal": "ATM or other withdrawal",
                        "credit_card_payment": "Credit card payment",
                        "zelle_payment": "Zelle payment to somebody",
                        "utility_payment": "Utility payment (e.g. electricity, gas, water)",
                        "communication_payment": "Communication payment to telecom or ISP (e.g. att, verizon, comcast, etc)",
                        "mortgage_payment": "Mortgage payment",
                        "hoa_payment": "Stillwater Homeowners Association (HOA) payment",
                        "property_tax_payment": "Property tax payment",
                        "other": "Other paymennt"
                    },
                ),
            },
        )
        exclusion = response.choices["exclude"]
        exclusion_confidence = exclusion.confidence
        exclusion_probabilities = exclusion.probabilities
        exclusion_choice = exclusion.choice
        expense_category = response.choices["category"]
        category_confidence = expense_category.confidence
        category_probabilities = expense_category.probabilities
        category_choice = expense_category.choice
        if exclusion_choice != "other":
            print(f"Transaction description: {transaction_description}:")
            print(f"    exclusion_choice: {exclusion_choice}, exclusion_confidence: {exclusion_confidence}, exclusion_probabilities: {exclusion_probabilities}")        
            print(f"    category_choice: {category_choice}, category_confidence: {category_confidence}, category_probabilities: {category_probabilities}")
        
        return (exclusion_choice, category_choice)
        
if __name__ == "__main__":
    client = TypeSafeClientWrapper()
    # print(client.exclude_transaction("Pl*Paylease"))
    (exclusion_choice, category_choice) = client.exclude_transaction("Cat*Williltax Williltax Web ID: 0011160600")
    print(f"Exclusion Choice: {exclusion_choice}")
    print(f"Category Choice: {category_choice}")