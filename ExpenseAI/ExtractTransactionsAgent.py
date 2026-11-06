import os
import json
from typing import List
from ExpenseAI.ExtractTransactionsByPdf import ExtractTransactionsByPdf
from Utils.utils import save_to_s3, save_to_local_dir
from ExpenseAI.TypesafeWrapper import TypeSafeClientWrapper
from ExpenseAI.Models import Transaction, CategorizedTransaction

### Agent
class ExtractTransactionsAgent:
    def __init__(self):
        self.extractTransactionsByPdf = ExtractTransactionsByPdf()  
        self.typesafe_client = TypeSafeClientWrapper()

    def save_transactions(self, pdf_path: str, transactions: List[CategorizedTransaction]):
        arr = pdf_path.split("/")
        name = arr[len(arr) - 1].split("-")[0]
        path = "/".join(arr[0:len(arr) - 2]) + "/transactions/"
        json_file = path + name + ".json"
        if os.getenv("LOCAL_MODE", "false").lower() == "true":
            with open(json_file, "w") as f:
                json.dump(transactions, f, indent=4)
        else:
            save_to_local_dir(json.dumps(transactions), os.path.join(os.getcwd(), "transactions"), name + ".json") if os.getenv("USE_LOCAL", "false").lower() == "true" else save_to_s3(json.dumps(transactions), "transactions", name + ".json")

    def save_year_total(self, pdf_path: str, transactions_sum: float):
        arr = pdf_path.split("/")
        name = arr[len(arr) - 1].split("-")[0]
        path = "/".join(arr[0:len(arr) - 2]) + "/year_total/"
        json_file = path + name + ".json"
        if os.getenv("LOCAL_MODE", "false").lower() == "true":
            with open(json_file, "w") as f:
                json.dump({"year_total": transactions_sum}, f, indent=4)
        else:
            save_to_local_dir(json.dumps({"year_total": transactions_sum}), os.path.join(os.getcwd(), "year_total"), name + ".json") if os.getenv("USE_LOCAL", "false").lower() == "true" else save_to_s3(json.dumps({"year_total": transactions_sum}), "year_total", name + ".json")

    def extract_transactions(self, pdf_path: str) -> List[Transaction]:
        transactions = self.extractTransactionsByPdf.retrieve_transactions(pdf_path)
        transactions = [json.loads(transaction.model_dump_json(indent=2)) for transaction in transactions]

        return transactions

    def exclude_transactions(self, transactions: List[Transaction]) -> List[CategorizedTransaction]:
        filtered_transactions: List[CategorizedTransaction] = []
        for transaction in transactions:
            exclude, category = self.typesafe_client.exclude_transaction(transaction["description"])
            if exclude == "other":
                transaction["category"] = category
                filtered_transactions.append(transaction)
            else:
                print(f"Excluding transaction: {transaction}, its category is {category}")

        return filtered_transactions

    def sum_transactions(self, transactions: List[CategorizedTransaction]):
        transactions_sum = sum(transaction["amount"] for transaction in transactions)
        return transactions_sum

    def run(self, pdf_path: str):
        transactions: List[Transaction] = self.extract_transactions(pdf_path)
        categorized_transactions: List[CategorizedTransaction] = self.exclude_transactions(transactions)
        transactions_sum: float = self.sum_transactions(categorized_transactions)
        self.save_transactions(pdf_path, categorized_transactions)
        self.save_year_total(pdf_path, transactions_sum)

        return {"transactions": categorized_transactions, "transactions_sum": transactions_sum}
