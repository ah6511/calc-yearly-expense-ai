import os 
from typing import List
from ExpenseAI.Prompts import system_prompt
from ExpenseAI.Models import Transactions
from ExpenseAI.llm_factory import LLMFactory
from Utils.llm_provider import LLMProvider
from ExpenseAI.Models import Transaction
from Utils.utils import extract_text_from_pdf, download_pdf_from_s3

class ExtractTransactionsByPdf:
    def __init__(self):
        llm_provider = LLMProvider(os.environ["LLM_PROVIDER"].lower())
        match llm_provider:
            case LLMProvider.OPENAI:
                self.llm = LLMFactory("openai")
            case LLMProvider.CLAUDE:
                self.llm = LLMFactory("claude")
            case LLMProvider.BEDROCK:
                self.llm = LLMFactory("bedrock")
            case _:
                self.llm = LLMFactory("openai")

    def retrieve_transactions(self, pdf_path: str) -> List[Transaction]:
        transactions = []
        try:
            doc = extract_text_from_pdf(pdf_path)
            completion = self.llm.create_completion(
                response_model=Transactions,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": doc
                    }
                ],
            )
            assert isinstance(completion, Transactions)
            transactions = completion.transactions
            transactions.sort(key=lambda transaction: transaction.date)
        except Exception as e:
            print(f"retrieve_transactions with pdf_path = {pdf_path} got error: {e}")
            raise e

        return transactions

    def retrieve_transactions_from_s3(self, key: str) -> List[Transaction]:
        pdf_path = "/tmp/" + key.split("/")[-1]
        download_pdf_from_s3(key, pdf_path)
        return self.retrieve_transactions(pdf_path)