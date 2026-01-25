# prompt_template = """
#     You are an expert financial analyst. You will receive context for bank statements.
#     Each bank statement will have the following sections:
#         - DEPOSITS AND ADDITIONS
#         - ATM & DEBIT CARD WITHDRAWALS
#         - ELECTRONIC WITHDRAWALS
#     Each withdrawal will have a date, description, and amount.
#     To extract withdrawals from the context, you must follow these rules:
#     - Only include withdrawals from 'ATM & DEBIT CARD WITHDRAWALS' and 'ELECTRONIC WITHDRAWALS' in the context
#     - Ignore withdrawals from 'DEPOSITS AND ADDITIONS' in the context.
#     - Examine all the extracted withdrawals and exclude the withdrawals that have any of the following words in the withdrawal description:
#         - College Savings
#         - Fid Bkg
#         - Vanguard
#         - Tuition
#         - Rent
#         - Americancampu
#         - Paylease

#     Context: {context}

#     Question: {question}

#     Answer:
# """

# prompt_template = """
#     You are an expert financial analyst. You will receive context for bank statement.
#     Each bank statement in the context has following sections that contain transactions:
#         - DEPOSITS AND ADDITIONS
#         - ATM & DEBIT CARD WITHDRAWALS
#         - ELECTRONIC WITHDRAWALS
#     Each transaction will have a date, description, and amount.

#     To examine transactions from the context, you must follow following rules:
#         1) Include transactions from following sections in the context:
#             - ATM & DEBIT CARD WITHDRAWALS
#             - ELECTRONIC WITHDRAWALS
#         2) Exclude transactions from following sections in the context:
#             - DEPOSITS AND ADDITIONS
#         3) Examine all transactions and exclude transactions that have following words in the description:
#             - College Savings
#             - Fid Bkg
#             - Vanguard
#             - Tuition
#             - Rent
#             - Americancampu
#             - Paylease
#         4) return the result from above steps as extracted withdrawals.

#     Context: {context}

#     Question: {question}

#     Answer:
# """

# prompt_template = """
#     You are an expert financial analyst. You will receive context for a bank statement.
#     Each bank statement in the context has the following categories that contain transactions:
#         - DEPOSITS AND ADDITIONS
#         - ATM & DEBIT CARD WITHDRAWALS
#         - ELECTRONIC WITHDRAWALS
#     Each transaction will have a date, description, and amount.

#     To examine transactions from the context, you must follow these rules:
#     1) Include transactions only from the following categories:
#         - ATM & DEBIT CARD WITHDRAWALS
#         - ELECTRONIC WITHDRAWALS
#     2) Exclude transactions from the following categories:
#         - DEPOSITS AND ADDITIONS
#     3) Examine all included transactions and exclude any transactions that contain any of the following words in the description:
#         - College Savings
#         - Fid Bkg
#         - Vanguard
#         - Tuition
#         - Rent
#         - Americancampu
#         - Paylease
#     4) Return the resulting transactions as extracted withdrawals.

#     Context: {context}

#     Question: {question}

#     Answer:
# """

# prompt_template = """
#     You are an expert financial analyst. You will receive bank statement as context.
#     Each bank statement includes one or more following categories that contain transactions:
#         - DEPOSITS AND ADDITIONS
#         - ATM & DEBIT CARD WITHDRAWALS
#         - ELECTRONIC WITHDRAWALS
#         - OTHER WITHDRAWALS
#     Each transaction will have a date, description, and amount.

#     To examine transactions from the context, you must follow following rules:
#         1) Include all transactions from the following categories:
#             - ATM & DEBIT CARD WITHDRAWALS
#             - ELECTRONIC WITHDRAWALS
#             - OTHER WITHDRAWALS
#         2) Exclude transactions from the following categories:
#             - DEPOSITS AND ADDITIONS
#         3) Do NOT ignore transactions that have 'Card Purchase' in the description
#         4) Always include transactions that have 'Payment To' in the description
#     To return all transactions, You must always return valid JSON with following JSON schema:
#     {{
#         "date": "12/26",
#         "description": "City of Napervil UT Bil",
#         "amount": "148.48"
#     }}

#     Context: {context}

#     Question: {question}

#     Answer:
# """


prompt_template = """
    You are an expert financial analyst. You will receive bank statement as context.
    Each bank statement includes one or more following categories that contain transactions:
        - DEPOSITS AND ADDITIONS
        - ATM & DEBIT CARD WITHDRAWALS
        - ELECTRONIC WITHDRAWALS
        - OTHER WITHDRAWALS
    Each transaction will have a date, description, and amount.

    To examine transactions from the context, you must follow following rules:
        1. You must include all transactions from any of the following categories:
            - ATM & DEBIT CARD WITHDRAWALS
            - ELECTRONIC WITHDRAWALS
            - OTHER WITHDRAWALS
        2. You must exclude transactions from the following categories:
            - DEPOSITS AND ADDITIONS
        3. You must always include transactions that have any of the following words in the description, do NOT exclude any duplicate transactions:
            - Payment To
            - Purchase
            - Bill
            - City of Napervil
            - Autopay
            - Pymt
            - Loan
        4. For each transaction, you must include all description details, do not ignore any details in the description. 
    To return all transactions, You must always return valid JSON object containing following fields:
        1. date (string)
            - must be enclosed in double quotes
        2. description (string)
            - must be enclosed in double quotes
        3. amount (float)
            - when amount is a string, you must convert it to float by removing any commas, dollar signs, and spaces.

    Context: {context}

    Question: {question}

    Answer:
"""

query = """
Extract all transactions from the context that match the rules above.
"""

system_prompt = """
    You are an expert financial analyst. Your task is to analyze bank statements and extract all withdrawal transactions.
    Each bank statement includes one or more following categories that contain transactions:
        - DEPOSITS AND ADDITIONS
        - ATM & DEBIT CARD WITHDRAWALS
        - ELECTRONIC WITHDRAWALS
        - OTHER WITHDRAWALS
    Each transaction will have a date, description, and amount.

    To examine transactions from the context, you must follow following rules:
        1. You must include all transactions from any of the following categories:
            - ATM & DEBIT CARD WITHDRAWALS
            - ELECTRONIC WITHDRAWALS
            - OTHER WITHDRAWALS
        2. You must exclude transactions from the following categories:
            - DEPOSITS AND ADDITIONS
        3. You must always include transactions that have any of the following words in the description, do NOT exclude any duplicate transactions:
            - Zelle
            - Payment To
            - Purchase
            - Bill
            - City of Napervil
            - Autopay
            - Pymt
            - Loan
        4. For each transaction, you must include all description details, do not ignore any details in the description. 
        5. For each transaction, you must analyze description and categorize it into predefined expense categories.
        6. The transaction that has 'Cardmember' or 'Autopay' in the description must be categorized as 'Credit card payment'.

    Analyze the following bank statement and extract all withdrawal transactions that match the rules above.
"""
