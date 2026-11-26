# calc-yearly-expense-ai

AI-powered pipeline that extracts withdrawal transactions from bank statement PDFs using structured LLM outputs (`instructor`), filters and categorizes transactions with TypeSafe AI (`typesafe-sdk`), and aggregates monthly and yearly expenses.

## Project Layout

```text
calc-yearly-expense-ai/
├── pyproject.toml                  # Dependencies & tool configuration (uv, ruff, mypy, pytest)
├── .env.example                    # Example environment variable configuration
├── src/expense_ai/
│   ├── config.py                   # Centralized Pydantic Settings (loads .env)
│   ├── logging_config.py           # Structured logging configuration
│   ├── models.py                   # Domain models (Transaction, CategorizedTransaction, enums)
│   ├── pdf.py                      # PDF text extraction
│   ├── prompts/
│   │   └── extract_transactions.md # Versioned extraction system prompt
│   ├── llm/
│   │   ├── base.py                 # LLMClient protocol
│   │   └── factory.py              # Instructor client for OpenAI, Claude, Bedrock, and Gemini
│   ├── classification/
│   │   ├── criteria.py             # Declarative exclusion & category criteria
│   │   └── typesafe_classifier.py  # Sync & bounded-concurrency async TypeSafe classifier
│   ├── storage/
│   │   ├── base.py                 # StorageBackend protocol
│   │   ├── local.py                # Local filesystem storage
│   │   └── s3.py                   # Amazon S3 storage
│   ├── pipeline.py                 # End-to-end ExpensePipeline with dependency injection
│   └── cli.py                      # CLI entrypoint (`expense-ai` or `python -m expense_ai`)
├── handlers/
│   └── lambda_handler.py           # AWS Lambda handler (`handlers.lambda_handler.lambda_handler`)
└── tests/                          # Offline unit & pipeline integration tests
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management:

```bash
uv sync
cp .env.example .env
# Edit .env with your API keys
```

## PDF Storage Locations

- **Running locally (`STORAGE_BACKEND=local`)**: Statement PDFs are read from the local `pdf/` folder.
- **Running in AWS (`STORAGE_BACKEND=s3`)**: Statement PDFs are read from the S3 bucket specified by the `BUCKET_NAME` environment variable under the `pdfs` prefix (e.g., `s3://<BUCKET_NAME>/pdfs/`).

## Usage

Run across all statement PDFs in `pdf/`:

```bash
uv run expense-ai
```

Run on a single statement PDF:

```bash
uv run expense-ai --pdf 20251125-statements-8506-.pdf
```

Override provider or storage backend from the CLI:

```bash
uv run expense-ai --provider claude --storage local --log-level INFO
```

### AWS Lambda

Configure the Lambda handler to:

```text
handlers.lambda_handler.lambda_handler
```

## Development & Verification

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

