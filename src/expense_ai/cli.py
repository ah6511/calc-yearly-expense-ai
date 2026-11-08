"""Command-line entrypoint for running the Expense AI pipeline."""

import argparse
import asyncio
import logging

from expense_ai.config import get_settings
from expense_ai.logging_config import configure_logging
from expense_ai.pipeline import ExpensePipeline

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="expense-ai",
        description="Extract, classify, and sum bank statement expenses.",
    )
    parser.add_argument(
        "--pdf",
        help="Process a single PDF filename or path instead of all PDFs in storage.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "claude", "bedrock"],
        help="Override LLM provider (defaults to LLM_PROVIDER in .env).",
    )
    parser.add_argument(
        "--storage",
        choices=["local", "s3"],
        help="Override storage backend (defaults to STORAGE_BACKEND in .env).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if args.provider:
        settings.llm_provider = args.provider
    if args.storage:
        settings.storage_backend = args.storage

    configure_logging(args.log_level or settings.log_level)
    pipeline = ExpensePipeline(settings=settings)

    single_pdf = args.pdf or (
        settings.single_file_mode_pdf_name if settings.single_file_mode else None
    )
    if single_pdf:
        result = asyncio.run(pipeline.process_statement_async(single_pdf))
        month = (
            result.statement_id[4:6]
            if len(result.statement_id) >= 6
            else result.statement_id
        )
        logger.info("Total withdrawals sum for month %s: %.2f", month, result.transactions_sum)
    else:
        asyncio.run(pipeline.process_all_statements())


if __name__ == "__main__":
    main()
