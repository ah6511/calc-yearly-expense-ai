"""AWS Lambda entrypoint wrapping the ExpensePipeline."""

import asyncio
import logging
from typing import Any

from expense_ai.config import get_settings
from expense_ai.logging_config import configure_logging
from expense_ai.pipeline import ExpensePipeline

logger = logging.getLogger(__name__)


def lambda_handler(
    event: dict[str, Any] | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    """AWS Lambda handler supporting single-file and full-year batch modes."""
    settings = get_settings()
    configure_logging(settings.log_level)
    pipeline = ExpensePipeline(settings=settings)

    if settings.single_file_mode:
        pdf_name = settings.single_file_mode_pdf_name
        result = asyncio.run(pipeline.process_statement_async(pdf_name))
        month = (
            result.statement_id[4:6]
            if len(result.statement_id) >= 6
            else result.statement_id
        )
        logger.info("Total withdrawals sum for month %s: %.2f", month, result.transactions_sum)
        return {
            "statement_id": result.statement_id,
            "total_expense": result.transactions_sum,
            "transactions_count": len(result.transactions),
        }

    total_sum, withdrawals = asyncio.run(pipeline.process_all_statements())
    return {
        "total_expense": total_sum,
        "transactions_count": len(withdrawals),
    }


if __name__ == "__main__":
    lambda_handler()
