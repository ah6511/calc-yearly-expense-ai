"""Amazon S3 storage backend with a shared boto3 client and explicit error propagation."""

import json
import logging
from pathlib import Path
from typing import Any

import boto3

from expense_ai.config import Settings, get_settings
from expense_ai.models import CategorizedTransaction
from expense_ai.storage.base import StorageBackend
from expense_ai.storage.local import serialize_transactions_csv

logger = logging.getLogger(__name__)


class S3Storage(StorageBackend):
    """Reads PDFs from S3 and writes JSON/CSV results back to S3."""

    def __init__(
        self,
        settings: Settings | None = None,
        s3_client: Any | None = None,
        tmp_dir: Path = Path("/tmp"),
    ) -> None:
        self.settings = settings or get_settings()
        if not self.settings.bucket_name:
            raise ValueError("BUCKET_NAME must be configured when using S3Storage.")
        self.bucket = self.settings.bucket_name
        self.tmp_dir = tmp_dir
        self.client = s3_client or self._create_client()

    def _create_client(self) -> Any:
        kwargs: dict[str, Any] = {
            "service_name": "s3",
            "region_name": self.settings.aws_region,
        }
        if self.settings.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.settings.aws_access_key_id
        if self.settings.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = (
                self.settings.aws_secret_access_key.get_secret_value()
            )
        if self.settings.aws_session_token:
            kwargs["aws_session_token"] = self.settings.aws_session_token.get_secret_value()
        return boto3.client(**kwargs)

    def list_pdfs(self, prefix: str = "pdfs") -> list[str]:
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        keys = [
            obj["Key"]
            for obj in response.get("Contents", [])
            if obj["Key"].lower().endswith(".pdf")
        ]
        keys.sort()
        return keys

    def resolve_local_pdf(self, pdf_ref: str) -> Path:
        local_candidate = Path(pdf_ref)
        if local_candidate.exists():
            return local_candidate
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        dest = self.tmp_dir / Path(pdf_ref).name
        logger.info("Downloading s3://%s/%s to %s", self.bucket, pdf_ref, dest)
        self.client.download_file(self.bucket, pdf_ref, str(dest))
        return dest

    def _put_text(self, key: str, body: str) -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body.encode("utf-8"))
        return f"s3://{self.bucket}/{key}"

    def save_transactions(
        self, statement_id: str, transactions: list[CategorizedTransaction]
    ) -> str:
        payload = json.dumps([tx.model_dump() for tx in transactions], indent=4)
        return self._put_text(f"transactions/{statement_id}.json", payload)

    def save_year_total(self, statement_id: str, transactions_sum: float) -> str:
        payload = json.dumps({"year_total": transactions_sum}, indent=4)
        return self._put_text(f"year_total/{statement_id}.json", payload)

    def save_total_withdrawals_csv(self, transactions: list[CategorizedTransaction]) -> str:
        csv_text = serialize_transactions_csv(transactions)
        return self._put_text("transactions/total_withdrawals.csv", csv_text)
