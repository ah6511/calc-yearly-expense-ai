"""Storage backends for reading statement PDFs and saving extracted results."""

from expense_ai.config import Settings, get_settings
from expense_ai.storage.base import StorageBackend
from expense_ai.storage.local import LocalStorage
from expense_ai.storage.s3 import S3Storage


def create_storage(settings: Settings | None = None) -> StorageBackend:
    """Create the configured storage backend ('local' or 's3')."""
    cfg = settings or get_settings()
    if cfg.storage_backend == "s3":
        return S3Storage(settings=cfg)
    return LocalStorage(settings=cfg)


__all__ = ["StorageBackend", "LocalStorage", "S3Storage", "create_storage"]
