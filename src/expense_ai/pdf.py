"""PDF text extraction utilities."""

import logging
from pathlib import Path
from typing import Any

from pypdf import PageObject, PdfReader

logging.getLogger("pypdf").setLevel(logging.CRITICAL)


def _extract_page_text_in_reading_order(
    page: PageObject, line_tolerance: float = 2.0
) -> str:
    """Extract text fragments from a PDF page sorted top-to-bottom, left-to-right."""
    items: list[tuple[float, float, str]] = []

    def visitor(
        text: str,
        _cm: list[float],
        tm: list[float],
        _font_dict: dict[str, Any] | None,
        _font_size: float,
    ) -> None:
        if text and text.strip():
            x, y = float(tm[4]), float(tm[5])
            items.append((round(-y, 1), round(x, 1), text.strip()))

    page.extract_text(visitor_text=visitor)
    if not items:
        return page.extract_text() or ""

    items.sort(key=lambda item: (item[0], item[1]))

    lines: list[str] = []
    current_y: float | None = None
    current_line: list[str] = []

    for y, _x, text in items:
        if current_y is None or abs(y - current_y) > line_tolerance:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [text]
            current_y = y
        else:
            current_line.append(text)

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract text from all pages of a PDF file separated by page break markers."""
    reader = PdfReader(str(pdf_path))
    pages = [_extract_page_text_in_reading_order(page) for page in reader.pages]
    # pages = [page.extract_text().strip() for page in reader.pages]
    return "\r\n--- Page Break ---\r\n".join(pages)

