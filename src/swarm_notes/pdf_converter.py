"""PDF-to-Markdown converter module.

Two public functions are provided:

* :func:`pdf_doi_to_markdown` — preferred for biorxiv/medrxiv/arxiv papers.
  Uses ``paperscraper.pdf.save_pdf`` which handles Cloudflare protection and
  publisher-specific fallbacks automatically.

* :func:`pdf_url_to_markdown` — generic fallback for arbitrary PDF URLs where
  no DOI is available.  Downloads with ``requests`` then converts with
  ``markitdown``.

Keeping both backends here makes it easy to swap or extend them later without
touching other modules.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_PDF_DOWNLOAD_TIMEOUT_SECONDS = 60


def pdf_doi_to_markdown(doi: str) -> str:
    """Download a PDF by *doi* using paperscraper and return its content as Markdown.

    ``paperscraper.pdf.save_pdf`` handles Cloudflare-protected servers
    (biorxiv since May 2025), publisher-specific endpoints, and automatic
    fallbacks (BioC-PMC, eLife XML, etc.).

    Returns an empty string if *doi* is empty, the download fails, or the
    conversion fails.  All errors are logged so callers do not need to handle
    exceptions.
    """
    if not doi:
        return ""

    try:
        from paperscraper.pdf import save_pdf  # noqa: PLC0415
    except ImportError as exc:
        logger.error("PdfConverter: paperscraper is not installed: %s", exc)
        return ""

    logger.info("PdfConverter: downloading PDF for DOI %s via paperscraper", doi)
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            # save_pdf appends ".pdf" to the base path automatically.
            tmp_base = os.path.join(tmp_dir, "paper")
            success = save_pdf({"doi": doi}, filepath=tmp_base)
            if not success:
                logger.warning("PdfConverter: paperscraper could not download PDF for DOI %s", doi)
                return ""

            pdf_path = tmp_base + ".pdf"
            if not os.path.exists(pdf_path):
                logger.warning("PdfConverter: expected PDF not found at %s for DOI %s", pdf_path, doi)
                return ""

            # Capture the result inside the `with` block so that a cleanup
            # error on __exit__ (e.g. [Errno 66] on macOS) cannot discard it.
            text = _convert_pdf_file_to_markdown(pdf_path, label=doi)
        return text
    except Exception as exc:
        logger.error("PdfConverter: failed to obtain PDF for DOI %s: %s", doi, exc)
        return ""


def pdf_url_to_markdown(url: str, timeout: int = _PDF_DOWNLOAD_TIMEOUT_SECONDS) -> str:
    """Download a PDF from *url* and return its content as Markdown.

    Returns an empty string if the URL is empty, the download fails, or the
    conversion fails.  All errors are logged at ERROR level so callers do not
    have to handle exceptions.
    """
    if not url:
        return ""

    logger.info("PdfConverter: downloading PDF from %s", url)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("PdfConverter: failed to download PDF from %s: %s", url, exc)
        return ""

    logger.info("PdfConverter: converting %d bytes to Markdown", len(resp.content))
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        try:
            return _convert_pdf_file_to_markdown(tmp_path, label=url)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    except Exception as exc:
        logger.error("PdfConverter: failed to convert PDF from %s: %s", url, exc)
        return ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convert_pdf_file_to_markdown(pdf_path: str, *, label: str = "") -> str:
    """Convert an on-disk PDF file to Markdown using markitdown.

    *label* is used only for log messages (e.g. the source URL or DOI).
    Returns an empty string on failure.
    """
    try:
        from markitdown import MarkItDown  # noqa: PLC0415

        md = MarkItDown()
        result = md.convert(pdf_path)
        text = result.text_content or ""
        logger.info("PdfConverter: converted %s to %d chars of Markdown", label, len(text))
        return text
    except Exception as exc:
        logger.error("PdfConverter: markitdown conversion failed for %s: %s", label, exc)
        return ""
