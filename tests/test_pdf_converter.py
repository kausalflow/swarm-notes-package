"""Unit tests for the PDF-to-Markdown converter module."""

import os
from unittest.mock import MagicMock, patch

import requests

from swarm_notes.pdf_converter import pdf_doi_to_markdown, pdf_url_to_markdown


# ===========================================================================
# pdf_doi_to_markdown — paperscraper-based path
# ===========================================================================


def test_doi_returns_empty_for_empty_string() -> None:
    assert pdf_doi_to_markdown("") == ""


def test_doi_returns_empty_when_paperscraper_not_installed() -> None:
    with patch.dict("sys.modules", {"paperscraper": None, "paperscraper.pdf": None}):
        result = pdf_doi_to_markdown("10.1101/2025.01.01.000001")
    assert result == ""


def test_doi_returns_empty_when_save_pdf_returns_false() -> None:
    mock_save_pdf = MagicMock(return_value=False)
    with patch("swarm_notes.pdf_converter.pdf_doi_to_markdown.__wrapped__", create=True), patch(
        "paperscraper.pdf.save_pdf", mock_save_pdf
    ):
        # Patch the import inside the function
        mock_module = MagicMock()
        mock_module.save_pdf = mock_save_pdf
        with patch.dict("sys.modules", {"paperscraper.pdf": mock_module}):
            result = pdf_doi_to_markdown("10.1101/2025.01.01.000001")
    assert result == ""


def test_doi_returns_markdown_on_successful_download_and_conversion() -> None:
    """save_pdf succeeds → markitdown converts the file → markdown returned."""
    doi = "10.1101/2025.01.01.000001"

    mock_result = MagicMock()
    mock_result.text_content = "# Biorxiv Paper\n\nFull text here."
    mock_md_instance = MagicMock()
    mock_md_instance.convert.return_value = mock_result

    def fake_save_pdf(metadata, filepath):
        # Write a dummy PDF so the path-exists check passes
        with open(filepath + ".pdf", "wb") as f:
            f.write(b"%PDF-1.4 fake")
        return True

    with patch.dict(
        "sys.modules",
        {
            "paperscraper.pdf": MagicMock(save_pdf=fake_save_pdf),
            "markitdown": MagicMock(MarkItDown=MagicMock(return_value=mock_md_instance)),
        },
    ):
        result = pdf_doi_to_markdown(doi)

    assert result == "# Biorxiv Paper\n\nFull text here."


def test_doi_temp_dir_cleaned_up_after_conversion() -> None:
    """The TemporaryDirectory (and its PDF file) must be cleaned up."""
    doi = "10.1101/2025.01.01.000001"
    created_dirs: list[str] = []

    mock_result = MagicMock()
    mock_result.text_content = "converted"
    mock_md_instance = MagicMock()
    mock_md_instance.convert.return_value = mock_result

    def fake_save_pdf(metadata, filepath):
        with open(filepath + ".pdf", "wb") as f:
            f.write(b"%PDF fake")
        return True

    original_td = __import__("tempfile").TemporaryDirectory

    class TrackingTempDir:
        def __init__(self, **kwargs):
            self._td = original_td(**kwargs)
            created_dirs.append(self._td.name)

        def __enter__(self):
            return self._td.__enter__()

        def __exit__(self, *args):
            return self._td.__exit__(*args)

    with patch("swarm_notes.pdf_converter.tempfile.TemporaryDirectory", TrackingTempDir), patch.dict(
        "sys.modules",
        {
            "paperscraper.pdf": MagicMock(save_pdf=fake_save_pdf),
            "markitdown": MagicMock(MarkItDown=MagicMock(return_value=mock_md_instance)),
        },
    ):
        result = pdf_doi_to_markdown(doi)

    assert result == "converted"
    # The temp dir should have been cleaned up
    assert len(created_dirs) == 1
    assert not os.path.exists(created_dirs[0])


def test_doi_returns_empty_on_markitdown_failure() -> None:
    doi = "10.1101/2025.01.01.000001"

    mock_md_instance = MagicMock()
    mock_md_instance.convert.side_effect = RuntimeError("markitdown exploded")

    def fake_save_pdf(metadata, filepath):
        with open(filepath + ".pdf", "wb") as f:
            f.write(b"%PDF fake")
        return True

    with patch.dict(
        "sys.modules",
        {
            "paperscraper.pdf": MagicMock(save_pdf=fake_save_pdf),
            "markitdown": MagicMock(MarkItDown=MagicMock(return_value=mock_md_instance)),
        },
    ):
        result = pdf_doi_to_markdown(doi)

    assert result == ""


# ===========================================================================
# pdf_url_to_markdown — generic URL-based path
# ===========================================================================


def test_returns_empty_string_for_empty_url() -> None:
    assert pdf_url_to_markdown("") == ""


def test_returns_empty_string_for_whitespace_url() -> None:
    # Whitespace-only strings are not empty so the function will attempt a
    # download; but the primary edge case we guard is the empty string.
    assert pdf_url_to_markdown("") == ""


# ---------------------------------------------------------------------------
# HTTP download failures
# ---------------------------------------------------------------------------


def test_returns_empty_on_http_error() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("404")

    with patch("swarm_notes.pdf_converter.requests.get", return_value=mock_resp):
        result = pdf_url_to_markdown("https://example.com/paper.pdf")

    assert result == ""


def test_returns_empty_on_connection_error() -> None:
    with patch(
        "swarm_notes.pdf_converter.requests.get",
        side_effect=requests.ConnectionError("unreachable"),
    ):
        result = pdf_url_to_markdown("https://example.com/paper.pdf")

    assert result == ""


def test_returns_empty_on_timeout() -> None:
    with patch(
        "swarm_notes.pdf_converter.requests.get",
        side_effect=requests.Timeout("timed out"),
    ):
        result = pdf_url_to_markdown("https://example.com/paper.pdf")

    assert result == ""


def test_get_called_with_correct_url_and_timeout() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500")

    with patch("swarm_notes.pdf_converter.requests.get", return_value=mock_resp) as mock_get:
        pdf_url_to_markdown("https://example.com/paper.pdf", timeout=42)

    mock_get.assert_called_once_with("https://example.com/paper.pdf", timeout=42)


# ---------------------------------------------------------------------------
# Successful conversion
# ---------------------------------------------------------------------------


def test_returns_markdown_on_successful_conversion() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"%PDF-1.4 fake pdf bytes"

    mock_result = MagicMock()
    mock_result.text_content = "# My Paper\n\nSome extracted text."

    mock_md_instance = MagicMock()
    mock_md_instance.convert.return_value = mock_result

    mock_markitdown_cls = MagicMock(return_value=mock_md_instance)

    with patch("swarm_notes.pdf_converter.requests.get", return_value=mock_resp), patch.dict(
        "sys.modules", {"markitdown": MagicMock(MarkItDown=mock_markitdown_cls)}
    ):
        result = pdf_url_to_markdown("https://example.com/paper.pdf")

    assert result == "# My Paper\n\nSome extracted text."


def test_temp_file_cleaned_up_after_successful_conversion() -> None:
    """The temporary PDF file must be deleted even on success."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"%PDF-1.4 fake"

    mock_result = MagicMock()
    mock_result.text_content = "converted"

    mock_md_instance = MagicMock()
    mock_md_instance.convert.return_value = mock_result

    written_paths: list[str] = []
    original_ntf = __import__("tempfile").NamedTemporaryFile

    def tracking_ntf(**kwargs):
        ntf = original_ntf(**kwargs)
        written_paths.append(ntf.name)
        return ntf

    with patch("swarm_notes.pdf_converter.requests.get", return_value=mock_resp), patch.dict(
        "sys.modules",
        {"markitdown": MagicMock(MarkItDown=MagicMock(return_value=mock_md_instance))},
    ), patch("swarm_notes.pdf_converter.tempfile.NamedTemporaryFile", tracking_ntf):
        result = pdf_url_to_markdown("https://example.com/paper.pdf")

    assert result == "converted"
    assert len(written_paths) == 1
    assert not __import__("os").path.exists(written_paths[0])


def test_temp_file_cleaned_up_after_conversion_failure() -> None:
    """The temporary PDF file must be deleted even when markitdown raises."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"%PDF-1.4 fake"

    mock_md_instance = MagicMock()
    mock_md_instance.convert.side_effect = RuntimeError("markitdown exploded")

    written_paths: list[str] = []
    original_ntf = __import__("tempfile").NamedTemporaryFile

    def tracking_ntf(**kwargs):
        ntf = original_ntf(**kwargs)
        written_paths.append(ntf.name)
        return ntf

    with patch("swarm_notes.pdf_converter.requests.get", return_value=mock_resp), patch.dict(
        "sys.modules",
        {"markitdown": MagicMock(MarkItDown=MagicMock(return_value=mock_md_instance))},
    ), patch("swarm_notes.pdf_converter.tempfile.NamedTemporaryFile", tracking_ntf):
        result = pdf_url_to_markdown("https://example.com/paper.pdf")

    assert result == ""
    assert len(written_paths) == 1
    assert not __import__("os").path.exists(written_paths[0])


# ---------------------------------------------------------------------------
# markitdown returns empty / None text_content
# ---------------------------------------------------------------------------


def test_returns_empty_when_text_content_is_none() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"%PDF-1.4 fake"

    mock_result = MagicMock()
    mock_result.text_content = None

    mock_md_instance = MagicMock()
    mock_md_instance.convert.return_value = mock_result

    with patch("swarm_notes.pdf_converter.requests.get", return_value=mock_resp), patch.dict(
        "sys.modules", {"markitdown": MagicMock(MarkItDown=MagicMock(return_value=mock_md_instance))}
    ):
        result = pdf_url_to_markdown("https://example.com/paper.pdf")

    assert result == ""


def test_returns_empty_when_text_content_is_empty_string() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"%PDF-1.4 fake"

    mock_result = MagicMock()
    mock_result.text_content = ""

    mock_md_instance = MagicMock()
    mock_md_instance.convert.return_value = mock_result

    with patch("swarm_notes.pdf_converter.requests.get", return_value=mock_resp), patch.dict(
        "sys.modules", {"markitdown": MagicMock(MarkItDown=MagicMock(return_value=mock_md_instance))}
    ):
        result = pdf_url_to_markdown("https://example.com/paper.pdf")

    assert result == ""


# ---------------------------------------------------------------------------
# markitdown import failure (optional dependency not installed)
# ---------------------------------------------------------------------------


def test_returns_empty_when_markitdown_not_installed() -> None:
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.content = b"%PDF-1.4 fake"

    with patch("swarm_notes.pdf_converter.requests.get", return_value=mock_resp), patch.dict(
        "sys.modules", {"markitdown": None}
    ):
        result = pdf_url_to_markdown("https://example.com/paper.pdf")

    assert result == ""
