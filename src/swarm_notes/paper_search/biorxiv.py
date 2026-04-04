from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import requests

from swarm_notes.paper_search.base import RawPaper
from swarm_notes.paper_search.common import (
    _TRANSIENT_HTTP_STATUS_CODES,
    collapse_whitespace,
    normalise_publication_date,
)

logger = logging.getLogger(__name__)

_BIORXIV_API_BASE = "https://api.biorxiv.org/details"
_BIORXIV_QUERY_TIMEOUT_SECONDS = 30
_BIORXIV_RETRY_DELAYS_SECONDS = (2.0, 5.0)
_BIORXIV_MIN_INTERVAL_SECONDS = 1.0
_BIORXIV_PAGE_SIZE = 100  # bioRxiv API always returns up to 100 per page


class BiorxivPaperProvider:
    """Fetch papers from the bioRxiv (or medRxiv) content API.

    The bioRxiv API does not support keyword search.  Instead, this provider
    fetches all preprints posted in the last ``max_history_days`` days and
    filters them client-side by matching keywords against each paper's title
    and abstract.
    """

    def __init__(
        self,
        *,
        max_history_days: int = 30,
        server: str = "biorxiv",
        session: requests.Session | None = None,
    ) -> None:
        self._max_history_days = max(1, max_history_days)
        self._server = server.lower().strip()
        self._session = session or requests.Session()
        self._last_request_at: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, keyword: str, max_results: int) -> list[RawPaper]:
        return self.search_many([keyword], max_results)

    def search_many(self, keywords: list[str], max_results: int) -> list[RawPaper]:
        cleaned = [kw.strip() for kw in keywords if kw and kw.strip()]
        if not cleaned:
            return []

        today = datetime.now(tz=timezone.utc).date()
        cutoff = today - timedelta(days=self._max_history_days)
        start_date = cutoff.strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        lower_keywords = [kw.lower() for kw in cleaned]
        collected: dict[str, RawPaper] = {}
        cursor = 0

        while len(collected) < max_results:
            url = f"{_BIORXIV_API_BASE}/{self._server}/{start_date}/{end_date}/{cursor}/json"
            payload = self._fetch_with_retries(url)
            if payload is None:
                break

            messages = payload.get("messages", [])
            total = None
            if messages and isinstance(messages[0], Mapping):
                total = messages[0].get("count")

            collection = payload.get("collection", [])
            if not collection:
                break

            for item in collection:
                if not isinstance(item, Mapping):
                    continue

                paper = _parse_biorxiv_item(item, self._server)
                if paper is None:
                    continue

                matched = _match_keywords(paper, lower_keywords)
                if not matched:
                    continue

                paper.keywords_matched = matched
                collected.setdefault(paper.arxiv_id, paper)
                if len(collected) >= max_results:
                    break

            if len(collected) >= max_results:
                break

            cursor += _BIORXIV_PAGE_SIZE
            if total is not None and cursor >= int(total):
                break

        logger.info(
            "BiorxivPaperProvider(%s): found %d paper(s) for keywords %s within the last %d day(s)",
            self._server,
            len(collected),
            cleaned,
            self._max_history_days,
        )
        return list(collected.values())[:max_results]

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _fetch_with_retries(self, url: str) -> Mapping[str, Any] | None:
        max_attempts = len(_BIORXIV_RETRY_DELAYS_SECONDS) + 1
        for attempt in range(1, max_attempts + 1):
            try:
                self._respect_rate_limit()
                logger.debug("BiorxivPaperProvider: GET %s", url)
                response = self._session.get(url, timeout=_BIORXIV_QUERY_TIMEOUT_SECONDS)
                response.raise_for_status()
                payload = response.json()
                return payload if isinstance(payload, Mapping) else None
            except requests.RequestException as exc:
                if attempt == max_attempts or not self._is_retryable_error(exc):
                    logger.error("BiorxivPaperProvider: failed to fetch %s: %s", url, exc)
                    return None

                delay = self._get_retry_delay(exc, attempt)
                logger.warning(
                    "BiorxivPaperProvider: transient error on attempt %d/%d for %s: %s. Retrying in %.0f s.",
                    attempt,
                    max_attempts,
                    url,
                    exc,
                    delay,
                )
                time.sleep(delay)
        return None

    def _respect_rate_limit(self) -> None:
        now = time.monotonic()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            remaining = _BIORXIV_MIN_INTERVAL_SECONDS - elapsed
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
        self._last_request_at = now

    def _is_retryable_error(self, exc: requests.RequestException) -> bool:
        if isinstance(exc, requests.Timeout):
            return True
        response = exc.response  # type: ignore[union-attr]
        return response is not None and response.status_code in _TRANSIENT_HTTP_STATUS_CODES

    def _get_retry_delay(self, exc: requests.RequestException, attempt: int) -> float:
        default_delay = _BIORXIV_RETRY_DELAYS_SECONDS[attempt - 1]
        response = exc.response  # type: ignore[union-attr]
        if response is None or response.status_code != 429:
            return default_delay
        retry_after = response.headers.get("Retry-After")
        if retry_after is None:
            return default_delay
        try:
            return max(default_delay, float(retry_after))
        except ValueError:
            return default_delay


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_biorxiv_item(item: Mapping[str, Any], server: str) -> RawPaper | None:
    """Convert a single bioRxiv API collection item into a :class:`RawPaper`."""
    doi = (item.get("doi") or "").strip()
    if not doi:
        return None

    title = collapse_whitespace(item.get("title") or "Untitled")
    abstract = collapse_whitespace(item.get("abstract") or "")
    authors_raw = item.get("authors") or ""
    authors = [a.strip() for a in authors_raw.split(";") if a.strip()] if authors_raw else []
    published = normalise_publication_date(item.get("date"), None)
    category = (item.get("category") or "").strip()
    jatsxml_url = (item.get("jatsxml") or "").strip()

    url = f"https://www.biorxiv.org/content/{doi}"
    pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf"

    return RawPaper(
        arxiv_id=doi,
        title=title,
        abstract=abstract,
        authors=authors,
        published=published,
        url=url,
        primary_category=category,
        source=server,
        keywords_matched=[],
        jatsxml_url=jatsxml_url,
        pdf_url=pdf_url,
    )


def _match_keywords(paper: RawPaper, lower_keywords: list[str]) -> list[str]:
    """Return matched keywords from ``lower_keywords`` found in title or abstract."""
    haystack = (paper.title + " " + paper.abstract).lower()
    return [kw for kw in lower_keywords if kw in haystack]


def fetch_jatsxml_text(jatsxml_url: str, timeout: int = 30) -> str:
    """Fetch a JATS XML document and return the plain text body.

    Walks ``<body>``/``<sec>``/``<p>`` elements and joins their text content.
    Returns an empty string if the URL is empty or the request fails.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    if not jatsxml_url:
        return ""

    try:
        response = requests.get(jatsxml_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("BiorxivPaperProvider: failed to fetch JATS XML %s: %s", jatsxml_url, exc)
        return ""

    try:
        root = ET.fromstring(response.content)  # noqa: S314
    except ET.ParseError as exc:
        logger.warning("BiorxivPaperProvider: failed to parse JATS XML %s: %s", jatsxml_url, exc)
        return ""

    # JATS XML body text is distributed across <body>, <sec>, <p>, <title> etc.
    # Strip all tags and concatenate text nodes.
    parts: list[str] = []
    for element in root.iter():
        if element.text:
            stripped = element.text.strip()
            if stripped:
                parts.append(stripped)
        if element.tail:
            stripped = element.tail.strip()
            if stripped:
                parts.append(stripped)

    return "\n".join(parts)
