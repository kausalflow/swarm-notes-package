"""Unit tests for the bioRxiv paper provider."""

from unittest.mock import MagicMock, patch

import requests

from swarm_notes.paper_search.biorxiv import BiorxivPaperProvider, fetch_jatsxml_text


def _build_collection_item(*, doi: str, title: str, abstract: str, jatsxml: str = "") -> dict[str, str]:
    return {
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "authors": "Jane Doe; John Smith",
        "date": "2026-03-30",
        "category": "neuroscience",
        "jatsxml": jatsxml,
    }


def test_biorxiv_provider_filters_by_keyword_and_sets_jatsxml() -> None:
    payload = {
        "messages": [{"count": 2}],
        "collection": [
            _build_collection_item(
                doi="10.1101/2020.04.04.024703",
                title="Neural circuit mechanisms for steering control",
                abstract="Heading control in Drosophila.",
                jatsxml="https://www.biorxiv.org/content/early/2025/03/30/2020.04.04.024703.source.xml",
            ),
            _build_collection_item(
                doi="10.1101/2020.04.04.999999",
                title="Unrelated paper",
                abstract="Protein transport and metabolism.",
            ),
        ],
    }

    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = payload
    mock_session.get.return_value = mock_response

    provider = BiorxivPaperProvider(max_history_days=30, server="biorxiv", session=mock_session)

    papers = provider.search_many(["neural", "drosophila"], max_results=5)

    assert len(papers) == 1
    paper = papers[0]
    assert paper.source == "biorxiv"
    assert paper.arxiv_id == "10.1101/2020.04.04.024703"
    assert paper.jatsxml_url.endswith("024703.source.xml")
    assert paper.keywords_matched == ["neural", "drosophila"]


def test_biorxiv_provider_respects_max_results_across_pages() -> None:
    first_page = {
        "messages": [{"count": 200}],
        "collection": [
            _build_collection_item(
                doi="10.1101/2020.01.01.111111",
                title="Graph neural signal model",
                abstract="A graph neural method.",
            )
        ],
    }
    second_page = {
        "messages": [{"count": 200}],
        "collection": [
            _build_collection_item(
                doi="10.1101/2020.01.01.222222",
                title="Another graph neural model",
                abstract="Another graph neural method.",
            )
        ],
    }

    mock_session = MagicMock()
    first_response = MagicMock()
    first_response.raise_for_status.return_value = None
    first_response.json.return_value = first_page

    second_response = MagicMock()
    second_response.raise_for_status.return_value = None
    second_response.json.return_value = second_page

    mock_session.get.side_effect = [first_response, second_response]

    provider = BiorxivPaperProvider(max_history_days=30, server="biorxiv", session=mock_session)
    papers = provider.search_many(["graph neural"], max_results=2)

    assert len(papers) == 2
    assert papers[0].arxiv_id == "10.1101/2020.01.01.111111"
    assert papers[1].arxiv_id == "10.1101/2020.01.01.222222"
    assert mock_session.get.call_count == 2


def test_fetch_jatsxml_text_extracts_plain_text() -> None:
    xml = b"""<?xml version='1.0' encoding='UTF-8'?>
    <article>
      <front><article-meta><title-group><article-title>Sample</article-title></title-group></article-meta></front>
      <body>
        <sec>
          <title>Introduction</title>
          <p>First paragraph.</p>
          <p>Second paragraph.</p>
        </sec>
      </body>
    </article>
    """

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.content = xml

    with patch("swarm_notes.paper_search.biorxiv.requests.get", return_value=mock_response):
        text = fetch_jatsxml_text("https://www.biorxiv.org/content/sample.source.xml")

    assert "Introduction" in text
    assert "First paragraph." in text
    assert "Second paragraph." in text


def test_fetch_jatsxml_text_returns_empty_on_request_error() -> None:
    with patch(
        "swarm_notes.paper_search.biorxiv.requests.get",
        side_effect=requests.RequestException("boom"),
    ):
        text = fetch_jatsxml_text("https://www.biorxiv.org/content/missing.source.xml")

    assert text == ""


def test_biorxiv_provider_filters_by_category() -> None:
    """Papers whose biorxiv category is not in the allowed list must be excluded."""
    payload = {
        "messages": [{"count": 2}],
        "collection": [
            _build_collection_item(
                doi="10.1101/2026.01.01.neuro",
                title="TMS modulation of motor cortex",
                abstract="We applied TMS to the primary motor cortex.",
            )
            | {"category": "neuroscience"},
            _build_collection_item(
                doi="10.1101/2026.01.01.biophys",
                title="Structural plasticity of peptides",
                abstract="Their structural plasticity ranging from disordered to folded states.",
            )
            | {"category": "biophysics"},
        ],
    }

    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = payload
    mock_session.get.return_value = mock_response

    provider = BiorxivPaperProvider(
        max_history_days=30,
        server="biorxiv",
        categories=["neuroscience"],
        session=mock_session,
    )

    papers = provider.search_many(["tms", "structural plasticity"], max_results=5)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "10.1101/2026.01.01.neuro"
