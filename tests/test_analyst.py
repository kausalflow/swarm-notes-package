"""Tests for analyst.py."""

from unittest.mock import MagicMock, patch

from swarm_notes.analyst import analyse
from swarm_notes.router import load_skills, get_general_skill
from swarm_notes.watcher import RawPaper


from pydantic import BaseModel

class MockOutput(BaseModel):
    """Mock structure for PaperAnalysis output."""
    arxiv_id: str = ""
    url: str = ""
    published: str = ""
    title: str = "Test Title"
    summary: str = "Test summary"
    key_contributions: list[str] = []
    domain: str = "nlp"
    authors: list[str] = []
    concepts: list[str] = []
    tags: list[str] = []
    source: str = ""


@patch("swarm_notes.analyst.Agent")
@patch("swarm_notes.analyst._load_taxonomy")
def test_analyse(mock_load_taxonomy: MagicMock, mock_agent_class: MagicMock) -> None:
    """Test the analyse function.

    Mocks the Pydantic AI Agent and taxonomy loader to simulate extracting
    information from a paper without calling the LLM.

    Args:
        mock_load_taxonomy: Mock for the taxonomy loader.
        mock_agent_class: Mock for the Agent class.
    """
    mock_load_taxonomy.return_value = {
        "tags": ["llm"],
        "architectures": ["transformer"],
        "domains": ["nlp"],
    }

    mock_instance = mock_agent_class.return_value
    mock_result = MagicMock()

    mock_result.output = MockOutput()
    mock_instance.run_sync.return_value = mock_result

    paper = RawPaper(
        arxiv_id="12345",
        title="Test Title",
        abstract="Test abstract",
        authors=["Author Name"],
        published="2023-01-01",
        url="http://url",
        primary_category="cs.CL",
        jatsxml_url="",
    )

    load_skills()
    result = analyse(paper, get_general_skill())

    assert result.arxiv_id == "12345"
    assert result.url == "http://url"
    assert result.published == "2023-01-01"
    assert result.authors == ["Author Name"]


def test_normalise_tags_reuses_existing_spelling() -> None:
    from swarm_notes.analyst import _normalise_tags

    assert _normalise_tags(
        ["forecasting", "Forecasting", " New Tag ", ""], ["Forecasting", "LLM"]
    ) == ["Forecasting", "New Tag"]


def test_output_schema_has_no_tag_enum() -> None:
    """Existing tags must not become a schema enum (Gemini rejects large enums)."""
    from swarm_notes.analyst import PaperAnalysis

    tags_schema = PaperAnalysis.model_json_schema()["properties"]["tags"]
    assert tags_schema["items"] == {"type": "string"}


def test_normalise_tags_merges_spacing_variants_to_most_used() -> None:
    from collections import Counter

    from swarm_notes.analyst import _normalise_tags

    existing = ["time series forecasting", "time-series-forecasting"]
    counts = Counter({"time series forecasting": 12, "time-series-forecasting": 2})
    assert _normalise_tags(
        ["Time-Series Forecasting", "time_series_forecasting"], existing, counts
    ) == ["time series forecasting"]


@patch("swarm_notes.analyst.Agent")
@patch("swarm_notes.analyst._load_taxonomy", return_value={})
def test_analyse_prompt_lists_a_bounded_relevant_subset_of_tags(
    _mock_taxonomy: MagicMock, mock_agent_class: MagicMock, tmp_path, monkeypatch
) -> None:
    from swarm_notes import analyst
    from swarm_notes.config import settings

    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    for i in range(analyst.MAX_PROMPT_TAGS + 100):
        (papers_dir / f"p{i}.md").write_text(f'---\ntags:\n  - "filler topic {i}"\n---\n')
    (papers_dir / "relevant.md").write_text('---\ntags:\n  - "anomaly detection"\n---\n')
    monkeypatch.setattr(settings, "vault_papers_dir", papers_dir)

    mock_result = MagicMock()
    mock_result.output = MockOutput(tags=["Anomaly-Detection"])
    mock_agent_class.return_value.run_sync.return_value = mock_result

    paper = RawPaper(
        arxiv_id="1",
        title="Streaming anomaly detection for sensors",
        abstract="We detect anomalies.",
        authors=[],
        published="2026-01-01",
        url="http://url",
        primary_category="cs.LG",
        jatsxml_url="",
    )
    load_skills()
    result = analyse(paper, get_general_skill())

    prompt = mock_agent_class.call_args.kwargs["system_prompt"]
    tag_line = prompt.split("EXISTING TAGS IN VAULT:\n", 1)[1].split("\n", 1)[0]
    listed = tag_line.split(", ")
    assert len(listed) == analyst.MAX_PROMPT_TAGS
    assert listed[0] == "anomaly detection"
    assert result.tags == ["anomaly detection"]


def test_normalise_tags_splits_comma_joined_tags() -> None:
    from swarm_notes.analyst import _normalise_tags

    assert _normalise_tags(["time-series,forecasting", "forecasting"], ["forecasting"]) == [
        "time-series",
        "forecasting",
    ]
