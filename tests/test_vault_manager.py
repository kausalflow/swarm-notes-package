"""Tests for the vault_manager module.

This module contains unit tests to ensure that the initialisation,
staging, discarding, and committing of the vault work correctly.
"""

import pytest
from pathlib import Path

from unittest.mock import MagicMock, patch

from swarm_notes import vault_manager


def test_init_vault() -> None:
    """Test the init_vault function.

    Ensures that the primary vault directories are created if they
    do not exist.
    """
    mock_mkdir = MagicMock()
    with patch("pathlib.Path.mkdir", mock_mkdir):
        vault_manager.init_vault()
    assert mock_mkdir.call_count == 7


def test_init_staging() -> None:
    """Test the init_staging function.

    Ensures that the staging directory is reset and re-initialised
    correctly when it already exists.
    """
    mock_mkdir = MagicMock()
    with patch("pathlib.Path.mkdir", mock_mkdir):
        with patch("swarm_notes.vault_manager.settings.tmp_vault_dir") as mock_dir:
            mock_dir.exists.return_value = True
            with patch("shutil.rmtree") as mock_rmtree:
                vault_manager.init_staging()
                mock_rmtree.assert_called_once()
    assert mock_mkdir.call_count == 6


def test_commit_staging() -> None:
    """Test the commit_staging function.

    Ensures that staged files are merged into the vault and
    the staging area is subsequently removed.
    """
    with patch("swarm_notes.vault_manager.settings.tmp_vault_dir") as mock_dir:
        mock_dir.exists.return_value = True
        with patch("swarm_notes.vault_manager._merge_directory") as mock_merge:
            with patch("shutil.rmtree") as mock_rmtree:
                vault_manager.commit_staging()
                assert mock_merge.call_count == 6
                mock_rmtree.assert_called_once()


def test_commit_staging_not_exists() -> None:
    """Test the commit_staging function when no staging exists.

    Ensures that no action is taken if the staging area is absent.
    """
    with patch("swarm_notes.vault_manager.settings.tmp_vault_dir") as mock_dir:
        mock_dir.exists.return_value = False
        with patch("swarm_notes.vault_manager._merge_directory") as mock_merge:
            with patch("shutil.rmtree") as mock_rmtree:
                vault_manager.commit_staging()
                mock_merge.assert_not_called()
                mock_rmtree.assert_not_called()


def test_discard_staging() -> None:
    """Test the discard_staging function.

    Ensures that the staging area is removed without merging.
    """
    with patch("swarm_notes.vault_manager.settings.tmp_vault_dir") as mock_dir:
        mock_dir.exists.return_value = True
        with patch("shutil.rmtree") as mock_rmtree:
            vault_manager.discard_staging()
            mock_rmtree.assert_called_once()


def test_get_existing_storage_ids_supports_legacy_and_source_prefixed_filenames(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault_papers_dir = tmp_path / "vault_papers"
    tmp_papers_dir = tmp_path / "tmp_papers"
    vault_papers_dir.mkdir()
    tmp_papers_dir.mkdir()

    # Legacy format: <paper_id>-<title>.md (implied arxiv source).
    (vault_papers_dir / "2503.01234-paper-one.md").write_text("", encoding="utf-8")
    # New format: <source>-<paper_id>-<title>.md.
    (tmp_papers_dir / "semantic_scholar-2503.01235-paper-two.md").write_text("", encoding="utf-8")

    monkeypatch.setattr(vault_manager.settings, "vault_papers_dir", vault_papers_dir)
    monkeypatch.setattr(vault_manager.settings, "tmp_papers_dir", tmp_papers_dir)

    storage_ids = vault_manager.get_existing_storage_ids()

    assert storage_ids == {
        "arxiv:2503.01234",
        "semantic_scholar:2503.01235",
    }


def test_get_existing_arxiv_ids_filters_non_arxiv_sources(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault_papers_dir = tmp_path / "vault_papers"
    tmp_papers_dir = tmp_path / "tmp_papers"
    vault_papers_dir.mkdir()
    tmp_papers_dir.mkdir()

    (vault_papers_dir / "2503.01234-paper-one.md").write_text("", encoding="utf-8")
    (tmp_papers_dir / "semantic_scholar-2503.01235-paper-two.md").write_text("", encoding="utf-8")
    (tmp_papers_dir / "arxiv-2503.01236-paper-three.md").write_text("", encoding="utf-8")

    monkeypatch.setattr(vault_manager.settings, "vault_papers_dir", vault_papers_dir)
    monkeypatch.setattr(vault_manager.settings, "tmp_papers_dir", tmp_papers_dir)

    arxiv_ids = vault_manager.get_existing_arxiv_ids()

    assert arxiv_ids == {"2503.01234", "2503.01236"}


def test_select_relevant_prefers_shared_words_then_frequency() -> None:
    from collections import Counter

    names = ["graph neural networks", "anomaly detection", "llm", "forecasting"]
    counts = Counter({"llm": 50, "forecasting": 10, "graph neural networks": 1})
    picked = vault_manager.select_relevant(
        names, "Anomaly detection with graph models", 3, counts
    )
    assert picked == ["anomaly detection", "graph neural networks", "llm"]
    assert vault_manager.select_relevant(names, "", 10) == names


def test_get_existing_entities_includes_entity_note_slugs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    papers_dir = tmp_path / "papers"
    concepts_dir = tmp_path / "concepts"
    papers_dir.mkdir()
    concepts_dir.mkdir()
    (papers_dir / "p.md").write_text('---\nconcepts:\n  - name: "from-frontmatter"\n---\n')
    (concepts_dir / "from-note.md").write_text("---\ntitle: x\n---\n")
    monkeypatch.setattr(vault_manager.settings, "vault_papers_dir", papers_dir)
    monkeypatch.setattr(vault_manager.settings, "vault_concepts_dir", concepts_dir)

    assert vault_manager.get_existing_entities("concepts") == ["from-frontmatter", "from-note"]


def test_paper_metadata_is_parsed_once_per_vault_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "p.md").write_text('---\ntags:\n  - "a"\n---\n')
    monkeypatch.setattr(vault_manager.settings, "vault_papers_dir", papers_dir)

    real_load = vault_manager.frontmatter.load
    with patch("swarm_notes.vault_manager.frontmatter.load", side_effect=real_load) as load:
        assert vault_manager.get_existing_tags() == ["a"]
        assert vault_manager.get_existing_tags() == ["a"]
        assert load.call_count == 1
        (papers_dir / "q.md").write_text('---\ntags:\n  - "b"\n---\n')
        assert vault_manager.get_existing_tags() == ["a", "b"]
