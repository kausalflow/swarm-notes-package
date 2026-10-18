"""Vault manager: staging pattern for atomic writes.

All agents write to ``tmp_vault/`` during a run.  On success, call
``commit_staging()`` to atomically move the staged files into ``vault/``.
On failure, call ``discard_staging()`` to remove ``tmp_vault/`` and prevent
partial corruption of the live vault.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import Counter
from pathlib import Path

import frontmatter

from swarm_notes.config import settings

logger = logging.getLogger(__name__)


def init_vault() -> None:
    """Create the primary vault directories if they do not exist."""
    settings.vault_papers_dir.mkdir(parents=True, exist_ok=True)
    settings.vault_concepts_dir.mkdir(parents=True, exist_ok=True)
    settings.vault_datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.vault_discussions_dir.mkdir(parents=True, exist_ok=True)
    settings.vault_daily_dir.mkdir(parents=True, exist_ok=True)
    settings.vault_daily_archive_dir.mkdir(parents=True, exist_ok=True)
    settings.vault_open_questions_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Vault structure initialised at %s", settings.vault_dir)


def init_staging() -> None:
    """Create (or recreate) the staging ``tmp_vault/`` directories."""
    if settings.tmp_vault_dir.exists():
        shutil.rmtree(settings.tmp_vault_dir)
        logger.debug("Removed stale tmp_vault at %s", settings.tmp_vault_dir)

    settings.tmp_papers_dir.mkdir(parents=True, exist_ok=True)
    settings.tmp_concepts_dir.mkdir(parents=True, exist_ok=True)
    settings.tmp_datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.tmp_discussions_dir.mkdir(parents=True, exist_ok=True)
    settings.tmp_daily_dir.mkdir(parents=True, exist_ok=True)
    settings.tmp_open_questions_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Staging area initialised at %s", settings.tmp_vault_dir)


def commit_staging() -> None:
    """Move staged files from ``tmp_vault/`` into the live ``vault/``.

    Files in staging *overwrite* existing vault files of the same name.
    New files are simply moved.  The staging directory is removed on
    completion.
    """
    if not settings.tmp_vault_dir.exists():
        logger.warning("commit_staging called but tmp_vault does not exist – nothing to do")
        return

    _merge_directory(settings.tmp_papers_dir, settings.vault_papers_dir)
    _merge_directory(settings.tmp_concepts_dir, settings.vault_concepts_dir)
    _merge_directory(settings.tmp_datasets_dir, settings.vault_datasets_dir)
    _merge_directory(settings.tmp_daily_dir, settings.vault_daily_dir)
    _merge_directory(settings.tmp_discussions_dir, settings.vault_discussions_dir)
    _merge_directory(settings.tmp_open_questions_dir, settings.vault_open_questions_dir)

    # 3. Clean up stagings.tmp_vault_dir)
    shutil.rmtree(settings.tmp_vault_dir)
    logger.info("Staging committed to vault and tmp_vault removed")


def discard_staging() -> None:
    """Remove ``tmp_vault/`` without touching the live vault."""
    if settings.tmp_vault_dir.exists():
        shutil.rmtree(settings.tmp_vault_dir)
        logger.info("Staging discarded – tmp_vault removed")
    else:
        logger.debug("discard_staging called but tmp_vault does not exist")


def _merge_directory(src: Path, dst: Path) -> None:
    """Copy all files from *src* into *dst*, overwriting on conflict."""
    dst.mkdir(parents=True, exist_ok=True)
    for src_file in src.iterdir():
        if src_file.is_file():
            dst_file = dst / src_file.name
            shutil.copy2(src_file, dst_file)
            logger.debug("Staged %s → %s", src_file, dst_file)


_ENTITY_DIR_SETTINGS = {
    "concepts": "vault_concepts_dir",
    "datasets": "vault_datasets_dir",
    "open_questions": "vault_open_questions_dir",
}

# Cache of parsed paper frontmatter, keyed on the papers directory and its
# state. A run only writes to tmp_vault/, so the live vault is parsed once per
# run instead of once per prompt.
_paper_metadata_cache: dict[tuple, list[dict]] = {}


def _load_paper_metadata() -> list[dict]:
    """Return the frontmatter of every live vault paper, parsed once per vault state."""
    papers_dir = settings.vault_papers_dir
    if not papers_dir.exists():
        return []
    files = sorted(papers_dir.glob("*.md"))
    key = (str(papers_dir), papers_dir.stat().st_mtime_ns, len(files))
    cached = _paper_metadata_cache.get(key)
    if cached is not None:
        return cached
    metadata: list[dict] = []
    for f in files:
        try:
            metadata.append(frontmatter.load(f).metadata)
        except Exception as exc:
            logger.warning("VaultManager: failed to parse frontmatter from %s: %s", f, exc)
    _paper_metadata_cache.clear()
    _paper_metadata_cache[key] = metadata
    return metadata


def get_tag_counts() -> Counter[str]:
    """Return how many live vault papers use each tag."""
    counts: Counter[str] = Counter()
    for meta in _load_paper_metadata():
        file_tags = meta.get("tags", [])
        if isinstance(file_tags, str):
            file_tags = [file_tags]
        if not isinstance(file_tags, list):
            continue
        # Some older notes hold several tags in one comma-joined string.
        split = {part.strip() for t in file_tags if isinstance(t, str) for part in t.split(",")}
        for t in split - {""}:
            counts[t] += 1
    return counts


def get_existing_tags() -> list[str]:
    """Return a deduplicated list of all tags currently found in live vault papers."""
    return sorted(get_tag_counts())


def get_existing_entities(entity_type: str) -> list[str]:
    """Return entity names (e.g. concepts, datasets, open_questions) known to the vault.

    Names come from paper frontmatter and from the slugs of the standalone
    entity notes (``vault/concepts/<slug>.md`` and so on).
    """
    entity_set: set[str] = set()
    for meta in _load_paper_metadata():
        entities = meta.get(entity_type, [])
        if isinstance(entities, list):
            for e in entities:
                if isinstance(e, dict) and e.get("name"):
                    entity_set.add(str(e["name"]))
    dir_setting = _ENTITY_DIR_SETTINGS.get(entity_type)
    entity_dir = getattr(settings, dir_setting, None) if dir_setting else None
    if entity_dir is not None and entity_dir.exists():
        entity_set.update(f.stem for f in entity_dir.glob("*.md") if not f.name.startswith("_"))
    return sorted(entity_set)


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "and", "are", "based", "for", "from", "into", "its", "new", "our",
    "the", "their", "this", "that", "towards", "using", "via", "with",
})


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def select_relevant(
    names: list[str],
    text: str,
    limit: int,
    counts: Counter[str] | None = None,
) -> list[str]:
    """Pick at most *limit* names, the ones sharing the most words with *text* first.

    Names with no word in common are used only to fill the remaining slots,
    most frequent first (by *counts*). This keeps prompts a fixed size however
    large the vault grows, while still showing the names a paper is likely to
    reuse.
    """
    if len(names) <= limit:
        return list(names)
    text_words = _words(text)
    counts = counts or Counter()

    def rank(name: str) -> tuple:
        name_words = _words(name)
        shared = len(name_words & text_words)
        # More shared words first, then names that are mostly shared words
        # ("forecasting" over "forecasting-for-sparse-retail-demand").
        return (-shared, -shared / max(len(name_words), 1), -counts.get(name, 0), name)

    return sorted(names, key=rank)[:limit]


def make_storage_id(source: str, paper_id: str) -> str:
    """Return the normalized storage identifier used for deduplication."""
    return f"{source.strip().lower()}:{paper_id.strip()}"


def get_existing_storage_ids() -> set[str]:
    """Return source-aware paper IDs already present in vault and staging.

    Supported filename formats:
    - New: ``<source>-<paper_id>-<title-slug>.md``
    - Legacy: ``<paper_id>-<title-slug>.md`` (assumed source ``arxiv``)
    """
    import re

    new_pattern = re.compile(r"^([a-z0-9_]+)-(\d{4}\.\d{4,5})-")
    legacy_pattern = re.compile(r"^(\d{4}\.\d{4,5})-")
    ids: set[str] = set()

    for search_dir in (settings.vault_papers_dir, settings.tmp_papers_dir):
        if not search_dir.exists():
            continue
        for file_path in search_dir.glob("*.md"):
            new_match = new_pattern.match(file_path.name)
            if new_match:
                ids.add(make_storage_id(new_match.group(1), new_match.group(2)))
                continue

            legacy_match = legacy_pattern.match(file_path.name)
            if legacy_match:
                ids.add(make_storage_id("arxiv", legacy_match.group(1)))

    return ids


def get_existing_arxiv_ids() -> set[str]:
    """Backward-compatible ArXiv-only view of existing storage IDs."""
    return {
        storage_id.split(":", 1)[1]
        for storage_id in get_existing_storage_ids()
        if storage_id.startswith("arxiv:")
    }
