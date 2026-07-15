"""Vault Writer: transforms PaperAnalysis into Markdown files.

Writes to the staging ``tmp_vault/`` directories.  Also maintains the
rolling ``public_feed.json`` file for the Hive Mind federation.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import frontmatter

from swarm_notes.analyst import PaperAnalysis, OpenQuestion
from swarm_notes.config import settings

logger = logging.getLogger(__name__)

# YAML frontmatter template
_FRONTMATTER_TEMPLATE = """\
---
# OKF Required Field
type: "paper"

# OKF Recommended Core Fields
title: "{title}"
description: "{description}"
resource: "{url}"
tags:
{tag_lines}
timestamp: "{processed_at}"

# CSL-compatible fields (Retained as custom)
author:
{author_lines}
issued:
  date-parts:
    - [{date_parts}]

# Domain-specific custom fields
paper_id: "{paper_id}"
paper_source: "{paper_source}"
architectures:
{architecture_lines}
concepts:
{concept_lines}
datasets:
{dataset_lines}
open_questions:
{open_question_lines}
skill: "{skill}"
processed_at: "{processed_at}"
created_at: "{created_at}"
---
"""


def write_paper(analysis: PaperAnalysis, skill_name: str) -> Path:
    """Write a paper Markdown file to the staging directory.

    Parameters
    ----------
    analysis:
        Structured extraction result from the Analyst.
    skill_name:
        Name of the Skill used (stored in frontmatter for traceability).

    Returns
    -------
    Path
        Path to the written staging file.
    """
    slug = _make_slug(analysis.source, analysis.arxiv_id, analysis.title)
    out_path = settings.tmp_papers_dir / f"{slug}.md"

    frontmatter = _build_frontmatter(analysis, skill_name)
    body = _build_body(analysis)
    content = f"{frontmatter}\n{body}\n"

    out_path.write_text(content, encoding="utf-8")
    logger.info("VaultWriter: wrote paper → %s", out_path)

    return out_path


def _build_frontmatter(analysis: PaperAnalysis, skill_name: str) -> str:
    """Render the YAML frontmatter block."""
    author_lines = "\n".join(
        f'  - literal: "{_yaml_escape(a)}"' for a in analysis.authors
    )
    if not author_lines:
        author_lines = "  []"

    # issued date-parts: [YYYY, MM, DD]
    try:
        dt = datetime.strptime(analysis.published, "%Y-%m-%d")
        date_parts = f"{dt.year}, {dt.month}, {dt.day}"
    except ValueError:
        date_parts = "null"

    tag_lines = "\n".join(f'  - "{t}"' for t in analysis.tags) or "  []"
    architecture_lines = (
        "\n".join(f'  - "{a}"' for a in analysis.architectures) or "  []"
    )
    
    def _render_okf_list(items) -> str:
        if not items:
            return "  []"
        lines = []
        for item in items:
            # Handle ConceptLink (slug/one_liner), DatasetLink (name/description), OpenQuestion (slug/description)
            name = getattr(item, "slug", getattr(item, "name", ""))
            desc = getattr(item, "one_liner", getattr(item, "description", ""))
            lines.append(f'  - name: "{_yaml_escape(name)}"\n    description: "{_yaml_escape(desc)}"')
        return "\n".join(lines)

    concept_lines = _render_okf_list(analysis.concepts)
    dataset_lines = _render_okf_list(analysis.datasets)
    open_question_lines = _render_okf_list(analysis.open_questions)

    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Escape summary for description field (strip newlines which might break YAML block if not multiline string)
    safe_description = _yaml_escape(analysis.summary.replace('\n', ' '))

    return _FRONTMATTER_TEMPLATE.format(
        title=_yaml_escape(analysis.title),
        description=safe_description,
        url=analysis.url,
        author_lines=author_lines,
        date_parts=date_parts,
        paper_id=analysis.arxiv_id,
        paper_source=analysis.source,
        domain=analysis.domain,
        tag_lines=tag_lines,
        architecture_lines=architecture_lines,
        concept_lines=concept_lines,
        dataset_lines=dataset_lines,
        open_question_lines=open_question_lines,
        skill=skill_name,
        processed_at=now_utc,
        created_at=now_utc,
    )


def _build_body(analysis: PaperAnalysis) -> str:
    """Build the Markdown body of the paper note."""
    lines: list[str] = []

    lines.append(f"# {analysis.title}\n")
    lines.append(f"**Authors**: {', '.join(analysis.authors)}")
    lines.append(f"**Date**: {analysis.published}")
    lines.append(f"**Paper ID**: [{analysis.source}:{analysis.arxiv_id}]({analysis.url})\n")

    lines.append("## Summary\n")
    lines.append(f"{analysis.summary}\n")

    lines.append("## Key Contributions\n")
    for contrib in analysis.key_contributions:
        lines.append(f"- {contrib}")
    lines.append("")

    if analysis.limitations:
        lines.append("## Limitations\n")
        lines.append(f"{analysis.limitations}\n")

    if analysis.open_questions:
        lines.append("## Open Questions & Future Work\n")
        for q in analysis.open_questions:
            lines.append(f"- [[{q.slug}]]")
        lines.append("")

    # Concepts
    if analysis.concepts:
        lines.append("## Key Concepts\n")
        for concept in analysis.concepts:
            wiki_link = f"[[{concept.slug}]]"
            lines.append(f"- {wiki_link}: {concept.one_liner}")
        lines.append("")

    if analysis.critic_review_summary or analysis.critic_rejected_candidates:
        lines.append("## Archivist Review\n")
        if analysis.critic_review_summary:
            lines.append(analysis.critic_review_summary)
            lines.append("")

        if analysis.concepts:
            lines.append("### Approved Concepts")
            for concept in analysis.concepts:
                details = concept.importance_reason or concept.reusability_reason or concept.evidence_excerpt
                if details:
                    lines.append(f"- {concept.display_name}: {details}")
            lines.append("")

        if analysis.open_questions:
            lines.append("### Approved Open Questions")
            for question in analysis.open_questions:
                details = question.importance_reason or question.evidence_excerpt
                if details:
                    lines.append(f"- {question.title}: {details}")
            lines.append("")

        if analysis.critic_rejected_candidates:
            lines.append("### Rejected Candidates")
            for rejection in analysis.critic_rejected_candidates:
                lines.append(
                    f"- [{rejection.candidate_type}] {rejection.candidate_title} "
                    f"(`{rejection.candidate_slug}`) - {rejection.reason_code}: {rejection.reason}"
                )
            lines.append("")

    # Datasets
    if analysis.datasets:
        lines.append("## Datasets\n")
        for ds in analysis.datasets:
            ds_slug = getattr(ds, "name", "unknown")
            lines.append(f"- [[{ds_slug}]]")
        lines.append("")

    # Links
    lines.append("## Links\n")
    lines.append(f"- [Abstract]({analysis.url})")
    lines.append(f"- [PDF]({analysis.url.replace('abs', 'pdf')})")
    lines.append("")

    return "\n".join(lines)


def _yaml_list(values: list[str]) -> str:
    """Render a YAML list with sane fallback for empty lists."""
    return "\n".join(f"  - \"{_yaml_escape(v)}\"" for v in values) if values else "  []"


# ---------------------------------------------------------------------------
# Public feed maintenance
# ---------------------------------------------------------------------------


def update_public_feed(analyses: list[PaperAnalysis]) -> None:
    """Add *analyses* to ``public_feed.json``, keeping the last N entries.

    The feed is updated in the repo root (not staging) so that it is always
    current and accessible as a raw URL for federation consumers.

    Parameters
    ----------
    analyses:
        List of :class:`PaperAnalysis` objects from the current run.
    """
    existing: list[dict] = _load_feed()
    existing_ids = {
        entry.get("paper_id") or entry.get("arxiv_id")
        for entry in existing
        if entry.get("paper_id") or entry.get("arxiv_id")
    }

    new_entries = [
        _analysis_to_feed_entry(a)
        for a in analyses
        if a.arxiv_id not in existing_ids
    ]

    combined = new_entries + existing
    trimmed = combined[:settings.public_feed_max_items]

    settings.public_feed_file.write_text(
        json.dumps(trimmed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Public feed updated: %d new, %d total entries",
        len(new_entries),
        len(trimmed),
    )


def write_site_config() -> None:
    """Write site identity metadata to ``settings.site_config_file``.

    The JSON file is read by the Astro website at build time to populate the
    hero section with the deployment-specific name, description, and tracked
    research topics.  It is committed alongside the vault so CI can build the
    website without re-loading the Python config.

    Fields written
    --------------
    site_name : str
        Human-readable name for this swarm-notes deployment.
    site_description : str
        Short description shown in the hero subtitle.
    paper_keywords : list[str]
        Research topics this instance tracks (rendered as pills in the hero).
    updated_at : str
        ISO-8601 UTC timestamp of the last pipeline run.
    """
    out_path = settings.site_config_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "site_name": settings.site_name,
        "site_description": settings.site_description,
        "paper_keywords": settings.paper_keywords,
        "arxiv_keywords": settings.paper_keywords,
        "updated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("VaultWriter: wrote site config → %s", out_path)


def append_daily_discussion(content: str) -> None:
    """Appends the discussion content to today's daily note in staging.

    Creates the file with a ``created_at`` frontmatter timestamp if it is new,
    or prepends / updates a ``modified_at`` timestamp on subsequent appends.
    """
    today_str = datetime.now().date().isoformat()
    filename = f"{today_str}.md"

    live_path = settings.vault_daily_dir / filename
    staging_path = settings.tmp_daily_dir / filename

    # If not already staged, but exists in live vault, copy it over first
    if not staging_path.exists() and live_path.exists():
        shutil.copy2(live_path, staging_path)

    is_new = not staging_path.exists()
    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n\n## Discussion for {now_str}\n\n"

    if is_new:
        # Write frontmatter + heading first
        front = f"---\ncreated_at: \"{now_utc}\"\n---\n\n# Daily Notes: {today_str}\n"
        staging_path.write_text(front, encoding="utf-8")
        with staging_path.open("a", encoding="utf-8") as f:
            f.write(header + content + "\n")
    else:
        # Update modified_at in existing frontmatter then append
        try:
            post = frontmatter.load(staging_path)
            post.metadata["modified_at"] = now_utc
            with staging_path.open("w", encoding="utf-8") as f:
                frontmatter.dump(post, f)
        except Exception as exc:
            logger.warning("VaultWriter: could not update modified_at for %s: %s", filename, exc)
        with staging_path.open("a", encoding="utf-8") as f:
            f.write(header + content + "\n")

    logger.info("VaultWriter: appended to daily discussion → %s", staging_path)


def compact_daily_notes() -> dict[str, int]:
    """Archive old daily notes and regenerate the rolling overview file."""
    archived_count = _archive_old_daily_notes(settings.daily_archive_cutoff_days)
    note_count = generate_daily_overview(settings.daily_overview_include_archived)
    return {"archived_count": archived_count, "overview_note_count": note_count}


def generate_daily_overview(include_archived: bool = False) -> int:
    """Generate one deterministic overview markdown file grouped by month/week."""
    active_notes = _collect_daily_notes(settings.vault_daily_dir, archived=False)
    archived_notes = (
        _collect_daily_notes(settings.vault_daily_archive_dir, archived=True)
        if include_archived
        else []
    )
    all_notes = sorted(active_notes + archived_notes, key=lambda item: item["date"], reverse=True)

    overview = _render_daily_overview(all_notes)
    settings.vault_overview_file.parent.mkdir(parents=True, exist_ok=True)

    existing = ""
    if settings.vault_overview_file.exists():
        existing = settings.vault_overview_file.read_text(encoding="utf-8")
    if existing != overview:
        settings.vault_overview_file.write_text(overview, encoding="utf-8")
        logger.info("VaultWriter: wrote overview file → %s", settings.vault_overview_file)
    else:
        logger.debug("VaultWriter: overview file unchanged at %s", settings.vault_overview_file)

    return len(all_notes)


def _archive_old_daily_notes(cutoff_days: int) -> int:
    """Move daily note files older than cutoff into the archive folder."""
    if cutoff_days <= 0:
        return 0

    if not settings.vault_daily_dir.exists():
        return 0

    cutoff_date = datetime.now(tz=timezone.utc).date() - timedelta(days=cutoff_days)
    archived = 0

    for note_path in sorted(settings.vault_daily_dir.glob("*.md")):
        note_date = _parse_daily_filename(note_path)
        if note_date is None or note_date >= cutoff_date:
            continue

        archive_target = (
            settings.vault_daily_archive_dir
            / f"{note_date.year:04d}"
            / f"{note_date.month:02d}"
            / note_path.name
        )
        archive_target.parent.mkdir(parents=True, exist_ok=True)
        if archive_target.exists():
            logger.warning(
                "VaultWriter: archive target already exists (%s); skipping archive move for %s",
                archive_target,
                note_path,
            )
            continue
        note_path.rename(archive_target)
        archived += 1

    if archived:
        logger.info("VaultWriter: archived %d daily note(s)", archived)
    return archived


def _collect_daily_notes(root: Path, archived: bool) -> list[dict]:
    notes: list[dict] = []
    if not root.exists():
        return notes

    pattern = "**/*.md" if archived else "*.md"
    for note_path in sorted(root.glob(pattern)):
        note_date = _parse_daily_filename(note_path)
        if note_date is None:
            continue
        notes.append(
            {
                "date": note_date,
                "path": note_path,
                "archived": archived,
                "snippet": _extract_summary_snippet(note_path),
            }
        )
    return notes


def _parse_daily_filename(path: Path):
    try:
        return datetime.strptime(path.stem, "%Y-%m-%d").date()
    except ValueError:
        return None


def _extract_summary_snippet(path: Path) -> str:
    in_frontmatter = False
    past_frontmatter = False
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not past_frontmatter and not in_frontmatter and stripped == "---":
                in_frontmatter = True
                continue
            if in_frontmatter:
                if stripped == "---":
                    in_frontmatter = False
                    past_frontmatter = True
                continue
            if not stripped or stripped.startswith("#"):
                continue
            normalized = re.sub(r"\s+", " ", stripped).strip()
            if normalized:
                return normalized[:120]
    return "No summary available."


def _render_daily_overview(notes: list[dict]) -> str:
    lines = [
        "# Daily Notes Overview",
        "",
        "This file is auto-generated from daily discussion notes.",
        "",
    ]

    if not notes:
        lines.append("_No daily notes found._")
        lines.append("")
        return "\n".join(lines)

    months: dict[str, list[dict]] = {}
    for note in notes:
        month_key = note["date"].strftime("%Y-%m")
        months.setdefault(month_key, []).append(note)

    for month_key in sorted(months.keys(), reverse=True):
        month_notes = months[month_key]
        lines.append(f"## {month_key}")
        lines.append(f"_{_period_summary(month_notes)}_")
        lines.append("")

        weeks: dict[str, list[dict]] = {}
        for note in month_notes:
            iso_year, iso_week, _ = note["date"].isocalendar()
            week_key = f"{iso_year}-W{iso_week:02d}"
            weeks.setdefault(week_key, []).append(note)

        for week_key in sorted(weeks.keys(), reverse=True):
            week_notes = weeks[week_key]
            lines.append(f"### Week {week_key}")
            lines.append(f"_{_period_summary(week_notes)}_")
            lines.append("")

            for note in sorted(week_notes, key=lambda item: item["date"], reverse=True):
                link = _daily_note_route(note["path"])
                archived_suffix = " *(archived)*" if note["archived"] else ""
                lines.append(
                    f"- [{note['date'].isoformat()}]({link}){archived_suffix} — {note['snippet']}"
                )
            lines.append("")

    return "\n".join(lines)


def _period_summary(notes: list[dict]) -> str:
    snippets = [note["snippet"] for note in notes if note["snippet"] != "No summary available."]
    if not snippets:
        return f"{len(notes)} notes."

    unique_snippets: list[str] = []
    for snippet in snippets:
        if snippet not in unique_snippets:
            unique_snippets.append(snippet)
        if len(unique_snippets) == 2:
            break

    joined = "; ".join(unique_snippets)
    return f"{len(notes)} notes. Highlights: {joined}"


def _daily_note_route(path: Path) -> str:
    try:
        rel_path = path.relative_to(settings.vault_discussions_dir)
    except ValueError:
        try:
            rel_path = path.relative_to(settings.vault_dir)
        except ValueError:
            logger.warning(
                "VaultWriter: cannot compute relative route for %s; falling back to filename",
                path,
            )
            rel_path = Path(path.name)

    rel = rel_path.as_posix()
    rel_without_ext = rel[:-3] if rel.endswith(".md") else rel
    prefix = settings.daily_overview_link_prefix.rstrip("/")
    return f"{prefix}/{rel_without_ext}" if prefix else f"/{rel_without_ext}"


def _analysis_to_feed_entry(analysis: PaperAnalysis) -> dict:
    return {
        "paper_id": analysis.arxiv_id,
        "paper_source": analysis.source,
        "title": analysis.title,
        "authors": analysis.authors,
        "published": analysis.published,
        "url": analysis.url,
        "domain": analysis.domain,
        "tags": analysis.tags,
        "concepts": [
            {"name": getattr(c, "slug", getattr(c, "name", "")), "description": getattr(c, "one_liner", getattr(c, "description", ""))}
            for c in analysis.concepts
        ],
        "datasets": [
            {"name": getattr(d, "name", ""), "description": getattr(d, "description", "")}
            for d in analysis.datasets
        ],
        "open_questions": [
            {"name": getattr(q, "slug", getattr(q, "name", "")), "description": getattr(q, "description", "")}
            for q in analysis.open_questions
        ],
        "summary": analysis.summary,
        "processed_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _load_feed() -> list[dict]:
    try:
        with settings.public_feed_file.open() as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_slug(source: str, paper_id: str, title: str) -> str:
    """Create a filename slug: ``<source>-<paper_id>-<title-slug>``."""
    source_slug = _slugify(source) or "unknown"
    paper_id_slug = _slugify(paper_id) or "unknown"
    title_slug = _slugify(title)[:60].rstrip("-")
    return f"{source_slug}-{paper_id_slug}-{title_slug}"


def _slugify(text: str) -> str:
    """Convert *text* to a lowercase hyphen-separated slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _yaml_escape(text: str) -> str:
    """Escape double quotes in YAML string values."""
    return text.replace('"', '\\"')
