"""Analyst: structured extraction agent powered by pydantic-ai.

Reads the abstract, injects the taxonomy, and returns a fully typed
:class:`PaperAnalysis` object using the domain-specific skill prompt.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from swarm_notes.config import get_model, settings
from swarm_notes.router import SkillSpec
from swarm_notes.vault_manager import get_tag_counts, select_relevant
from swarm_notes.watcher import RawPaper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ConceptLink(BaseModel):
    """A concept mentioned in the paper that deserves its own vault entry."""

    slug: str = Field(
        description=(
            "URL-safe slug for the concept (e.g. 'flash-attention', 'mamba'). "
            "Use hyphens, no spaces."
        )
    )
    display_name: str = Field(
        description="Human-readable display name (e.g. 'Flash Attention', 'Mamba')."
    )
    one_liner: str = Field(
        description="One sentence that defines this concept in the context of the paper."
    )
    importance_reason: str = Field(
        default="",
        description=(
            "Why this concept is central to the paper's contribution or novelty. "
            "Leave empty if not applicable."
        ),
    )
    reusability_reason: str = Field(
        default="",
        description=(
            "Why this concept is likely to recur across future papers and deserves "
            "a permanent vault note. Leave empty if not applicable."
        ),
    )
    evidence_excerpt: str = Field(
        default="",
        description=(
            "Short phrase or sentence from the title or abstract supporting this concept. "
            "Leave empty if not applicable."
        ),
    )


class OpenQuestion(BaseModel):
    """An explicit open question, unresolved problem, or future work direction."""
    slug: str = Field(
        description=("URL-safe slug for the question (e.g. 'llm-reasoning-limits'). "
                     "Use hyphens, no spaces.")
    )
    title: str = Field(
        description="A concise, 3-5 word title for the open question."
    )
    background: str = Field(
        description="A highly concise, standalone 1-2 sentence background context unifying the open question. Must be written objectively as a universal encyclopedic fact. DO NOT use phrases like 'this paper', 'the authors', or 'this study'."
    )
    description: str = Field(
        description="Detailed explanation of the open question or future work."
    )
    importance_reason: str = Field(
        default="",
        description=(
            "Why this open question is technically important and worth tracking across papers. "
            "Leave empty if not applicable."
        ),
    )
    evidence_excerpt: str = Field(
        default="",
        description=(
            "Short phrase or sentence from the paper text supporting that this question is explicit and unresolved. "
            "Leave empty if not applicable."
        ),
    )


class RejectedCandidate(BaseModel):
    """Structured rejection record produced by the critic for auditability."""

    candidate_type: Literal["concept", "open_question", "dataset"] = Field(
        description="Whether the rejected candidate was a concept or open question."
    )
    candidate_slug: str = Field(
        description="Slug of the rejected candidate."
    )
    candidate_title: str = Field(
        description="Display name or title of the rejected candidate."
    )
    reason_code: Literal[
        "generic",
        "paper_local",
        "not_novel",
        "not_reusable",
        "weak_evidence",
        "low_impact",
        "duplicate_existing",
        "subcomponent_of_broader_mechanism",
        "other",
    ] = Field(
        description="Normalized rejection reason category for analytics and filtering."
    )
    reason: str = Field(
        description="Concise human-readable reason for rejection."
    )


class DatasetLink(BaseModel):
    """A dataset mentioned in the paper."""
    name: str = Field(description="Name of the dataset (e.g. 'european-space-agency-anomaly-dataset').")
    description: str = Field(description="Description of the dataset and how it is used in the paper.")


class PaperAnalysisBase(BaseModel):
    """Structured extraction output for a single academic paper (Base schema)."""

    title: str = Field(description="Clean, full title of the paper.")
    authors: list[str] = Field(description="List of author full names.")
    published: str = Field(description="Publication date as YYYY-MM-DD.")
    arxiv_id: str = Field(description="ArXiv identifier (e.g. '2301.12345').")
    source: str = Field(default="arxiv", description="Paper source backend (e.g. 'arxiv', 'semantic_scholar').")
    url: str = Field(description="Canonical URL of the paper.")

    summary: Annotated[
        str,
        Field(
            description=(
                "A concise 3-5 sentence summary of the paper's core contribution, "
                "methodology, and key results. Written for a technical ML researcher."
            )
        ),
    ]

    key_contributions: list[str] = Field(
        description="Bullet-point list of 3-5 key contributions."
    )

    architectures: list[str] = Field(
        default_factory=list,
        description=(
            "Architecture names used or proposed (from the taxonomy architectures list). "
            "Leave empty if not applicable."
        ),
    )

    datasets: list[DatasetLink] = Field(
        default_factory=list,
        description=(
            "Names of CRITICAL datasets used for evaluation or training "
            "(e.g. 'ImageNet', 'ETTh1', 'LAION-5B'). "
            "Prefer 0-2 named datasets and omit generic aggregate labels."
        ),
    )

    concepts: list[ConceptLink] = Field(
        default_factory=list,
        description=(
            "Key concepts introduced or heavily used by the paper that "
            "should have their own vault entry (e.g. a new architecture, "
            "technique, or dataset). Maximum 5."
        ),
    )

    limitations: str = Field(
        default="",
        description="Brief statement of known limitations or future work.",
    )

    domain: str = Field(
        description=(
            "Primary domain from the taxonomy domains list "
            "(e.g. 'nlp', 'computer-vision', 'time-series')."
        )
    )

    open_questions: list[OpenQuestion] = Field(
        default_factory=list,
        description="List of open questions or future research identified by the domain expert.",
    )

    critic_review_summary: str = Field(
        default="",
        description="Short archivist review summary explaining why items were accepted or rejected.",
    )

    critic_rejected_candidates: list[RejectedCandidate] = Field(
        default_factory=list,
        description="Structured rejection records for candidate concepts or open questions rejected by the archivist review.",
    )


class PaperAnalysis(PaperAnalysisBase):
    """Structured extraction output for a single academic paper."""
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Tags for this paper. Reuse tags from the EXISTING TAGS list whenever one applies; "
            "only add a new tag if no existing tag describes the idea. Maximum 10 tags."
        ),
    )


# Upper bound on how many existing tags are listed in the prompt. Tags are
# passed as prompt text rather than as an enum in the output schema: a schema
# enum with hundreds of values makes Gemini reject the request with
# 400 INVALID_ARGUMENT. The tags listed are the ones most relevant to the
# paper (see ``select_relevant``), so the prompt stays the same size as the
# vault grows.
MAX_PROMPT_TAGS = 200


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_system_prompt(
    skill: SkillSpec, taxonomy: dict, existing_tags: list[str] | None = None
) -> str:
    architectures_list = ", ".join(taxonomy.get("architectures", []))
    domains_list = ", ".join(taxonomy.get("domains", []))
    existing_tags_str = (
        ", ".join(existing_tags[:MAX_PROMPT_TAGS])
        if existing_tags
        else "(No tags currently exist in the vault)"
    )

    base_prompt = f"""You are an expert academic paper analyst specialising in machine learning research.

Your task is to analyse the provided paper abstract and extract structured information.

TAXONOMY (use ONLY these values for architectures/domains):
- Available architectures: {architectures_list}
- Available domains: {domains_list}

EXISTING TAGS IN VAULT:
{existing_tags_str}

SKILL CONTEXT ({skill.name}):
{skill.extra_system_prompt}

IMPORTANT RULES:
1. Prefer tags from the EXISTING TAGS list, spelled exactly as listed. Add a new tag only when none applies, and keep new tags to an absolute minimum.
2. Select domain ONLY from the taxonomy domains list above.
3. Select architectures ONLY from the taxonomy architectures list above (or leave empty).
4. The summary must be 3-5 sentences, technically precise but accessible.
5. Key contributions must be specific, not generic ("we propose X" → "X achieves Y on Z benchmark").
6. Concepts are candidate proposals for a later archivist review, not a dump of every term in the paper.
7. Default to zero concepts unless a concept is central to the paper's novelty and likely to matter beyond this single paper.
8. For each proposed concept, fill importance_reason, reusability_reason, and evidence_excerpt with concrete, specific justification from the title or abstract.
9. Reject generic terminology, routine datasets, benchmark names, and umbrella phrases unless the paper makes them a distinct reusable idea.
10. Prefer 0-2 excellent concept candidates over 5 weak ones.
11. Datasets are also candidate proposals for archival notes. Include only critical, named datasets that are central to evaluation claims.
12. Reject generic or aggregate dataset labels like "real-world datasets", "benchmark datasets", "10 datasets", or unnamed proprietary buckets.
13. Prefer 0-2 datasets. If uncertain a dataset is important enough for a standalone note, omit it.
"""
    if skill.analyst_system_prompt_override:
        return skill.analyst_system_prompt_override
    return base_prompt


def analyse(paper: RawPaper, skill: SkillSpec) -> PaperAnalysis:
    """Run the analyst agent on *paper* using the provided *skill* context.

    Parameters
    ----------
    paper:
        Raw paper data from the watcher.
    skill:
        Domain skill selected by the router.

    Returns
    -------
    PaperAnalysis
        Structured extraction result.
    """
    taxonomy = _load_taxonomy()
    tag_counts = get_tag_counts()
    existing_tags = sorted(tag_counts)
    # List one spelling per tag (the most used), so variants such as
    # "time-series" and "time series" don't take up several slots.
    canonical_tags = sorted(set(_canonical_tags(existing_tags, tag_counts).values()))
    prompt_tags = select_relevant(
        canonical_tags,
        f"{paper.title}\n{paper.abstract}",
        MAX_PROMPT_TAGS,
        tag_counts,
    )
    system_prompt = _build_system_prompt(skill, taxonomy, prompt_tags)

    agent: Agent[None, PaperAnalysis] = Agent(
        model=get_model(),
        output_type=PaperAnalysis,
        system_prompt=system_prompt,
    )

    user_message = (
        f"Title: {paper.title}\n\n"
        f"Authors: {', '.join(paper.authors)}\n\n"
        f"Published: {paper.published}\n\n"
        f"ArXiv ID: {paper.arxiv_id}\n\n"
        f"URL: {paper.url}\n\n"
        f"Abstract:\n{paper.abstract}"
    )

    logger.info("Analyst: analysing paper %s with skill %s", paper.arxiv_id, skill.name)

    result = agent.run_sync(user_message)
    analysis: PaperAnalysis = result.output
    analysis.tags = _normalise_tags(analysis.tags, existing_tags, tag_counts)

    # Ensure metadata from the raw paper is preserved accurately
    analysis.arxiv_id = paper.arxiv_id
    analysis.source = paper.source
    analysis.url = paper.url
    analysis.published = paper.published
    if not analysis.authors:
        analysis.authors = paper.authors

    logger.debug(
        "Analyst: paper %s → %d tags, %d concepts",
        paper.arxiv_id,
        len(analysis.tags),
        len(analysis.concepts),
    )
    return analysis


def _tag_key(tag: str) -> str:
    """Key that treats case, spaces, hyphens and underscores as the same."""
    return re.sub(r"[\s_\-]+", " ", tag.strip().lower()).strip()


def _canonical_tags(existing_tags: list[str], counts: Counter[str] | None = None) -> dict[str, str]:
    """Map each tag key to the vault's most used spelling of it."""
    counts = counts or Counter()
    canonical: dict[str, str] = {}
    for t in sorted(existing_tags, key=lambda t: (-counts.get(t, 0), t)):
        canonical.setdefault(_tag_key(t), t)
    return canonical


def _normalise_tags(
    tags: list[str],
    existing_tags: list[str],
    counts: Counter[str] | None = None,
) -> list[str]:
    """Map tags onto the vault's existing spelling and drop duplicates.

    "Time-Series", "time series" and "time_series" all map to whichever
    spelling the vault already uses most.
    """
    canonical = _canonical_tags(existing_tags, counts)
    result: list[str] = []
    seen: set[str] = set()
    for tag in (part.strip() for t in tags for part in t.split(",")):
        key = _tag_key(tag)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(canonical.get(key, tag))
    return result


def _load_taxonomy() -> dict:
    """Load taxonomy.json, returning an empty dict on failure."""
    if not settings.taxonomy_file.exists():
        logger.debug("Taxonomy file not found at %s. Using empty taxonomy.", settings.taxonomy_file)
        return {}
        
    try:
        with settings.taxonomy_file.open() as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not load taxonomy from %s: %s", settings.taxonomy_file, exc)
        return {}

