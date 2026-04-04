"""Domain Expert module: Scrapes ArXiv HTML (or bioRxiv JATS XML) and extracts deep insights like open questions."""

import logging
import requests
from bs4 import BeautifulSoup
from pydantic_ai import Agent

from swarm_notes.analyst import OpenQuestion
from swarm_notes.config import settings
from swarm_notes.router import SkillSpec

logger = logging.getLogger(__name__)

def extract_open_questions(
    arxiv_id: str,
    skill: SkillSpec | None = None,
    jatsxml_url: str = "",
    pdf_url: str = "",
) -> list[OpenQuestion]:
    """Scrape the full text of a paper and extract open questions using the LLM.

    For arXiv papers the HTML view at ``arxiv.org/html/{arxiv_id}`` is used.
    For bioRxiv/medRxiv papers, set *jatsxml_url* to the JATS XML URL from the
    API response — the full body text will be extracted from there instead.

    When *jatsxml_url* is unavailable or returns no text, and *pdf_url* is set,
    the PDF is downloaded and converted to Markdown as a fallback.

    If no full text is available (404, missing URL, parse failure), returns [].
    """
    if jatsxml_url:
        text_content = _fetch_jatsxml_text(jatsxml_url, arxiv_id)
        if not text_content and pdf_url:
            logger.info("DomainExpert: JATS XML empty for %s, falling back to PDF", arxiv_id)
            text_content = _fetch_pdf_text(pdf_url, arxiv_id)
    elif pdf_url:
        text_content = _fetch_pdf_text(pdf_url, arxiv_id)
    else:
        text_content = _fetch_arxiv_html_text(arxiv_id)

    if not text_content:
        return []

    _save_raw_markdown(arxiv_id, text_content)
    logger.info("DomainExpert: Analysing full text of %s (%d chars)", arxiv_id, len(text_content))

    system_prompt = (
        "You are a meticulous Domain Expert in machine learning. "
        "Carefully read the provided full text of a research paper and "
        "extract explicit 'open questions', 'future work', or 'unresolved problems' "
        "mentioned by the authors. Return them as a precise, concise list of structured candidate objects. "
        "If none are explicitly specified with sufficient detail, return an empty list! DO NOT hallucinate open questions if the authors don't discuss them. "
        "If multiple papers share generic open questions, ensure the titles and slugs are general enough to be reusable. "
        "CRUCIAL: When writing the background and description, write them as universal, standalone concepts. DO NOT use phrases like 'In this paper', 'The authors', or 'This study'. Frame the context objectively. "
        "These are candidate proposals for later archivist review, so default to zero unless the question is technically substantial, broadly reusable, and clearly unresolved. "
        "For every proposed open question, fill importance_reason and evidence_excerpt with specific justification from the paper text. "
        "Reject generic future work such as 'improve performance', 'run more experiments', or 'collect more data' unless the authors identify a specific unresolved bottleneck. "
        "Prefer 0-2 excellent open-question candidates over a long list."
    )
    if skill and skill.domain_expert_context:
        system_prompt += f"\n\nSKILL-SPECIFIC CONTEXT:\n{skill.domain_expert_context}"

    agent: Agent[None, list[OpenQuestion]] = Agent(
        model=settings.llm_model,
        output_type=list[OpenQuestion],
        system_prompt=system_prompt
    )

    try:
        result = agent.run_sync(text_content[:250000])
        return result.output
    except Exception as exc:
        logger.error("DomainExpert: LLM analysis failed for %s: %s", arxiv_id, exc)
        return []


def _fetch_arxiv_html_text(arxiv_id: str) -> str:
    url = f"https://arxiv.org/html/{arxiv_id}"
    logger.info("DomainExpert: Fetching arXiv HTML for paper %s at %s", arxiv_id, url)
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            logger.warning("DomainExpert: HTML view not found or disabled for %s", arxiv_id)
            return ""
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception as exc:
        logger.error("DomainExpert: failed to fetch arXiv HTML for %s: %s", arxiv_id, exc)
        return ""


def _save_raw_markdown(paper_id: str, text: str) -> None:
    """Persist *text* to vault/raw/papers/<sanitised-id>.md for later inspection."""
    import re  # noqa: PLC0415

    raw_dir = settings.vault_raw_papers_dir
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r'[^\w.\-]', '_', paper_id)
        dest = raw_dir / f"{safe_id}.md"
        dest.write_text(text, encoding="utf-8")
        logger.info("DomainExpert: saved raw markdown for %s to %s", paper_id, dest)
    except Exception as exc:
        logger.warning("DomainExpert: could not save raw markdown for %s: %s", paper_id, exc)


def _fetch_jatsxml_text(jatsxml_url: str, paper_id: str) -> str:
    from swarm_notes.paper_search.biorxiv import fetch_jatsxml_text  # noqa: PLC0415

    logger.info("DomainExpert: Fetching JATS XML for paper %s at %s", paper_id, jatsxml_url)
    text = fetch_jatsxml_text(jatsxml_url)
    if not text:
        logger.warning("DomainExpert: JATS XML empty or unavailable for %s", paper_id)
    return text


def _fetch_pdf_text(pdf_url: str, paper_id: str) -> str:
    logger.info("DomainExpert: Fetching PDF for paper %s", paper_id)
    # biorxiv/medrxiv papers have a DOI as their paper_id (e.g. "10.1101/…").
    # Use paperscraper for DOI-identified papers: it handles Cloudflare
    # protection and publisher-specific fallbacks automatically.
    if paper_id and paper_id.startswith("10."):
        from swarm_notes.pdf_converter import pdf_doi_to_markdown  # noqa: PLC0415

        text = pdf_doi_to_markdown(paper_id)
        if text:
            return text
        logger.warning(
            "DomainExpert: DOI-based PDF download failed for %s, falling back to direct URL", paper_id
        )

    from swarm_notes.pdf_converter import pdf_url_to_markdown  # noqa: PLC0415

    text = pdf_url_to_markdown(pdf_url)
    if not text:
        logger.warning("DomainExpert: PDF conversion empty or failed for %s", paper_id)
    return text
