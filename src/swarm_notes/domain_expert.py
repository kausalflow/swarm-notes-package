"""Domain Expert module: Scrapes ArXiv HTML (or bioRxiv JATS XML) and extracts deep insights like open questions."""

import logging
import requests
from bs4 import BeautifulSoup
from pydantic_ai import Agent

from swarm_notes.analyst import OpenQuestion
from swarm_notes.config import settings
from swarm_notes.router import SkillSpec

logger = logging.getLogger(__name__)

def extract_open_questions(arxiv_id: str, skill: SkillSpec | None = None, jatsxml_url: str = "") -> list[OpenQuestion]:
    """Scrape the full text of a paper and extract open questions using the LLM.

    For arXiv papers the HTML view at ``arxiv.org/html/{arxiv_id}`` is used.
    For bioRxiv/medRxiv papers, set *jatsxml_url* to the JATS XML URL from the
    API response — the full body text will be extracted from there instead.

    If no full text is available (404, missing URL, parse failure), returns [].
    """
    if jatsxml_url:
        text_content = _fetch_jatsxml_text(jatsxml_url, arxiv_id)
    else:
        text_content = _fetch_arxiv_html_text(arxiv_id)

    if not text_content:
        return []

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


def _fetch_jatsxml_text(jatsxml_url: str, paper_id: str) -> str:
    from swarm_notes.paper_search.biorxiv import fetch_jatsxml_text  # noqa: PLC0415

    logger.info("DomainExpert: Fetching JATS XML for paper %s at %s", paper_id, jatsxml_url)
    text = fetch_jatsxml_text(jatsxml_url)
    if not text:
        logger.warning("DomainExpert: JATS XML empty or unavailable for %s", paper_id)
    return text
