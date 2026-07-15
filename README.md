# research-cruise 🚀

An autonomous, serverless, multi-agent system that tracks academic papers, extracts structured data, and weaves them into a local, interconnected Markdown knowledge graph — a **Second Brain** for ML research.  
Built to eventually communicate with other identical systems, forming a decentralised **Hive Mind**.

---

## Architecture

```
┌────────────────────────────────────────────┐
│                  Triggers                  │
└─────────────────────┬──────────────────────┘
                      │
         ┌────────────▼────────────┐
         │   Federation Agent      │  ← consumes external public_feed.json feeds
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │       Watcher           │  ← queries ArXiv API by keyword
         └────────────┬────────────┘
                      │  RawPaper[]
         ┌────────────▼────────────┐
         │    Router (Skill        │  ← routes each paper to a domain skill
         │    Registry)            │    (NLP, Vision, TimeSeries, …)
         └────────────┬────────────┘
                      │  Skill
         ┌────────────▼────────────┐
         │    Analyst              │  ← pydantic-ai structured extraction
         │    (pydantic-ai)        │    with taxonomy injection
         └────────────┬────────────┘
                      │  PaperAnalysis
         ┌────────────▼────────────┐
         │    Vault Writer         │  ← writes .md to tmp_vault/
         │                         │    generates concept stubs
         │                         │    updates public_feed.json
         └────────────┬────────────┘
                      │  atomic move
         ┌────────────▼────────────┐
         │       /vault            │  ← permanent, file-based knowledge graph
         │   papers/ concepts/     │
         │   datasets/             │
         └─────────────────────────┘
```

## Directory Structure

```
research-cruise/
├── .github/
│   └── workflows/
│       └── autonomous-tracker.yml   # CI/CD pipeline
├── vault/
│   ├── papers/                      # One .md file per paper
│   ├── concepts/                    # Auto-generated concept stubs
│   └── datasets/                    # Dataset stubs
├── swarm_notes/
│   ├── config.py                    # Configuration & env vars
│   ├── vault_manager.py             # Staging pattern (tmp_vault → vault)
│   ├── watcher.py                   # Configurable paper-source watcher
│   ├── router.py                    # Skill registry router
│   ├── analyst.py                   # pydantic-ai extraction agent
│   ├── vault_writer.py              # Markdown writer + public_feed.json
│   ├── federation.py                # Hive Mind federation agent
│   └── main.py                      # Pipeline orchestrator
```

## Quick Start

### Prerequisites

- Python 3.11+
- An LLM API key

### Local Dev Run

```bash
# Install dependencies
uv sync

# Set your API key in .env file
export LLM_API_KEY="sk-..."
export PAPER_SOURCE="semantic_scholar"
export SEMANTIC_SCHOLAR_API_KEY="..."

# prepare configs in configs/ folder
...

# Run the pipeline
python -m swarm_notes.main
```

### Configuration (Environment Variables)

Use the example in configs folder to create your own version.

#### bioRxiv / medRxiv: AWS credentials for full-text PDF download

Since May 2025 **biorxiv.org is protected by Cloudflare**, which blocks direct PDF
downloads. When `paper_source` is `biorxiv` or `medrxiv` and `enable_domain_expert`
is `true`, the pipeline uses the
[paperscraper](https://github.com/jannisborn/paperscraper) library to fall back to the
**biorxiv TDM (Text & Data Mining) API**, which serves PDFs from an AWS S3 bucket.

This requires an AWS IAM key with **read-only S3 access**:

1. Log in to the [AWS IAM console](https://console.aws.amazon.com/iam/).
2. Create a new user (or access key for an existing user).
3. Attach the `AmazonS3ReadOnlyAccess` managed policy.
4. Generate an **Access Key ID** and **Secret Access Key**.
5. Add them to your `.env` file (see `.env.example`):

```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

Without these credentials the domain-expert full-text step will be silently skipped
for biorxiv/medrxiv papers.

For CI/CD, add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as repository secrets
(same place as `LLM_API_KEY`).

## CI/CD Setup

### Add the required secret

The pipeline needs an OpenAI-compatible API key to run the LLM analyst step.

1. Open your forked repository on GitHub.
2. Go to **Settings → Secrets and variables → Actions**.
3. Click **New repository secret**.
4. Set **Name** to `LLM_API_KEY` and **Secret** to your API key (e.g. `sk-...`).
5. Click **Add secret**.

> **Note:** The workflow exposes `LLM_API_KEY` as both `LLM_API_KEY` and `OPENAI_API_KEY`
> so that pydantic-ai's OpenAI provider picks it up automatically.


## The Hive Mind (Federation)

Every successful run updates `public_feed.json` at the root of the repository with the metadata and summaries of the last 20 processed papers.

To subscribe to another agent's feed, pass their raw `public_feed.json` URL:

```bash
export FEDERATION_FEEDS="https://raw.githubusercontent.com/alice/research-cruise/main/public_feed.json,https://raw.githubusercontent.com/bob/research-cruise/main/public_feed.json"
python -m swarm_notes.main
```

**Conflict resolution:** If an external feed contains a review of a paper that already exists locally, the local metadata is preserved.  The external summary is appended under a `### External Perspectives` section:

```markdown
### External Perspectives

> "Transformers are over-engineered for this dataset." - @Agent_alice
> *(Retrieved 2024-01-15)*
```

## Vault File Format

Each paper note uses hybrid YAML frontmatter (CSL-compatible fields + custom fields):

```yaml
---
# CSL-compatible fields
title: "Attention Is All You Need"
author:
  - literal: "Ashish Vaswani"
issued:
  date-parts:
    - [2017, 6, 12]
url: "https://arxiv.org/abs/1706.03762"

# Custom fields
arxiv_id: "1706.03762"
domain: "nlp"
tags:
  - "transformer"
  - "attention-mechanism"
architectures:
  - "encoder-decoder"
datasets:
  - "WMT 2014"
skill: "NLPSkill"
processed_at: "2024-01-15T06:00:00Z"
---
```

Body sections: **Summary**, **Key Contributions**, **Key Concepts** (with relative links to `../concepts/`), **Datasets**, **Limitations**, **Links**.

## Taxonomy

`taxonomy.json` contains the controlled vocabulary of tags, architectures, and domains injected into the analyst's system prompt.  This prevents LLM hallucination and keeps metadata consistent.  Edit `taxonomy.json` to add new terms.

## Daily Note Compaction (Astro-friendly)

To avoid markdown file explosion on static sites, swarm-notes now maintains a **single rolling overview file** for daily discussions:

- `vault/discussions/_overview.md` (configurable via `vault_overview_file`)
- grouped by month and ISO week
- includes stable links to source daily notes (`/discussions/...` by default)
- includes short deterministic snippets derived from note content

Optional archival behavior:

- `daily_archive_cutoff_days`: move old files from `vault/discussions/daily/*.md` to `vault/discussions/archive/daily/YYYY/MM/`
- `daily_overview_include_archived`: include archived notes in overview (default `false`)

CLI overrides:

```bash
swarm-notes run \
  --archive-dailies-older-than 30 \
  --overview-file vault/discussions/_overview.md \
  --exclude-archived-in-overview
```

## License

MIT — see [LICENSE](LICENSE).