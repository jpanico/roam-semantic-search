# roam-semantic-search

Fully local semantic search over a [Roam Research](https://roamresearch.com) graph:
fetch clear-text content through the Roam Local API, embed it with a locally hosted
model, store vectors in a single SQLite file, and answer meaning-based queries from
a CLI or an MCP server. **Nothing about the graph's content ever leaves the
machine** — that constraint is the project's founding requirement, and it is
enforced in code: the embedding client refuses any non-loopback server URL.

```
Roam Desktop ──(Local API, localhost HTTP)──► fetch ──► normalize ──► embed ──► store
                                                                        ▲          │
                                                          Ollama (localhost)   SQLite (FTS5 + vector blobs)
                                                                                   │
                                              MCP server (stdio) ◄── query ◄───────┘
                                              CLI (roam-semantic-search search)
```

Full design, phase results, and decision log: [docs/design-plan.md](docs/design-plan.md).

## How it works

- **Fetch** — one flat Datalog pull of every entity carrying a `:block/uid`
  (pages and blocks alike) through the Roam Local API; ~1 s for a 10k-entity graph.
  For an encrypted graph, the running Roam Desktop client is the only clear-text
  doorway, so the indexer runs on the same machine.
- **Normalize** — each block embeds with its **breadcrumb**: the page title plus
  ancestor block texts, root-first (ordered by ancestor count, never by wire order,
  which is creation order and diverges from depth on ~12% of nested blocks).
  Roam markup is cleaned to prose (`[[refs]]` → text, `((uid))` references resolve
  to their target's text one level deep); `roam/js` and `roam/css` pages are
  skipped, and daily-note pages are indexed (skippable with `--no-daily-notes`).
- **Embed** — a local [Ollama](https://ollama.com) server running
  `nomic-embed-text` (768-dim), with the model's `search_document:` /
  `search_query:` retrieval prefixes. Loopback-only, enforced.
- **Store** — one SQLite file (default `~/.cache/roam-semantic-search/<graph>.db`):
  records + float32 embedding blobs, an FTS5 keyword mirror, and provenance meta.
  No SQLite extensions; vector KNN is a brute-force numpy matrix product
  (milliseconds at this scale).
- **Query** — hybrid retrieval: cosine KNN and BM25 rankings fused by reciprocal
  rank fusion, so paraphrase ("where do I argue…") and exact identifiers both rank.
- **Refresh** — incremental: re-fetch + re-normalize everything (cheap), then
  re-embed only records whose content hash changed and delete vanished uids.
  Selection is by content hash alone — an edit changes descendants' breadcrumbs
  and referrers' resolved text, which no per-entity timestamp can see. A no-change
  refresh takes ~2 s.

## Requirements

- **Roam Desktop** running locally with the Local API enabled (port, graph name,
  and a bearer token from Roam → Settings)
- **Ollama** with the embedding model pulled: `ollama pull nomic-embed-text`
  (`brew services start ollama` keeps it running at login)
- **Python ≥ 3.14** and a sibling checkout of
  [guffin](https://github.com/jpanico/guffin) (the Local API transport layer)

## Install

```bash
python3.14 -m venv .venv
.venv/bin/pip install -e ../guffin
.venv/bin/pip install -e ".[dev]"
```

## Configuration

The CLI and MCP server read the same environment the guffin tools use:

| Variable | Meaning |
|---|---|
| `GUFFIN_ROAM_LOCAL_API_PORT` | Roam Local API port (backs `--port`/`-p`) |
| `GUFFIN_ROAM_GRAPH_NAME` | Graph name (backs `--graph`/`-g`; also names the default DB) |
| `GUFFIN_ROAM_API_TOKEN` | Local API bearer token (backs `--token`/`-t`) |
| `ROAM_SEMANTIC_SEARCH_DB` | Explicit index DB path (else `~/.cache/roam-semantic-search/<graph>.db`) |
| `ROAM_SEMANTIC_SEARCH_OLLAMA_URL` | Embedding server URL (default `http://127.0.0.1:11434`; must be loopback) |

## CLI

```bash
roam-semantic-search build              # full fetch → normalize → embed → store (~100 s for ~8k records)
roam-semantic-search refresh            # incremental: re-embed only what changed (~2 s when idle)
roam-semantic-search search "why the human must stay responsible" -k 5
roam-semantic-search stats              # store provenance: model, counts, build/refresh moments
```

A hit shows the Roam uid (usable as a `((ref))`), the fused score, each ranking's
position (`v:` vector, `k:` keyword), the breadcrumb, and the text:

```
 1. ((9KMmmo5aH))  [block  score 0.0323  v:2 k:2]
    The new Programmer (in the age of AI assistants) › The human Programmer/engineer
    The human also remains the accountability boundary. The assistant can propose; ...
```

## MCP server

`roam-semantic-search-mcp` serves the index over stdio to any MCP client, with
three tools: `semantic_search` (hits plus index meta, so a caller can judge
staleness), `refresh_index`, and `index_stats`. Register with Claude Code:

```bash
claude mcp add --scope user roam-semantic-search --env GUFFIN_ROAM_GRAPH_NAME=<graph> -- $(pwd)/.venv/bin/roam-semantic-search-mcp
```

The Local API port and token are inherited from the shell environment rather than
stored in the client's config; without them `refresh_index` fails cleanly while
search keeps working.

## Development

```bash
.venv/bin/black .
.venv/bin/ruff check --fix src/ tests/
.venv/bin/pyright          # strict
.venv/bin/pytest
```

Conventions follow guffin's (Python 3.14, src layout, pyright strict,
`@validate_call`, `regex` not `re`, 120-char lines). The index DB contains the
graph's text in clear form — treat it like an export, and keep it out of anything
synced or shared.
