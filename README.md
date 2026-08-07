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
  Each record also carries **retrieval emphasis** in three weight tiers: the page
  names its own text references (`[[Page]]` and `#tag` alike — its *concepts*,
  highest), its direct-child `tags::` values (its *tags*, middle), and its plain
  words plus its whole subtree's folded text (base). The keyword leg realizes the
  tiers as per-column BM25 weights (4/2/1); the vector leg by embed-input
  composition (labeled concept/tag segments, descendant text truncated first).
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

- **Roam Desktop** running locally with the Local API enabled, and each graph you
  want to index **connected** (Roam records the connection; see *Graphs* below)
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

## Graphs

Several Roam graphs are typically connected at once, so **every command names the graph
it operates on**. A graph is named by the nickname it was connected under, or by its
canonical Roam name:

```bash
roam-semantic-search graphs             # what's connected, and what's indexed
```
```
scfh         SCFH         hosted   indexed (9392 records)
brain        hippo        hosted   not indexed
apple        Apple        offline  not indexed
```

Nicknames, canonical names, per-graph bearer tokens, and the Local API port are read from
Roam's own config files, so this project stores no graph configuration of its own:

| File | Contents |
|---|---|
| `~/.roam-local-api.json` | `{"port": 3333}` — **one port serves every graph**, not one per graph |
| `~/.roam-tools.json` | A `graphs` array of `{name, nickname, token, type, …}`, written when a graph is connected |

Each index lives at `~/.cache/roam-semantic-search/<canonical-name>.db` — keyed by the
canonical name, never the nickname, so the CLI and MCP server always address the same file.

**Offline graphs** (registry `type` of `offline` rather than `hosted`) cannot be indexed:
the `/api/<name>` Local API path serves hosted graphs only, and rejects them with *"Token
is valid for offline graph … not hosted"*. They appear in `graphs` but `build` refuses
them with that explanation.

## Configuration

Graph identity and credentials come from the files above. The remaining environment
variables tune everything else, and are all optional:

| Variable | Meaning |
|---|---|
| `ROAM_SEMANTIC_SEARCH_OLLAMA_URL` | Embedding server URL (default `http://127.0.0.1:11434`; must be loopback) |
| `ROAM_SEMANTIC_SEARCH_MAX_STALENESS` | MCP `semantic_search` auto-refresh threshold, seconds (default 3600; negative disables) |
| `GUFFIN_ROAM_LOCAL_API_PORT` | Overrides the port from `~/.roam-local-api.json` (backs `--port`/`-p`) |
| `GUFFIN_ROAM_API_TOKEN` | Overrides the graph's registry token (backs `--token`/`-t`) |
| `GUFFIN_ROAM_GRAPH_NAME` | Default value for `--graph`; a nickname or canonical name |
| `ROAM_SEMANTIC_SEARCH_DB` | Explicit index DB path for a CLI command (backs `--db`) |

An explicit flag always wins, then the registry, then the environment. The registry is
preferred over `GUFFIN_ROAM_API_TOKEN` deliberately: that variable names one default
graph, and its token is the wrong credential for any other.

## CLI

```bash
roam-semantic-search graphs                       # connected graphs and their index state
roam-semantic-search build --graph brain          # full fetch → normalize → embed → store (~100 s for ~8k records)
roam-semantic-search refresh --graph brain        # incremental: re-embed only what changed (~2 s when idle)
roam-semantic-search search --graph scfh "why the human must stay responsible" -k 5
roam-semantic-search stats --graph scfh           # store provenance: model, counts, build/refresh moments
```

`--port` and `--token` are needed only to override the registry, or to reach a graph it
does not know about.

A hit shows the Roam uid (usable as a `((ref))`), the fused score, each ranking's
position (`v:` vector, `k:` keyword), the breadcrumb, and the text:

```
 1. ((9KMmmo5aH))  [block  score 0.0323  v:2 k:2]
    The new Programmer (in the age of AI assistants) › The human Programmer/engineer
    The human also remains the accountability boundary. The assistant can propose; ...
```

## MCP server

`roam-semantic-search-mcp` serves every connected graph's index over stdio to any MCP
client, with four tools:

| Tool | Purpose |
|---|---|
| `list_indexes` | Connected graphs and which have a searchable index — call first |
| `semantic_search` | Ranked hits plus index meta; auto-refreshes a stale index first |
| `refresh_index` | Incremental refresh from the live graph, on demand |
| `index_stats` | One store's provenance |

**Staleness is bounded, and never silent.** `semantic_search` answers from the index — a
snapshot — but when the last capture is older than the staleness threshold
(`ROAM_SEMANTIC_SEARCH_MAX_STALENESS`, default one hour) it refreshes the index before
searching. A failed refresh (most commonly Roam Desktop not running) degrades gracefully
to the snapshot. Either way the response's `refresh` field reports what happened —
`fresh`, `refreshed`, `refresh-failed` (with the error), or `disabled` — so a caller who
cares about currency checks one field instead of doing timestamp arithmetic.

Register with Claude Code — no per-graph configuration, since the server reads Roam's
registry directly:

```bash
claude mcp add --scope user roam-semantic-search -- $(pwd)/.venv/bin/roam-semantic-search-mcp
```

**`semantic_search`, `refresh_index`, and `index_stats` each require a `graph`
argument.** That is deliberate rather than an ergonomic oversight: with several graphs
connected, a server-wide default would silently answer from whichever graph the process
happened to be configured for — a wrong answer that reads exactly like a right one.
`list_indexes` exists so a caller can discover the legal values instead of guessing.

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
