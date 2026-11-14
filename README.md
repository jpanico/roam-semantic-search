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
- **Attached PDFs** — each page of an attached PDF with a text layer becomes one or
  more chunk records (~2,000 chars, overlapping), set in the context of the block
  that embeds the PDF and carrying that block's uid. The text comes from a local
  cache that fetching fills (`extract-pdfs`, or `build` / `refresh` within a budget).
- **Attached SVGs** — an SVG upload a block names (as an image or a bare URL)
  contributes the words in its XML — titles, descriptions, text runs, and foreign
  objects — as an `image` record in its block's context; no OCR is involved. A bare
  Firebase Storage URL is stripped from block text, since it is never prose and it
  carries an access token.
- **Embed** — a local [Ollama](https://ollama.com) server running
  `nomic-embed-text` (768-dim), with the model's `search_document:` /
  `search_query:` retrieval prefixes. Loopback-only, enforced.
- **Store** — one SQLite file (default `~/.cache/roam-semantic-search/<graph>.db`),
  owner-only since it holds the graph's decrypted text: records + float32 embedding
  blobs, an FTS5 keyword mirror, and provenance meta.
  No SQLite extensions; vector KNN is a brute-force numpy matrix product
  (milliseconds at this scale).
- **Query** — hybrid retrieval: cosine KNN and BM25 rankings fused by reciprocal
  rank fusion, so paraphrase ("where do I argue…") and exact identifiers both rank.
  Hits collapse to the best-matching passage of each PDF, so one long PDF cannot
  fill the results.
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
- **macOS**, for OCR of scanned PDFs only: Apple's on-device Vision framework, reached
  through `pyobjc-framework-Vision` (installed with the package on macOS). Elsewhere PDFs
  are indexed from their text layers alone.

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
| `ROAM_SEMANTIC_SEARCH_REQUIRE_OPEN_WINDOW` | Truthy: never reach a graph Roam Desktop has no window open on (backs `--require-open-window`) |
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
roam-semantic-search extract-pdfs --graph brain   # cache attached PDFs' text (~17 min first time for ~1,400 PDFs)
roam-semantic-search search --graph scfh "why the human must stay responsible" -k 5
roam-semantic-search stats --graph scfh           # store provenance: model, counts, build/refresh moments
```

`--port` and `--token` are needed only to override the registry, or to reach a graph it
does not know about.

### PDF and SVG attachments: `extract-pdfs`

SVG uploads go through the same path as PDFs — fetched, cached, and indexed alongside
them — but their text is read straight from the XML, so they need no OCR.

`build` and `refresh` index attached PDFs (`--no-pdfs` leaves them out): each first
fetches the PDFs its text cache lacks, within a budget, then indexes the cached text.
`extract-pdfs` does only the fetching, for working a large backlog. Fetching takes every
PDF a block embeds (`{{pdf: …}}` / `{{[[pdf]]: …}}`) through the Local API — decrypted, for
an encrypted graph — holds it in memory only, and caches its text layer page by page in
`~/.cache/roam-semantic-search/<graph>.pdf-text.db` (`--pdf-cache` overrides, on all three
commands). That file holds
decrypted text, so it is created owner-only (`0600`); no PDF bytes are written to disk, and
entries are keyed by a digest of the URL rather than the URL (which embeds an access token).

A run skips every PDF already cached, so it resumes where the last one stopped.
`--max-pdfs N` and `--max-pdf-mb N` bound one run's fetching (default 500 MB) on all three
commands; `refresh --max-pdf-embeds N` also caps how many PDF passages it embeds, leaving
the rest for the next refresh (page and block edits are never held back). Each PDF lands as `ok`, `no-text` (a scan OCR could not read, or that no OCR run has seen), `password`,
`parse-failed`, or `fetch-failed`. A password-protected PDF is never refetched — an upload
never changes — while `--retry-failed` refetches the two failure kinds. Five fetch failures
in a row abort the run without caching them: at that point the graph, not its files, is
failing. Like `refresh`, it takes `--require-open-window`.

**Scanned pages are read by OCR.** A page with no text layer is rendered in memory and
read by macOS Vision, on-device — nothing leaves the machine, and no image is written to
disk. It costs about a second a page, so it is a CLI matter: `build`, `refresh`, and
`extract-pdfs` OCR by default (`--no-ocr` to skip), refetching cached PDFs whose blank pages
no OCR has read yet, while the MCP server's refreshes never OCR.

**Why the byte budget.** Every PDF passes through Roam Desktop as one base64 string, and on
2026-11-13 an unbudgeted first run crashed Roam Desktop after ~1.3 GB (about 1,250 PDFs).
A run that stops at its budget reports what is pending; run it again to continue.

**Work a large backlog from the CLI.** The MCP server's refreshes take only a small slice
of any PDF backlog per call (10 PDFs, 50 MB, 250 passages), so a search never waits on
minutes of PDF work; after adding many PDFs, run `extract-pdfs` and `refresh` yourself
(hippo's ~11,000 passages took ~11 minutes to embed).

### Unattended refreshes: `--require-open-window`

Reaching a graph opens it. The Local API is addressed per graph (`/api/<name>`), and a
request naming a graph that Roam Desktop has no window open on makes Roam open one — for
an encrypted graph, an unlock prompt that sits there waiting for a human. An hourly
scheduled refresh of a graph you keep closed therefore produces an hourly unlock prompt,
and the request fails anyway.

`--require-open-window` gates the run on Roam's window-restore list (`user-config.edn`,
read from disk — the check itself never touches Roam), and skips unless that list names
the graph:

```bash
roam-semantic-search refresh --graph scfh --require-open-window
# skipped: Roam Desktop has no window open on SCFH
```

A skip **succeeds** (exit 0), so a scheduler records no failure and the next run simply
retries. An unreadable list skips too — it is not a listing.

**What the flag can and cannot do.** The check is a *necessary* condition for
reachability, not a sufficient one, so it reduces prompts rather than abolishing them:

- An **unlisted** graph is certainly unservable, and skipping it is always right.
- A **listed** graph may still be closed or, for an encrypted graph, locked. The file is
  a restore list — written when the window set changes, then carried across a relaunch —
  and it records only that a window exists, never whether the graph behind it is
  unlocked. A relaunch re-locks an encrypted graph while faithfully restoring its window,
  so a stale listing sails through the gate and the request prompts anyway.

Nothing observable off the filesystem distinguishes a servable graph from a locked one:
the Local API has no graph-agnostic status route, so the request itself is the only test,
and issuing it is what raises the prompt. **A scheduled refresh of a graph you keep
locked will therefore prompt sooner or later — the durable fix is not to schedule one.**

The flag is opt-in — an unflagged `refresh` behaves exactly as it always has — because
an interactive refresh of a closed graph is a reasonable thing to ask for, and unlocking
the graph is then a deliberate answer rather than an ambush.

A hit shows the Roam uid (usable as a `((ref))`), the fused score, each ranking's
position (`v:` vector, `k:` keyword), the breadcrumb, and the text:

```
 1. ((9KMmmo5aH))  [block  score 0.0323  v:2 k:2]
    The new Programmer (in the age of AI assistants) › The human Programmer/engineer
    The human also remains the accountability boundary. The assistant can propose; ...
```

A PDF hit is labelled with its page (`[pdf p. 3  score …]`), and its breadcrumb ends with
the PDF's upload name.

## MCP server

`roam-semantic-search-mcp` serves every connected graph's index over stdio to any MCP
client, with four tools:

| Tool | Purpose |
|---|---|
| `list_indexes` | Connected graphs and which have a searchable index — call first |
| `semantic_search` | Ranked hits plus index meta; auto-refreshes a stale index first |
| `refresh_index` | Incremental refresh from the live graph, on demand |
| `index_stats` | One store's provenance |

A hit's `kind` is `page`, `block`, `pdf`, or `image`. A `pdf` hit is a passage of an
attached PDF and an `image` hit is text read from an attached SVG: its `uid` is the block
that carries the attachment, `attachment_name` names it, and `attachment_page` gives a
PDF's page (`pdf_name` / `pdf_page` repeat them for PDF hits). Each attachment appears at
most once, as its best-matching passage.

Every refresh the server runs carries a small PDF budget — 10 PDFs, 50 MB, 15 s, and 250
passage embeddings — and its summary reports `pdf_pending_count` and
`embed_deferred_count`, the PDF work left for later (also in a search's `refresh.summary`).

**Staleness is bounded, and never silent.** `semantic_search` answers from the index — a
snapshot — but when the last capture is older than the staleness threshold
(`ROAM_SEMANTIC_SEARCH_MAX_STALENESS`, default one hour) it refreshes the index before
searching. A failed refresh (most commonly Roam Desktop not running) degrades gracefully
to the snapshot. Either way the response's `refresh` field reports what happened —
`fresh`, `refreshed`, `refresh-failed` (with the error), `window-closed`, or `disabled`
— so a caller who cares about currency checks one field instead of doing timestamp
arithmetic.

Set `ROAM_SEMANTIC_SEARCH_REQUIRE_OPEN_WINDOW=1` and the *automatic* pre-search refresh
is gated the same way `--require-open-window` gates the CLI, reporting `window-closed`
and answering from the snapshot. Searching a closed encrypted graph then costs no unlock
prompt. An explicit `refresh_index` call is never gated — asking for it is deliberate,
and it is the way to say *yes, open the graph, I will unlock it*.

Register it **once per Claude client you use** — no per-graph configuration, since the
server reads Roam's registry directly. Registering the same server for two clients that
share sessions runs two copies, and only one of them answers.

- **The Claude desktop app** — its chats *and* its Code-tab sessions. Add the server to
  `~/Library/Application Support/Claude/claude_desktop_config.json`, then quit and reopen
  Claude:

  ```json
  {
    "mcpServers": {
      "roam-semantic-search": {
        "command": "/path/to/roam-semantic-search/.venv/bin/roam-semantic-search-mcp",
        "env": { "ROAM_SEMANTIC_SEARCH_REQUIRE_OPEN_WINDOW": "1" }
      }
    }
  }
  ```

- **`claude` run in a terminal** — the app's config does not reach it:

  ```bash
  claude mcp add --scope user roam-semantic-search -- $(pwd)/.venv/bin/roam-semantic-search-mcp
  ```

  Don't add this when you use Claude Code only inside the desktop app: its sessions
  already get the app's registration.

**A running server keeps the code it started with.** After changing this project, restart
whatever runs the server — quit and reopen the Claude app (⌘Q; a server that is still
connected offers no Reconnect in `/mcp`), or restart the terminal `claude` session. The
server notices when you forget: it fingerprints the source files it loaded (this project's
and guffin's) and, once any file's content has changed on disk, fails every tool call with
a message saying to restart it, rather than answering from old code. A git branch switch
or fast-forward that rewrites files with the same content does not count.

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
