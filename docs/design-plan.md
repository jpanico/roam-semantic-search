# Roam Semantic Index — Design Plan

Status: **plan** (no implementation yet). Decision log at the bottom.

A fully local semantic-search index over the SCFH Roam graph: fetch clear-text
content through the Roam Local API, embed it with a locally hosted model, store
vectors in a single SQLite file, and answer meaning-based queries through a
small MCP server (plus a CLI). Nothing about the graph's content ever leaves
the machine.

## Why

Keyword search (Roam's own, or the Roam MCP `search` tool) finds exact words,
not ideas. Roam ships an opt-in server-side embeddings feature (surfaced as the
Roam MCP `semantic_search` tool), but enabling it means Roam's servers compute
embeddings over the decrypted graph — unacceptable for SCFH, whose content is
encrypted precisely so it never leaves the author's control. **The privacy
constraint is the project's founding requirement, not a preference**: every
component must run locally, and the design must make that verifiable.

## Constraints

1. **No graph content to any external host, ever.** No cloud embedding APIs,
   no hosted vector databases, no telemetry carrying content. The only network
   egress permitted is one-time model *downloads* (which upload nothing).
2. **Roam Desktop is the only clear-text doorway.** For an encrypted graph,
   content is only readable through the Local API proxied by the running
   desktop client (guffin's `docs/roam-local-api.md`). The indexer therefore runs on
   the machine running Roam Desktop, like guffin's own commands.
3. **Results must link back to Roam.** Every hit carries the block `uid`, so a
   result is one click/`((ref))` away from its live context.
4. **Not part of guffin.** Guffin's charter is exporting pages to documents; a
   persistent vector index is a different concern with a different lifecycle.
   The indexer is a **sibling project** that depends on guffin as a library
   (its `roam/` sub-package: `local_api.py`, transitively `primitives.py`),
   the same relationship `guffin-companion` has to the server.

## Architecture

```
Roam Desktop ──(Local API, localhost HTTP)──► fetch ──► normalize ──► embed ──► store
                                                                        ▲          │
                                                          Ollama (localhost)   SQLite (FTS5 + vector blobs)
                                                                                   │
                                              MCP server (stdio) ◄── query ◄───────┘
                                              CLI (roam-semantic-search search)
```

### Fetch

A **full-graph pull**, not guffin's per-page tree fetch: every entity carrying
a `:block/uid` (pages and blocks alike), pulled with the same aliased pattern
guffin uses (guffin's `docs/roam-querying.md`, *Stripping makes distinct attributes
collide*), which includes the `:create/time`/`:edit/time` bookkeeping for
incrementality:

```datalog
[:find (pull ?e [* [:block/view-type :as "block-view-type"]
                   [:children/view-type :as "children-view-type"]])
 :where [?e :block/uid]]
```

Reuses `guffin.roam.local_api.ApiEndpoint.invoke_action` unchanged. A refresh
runs this same full pull — at ~1 s there is nothing worth narrowing (the
`[(> ?t ?since)]` predicate was validated in Phase 0 but ultimately not used;
see the decision log) — and detects deletions by diffing the fetched uid set
against the store's.

### Normalize

What gets embedded is not the raw block string:

- **Breadcrumb context.** A lone Roam block is often meaningless out of
  context ("Yes, but only for headings"). Each block embeds as
  `page title › ancestor · ancestor · … › block text`, giving the vector the
  context a human reads from the outline. Ancestor texts are truncated to keep
  the breadcrumb a prefix, not the payload.
- **Markup cleanup.** `[[...]]`/`#tags` keep their inner text; `((uid))` block
  refs resolve to their target's text (the fetch has every block, so
  resolution is a local map lookup); `{{[[TODO]]}}`-style directives and
  styling delimiters are stripped. This is a light regex pass, not guffin's
  full transcription pipeline — embedding models don't need Pandoc fidelity.
- **Skip list.** Blocks that would only pollute the index: empty/whitespace,
  pure-markup blocks (a lone image/embed/table marker), `roam/js` + `roam/css`
  pages and their subtrees, and (configurable) daily-note pages.

### Embed

A **locally hosted model via Ollama** (default `nomic-embed-text`). Ollama is
preferred over in-process `sentence-transformers` because it is a separate,
observable process — its network behaviour can be independently verified
(Little Snitch / `lsof`) — and models swap without touching the indexer's
Python environment. The model name and its vector dimension are recorded in
the index's metadata table; a model change invalidates the index and forces a
full rebuild (embeddings from different models are not comparable).

### Store

One plain SQLite database file — **no extension** (see the decision log: the host
Python's `sqlite3` is built without loadable-extension support, so `sqlite-vec` was
dropped for brute-force KNN over float32 blobs, which at ~10k vectors costs
single-digit milliseconds via one numpy matrix product):

| Table | Contents |
|---|---|
| `records` | `uid` (PK), page title, breadcrumb, normalized text, embed input, content hash, `edited_at`, `is_page`, embedding (float32 blob) |
| `records_fts` | FTS5 over the normalized text + breadcrumb — the keyword half of hybrid search |
| `meta` | embedding model + dimension, graph name, build timestamp, record count |

No server process for the store; the DB file lives under
`~/.cache/roam-semantic-search/<graph>.db` (configurable).

- **Content hash short-circuit:** `edit/time` moves on bookkeeping-only
  changes (guffin's `TRANSIENT_RAW_KEYS` lesson), so re-embedding is gated on
  the hash of the *normalized text*, not the timestamp — the timestamp only
  selects candidates; the hash decides whether the (comparatively expensive)
  embedding call happens.

### Query

**Hybrid retrieval:** vector KNN and FTS5/BM25 run side by side, fused with
reciprocal rank fusion. Pure-vector search is famously weak on exact names and
identifiers (the things an author actually remembers); BM25 is weak on
paraphrase. RRF needs no tuned weights and degrades gracefully to whichever
side has signal.

Two front ends over one query function:

- **MCP server (primary):** stdio, built on the official `mcp` Python SDK.
  One tool, `semantic_search(query, k, scope)`, returning ranked hits — uid,
  page, breadcrumb, text, score — mirroring the Roam MCP tool's shape so an
  agent session uses it the same way. Registered in Claude Code as a local
  stdio server.
- **CLI:** `roam-semantic-search build` (full), `roam-semantic-search refresh` (incremental),
  `roam-semantic-search search "..."` (human-readable ranked output).

### Refresh

On-demand (`roam-semantic-search refresh`), not daemonized, at least initially: the
incremental pass — full fetch + normalize, content-hash diff, re-embed only
changed records, delete vanished uids — takes seconds, and the MCP server can surface
index staleness (`last refresh: N days ago`) in its results rather than
pretending to be live. A scheduled refresh (launchd) is a later convenience,
not architecture.

## Project skeleton

Sibling repo **`roam-semantic-search`**, guffin conventions
inherited (Python 3.14, src layout, pyright strict, `@validate_call`,
`regex`-not-`re`, Black/Ruff at 120):

```
roam-semantic-search/
├── pyproject.toml            # deps: guffin (editable path), sqlite-vec, mcp, httpx, typer
├── src/roam_semantic_search/
│   ├── fetch.py              # full + incremental Local API pulls (thin over guffin.roam)
│   ├── normalize.py          # breadcrumb assembly, markup cleanup, skip rules
│   ├── embed.py              # Ollama client (localhost only), model/dim handshake
│   ├── store.py              # SQLite + sqlite-vec + FTS5 schema and upserts
│   ├── query.py              # hybrid retrieval + RRF (pure; no MCP/CLI deps)
│   ├── graph_registry.py     # Roam's connected graphs: nickname → name, port, token
│   ├── mcp_server.py         # stdio MCP front end
│   └── cli.py                # typer front end (graphs / build / refresh / search / stats)
└── tests/
```

## Phases

- **Phase 0 — spike: DONE 2026-08-04** (`spike/phase0_full_graph_pull.py`;
  results below). Full-graph pull against SCFH; measure block count, payload
  size, wall time. Validates the one genuinely unmeasured assumption (Local
  API behaviour on a whole-graph query) before anything is built on it.
- **Phase 1 — index + CLI search: DONE 2026-08-04.** fetch → normalize → embed
  → store, plus `search`; full rebuilds only. Usable end-to-end. Live SCFH
  build: 9,965 entities → 7,775 records (768-dim, `nomic-embed-text`) in
  ~100 s; the store is ~30 MB at `~/.cache/roam-semantic-search/SCFH.db`.
  Full check pipeline green (pyright strict, ruff, black, 31 tests).
- **Phase 2 — incremental refresh: DONE 2026-08-04.** The `refresh` command:
  full fetch + normalize (~1 s), content-hash diff against the store, re-embed
  and upsert only changed/new records, delete vanished uids, stamp
  `refreshed_at` + `record_count` into the meta. A no-op refresh of live SCFH
  answers in 1.7 s (0 re-embedded, 7,775 unchanged). Timestamp candidate
  selection was dropped entirely — see the decision log.
- **Phase 3 — MCP server: DONE 2026-08-05.** `mcp_server.py` (official `mcp`
  SDK 2.0, `MCPServer`, stdio; entry point `roam-semantic-search-mcp`).
  Three tools: `semantic_search` (hits + index meta, so callers can judge
  staleness), `refresh_index`, `index_stats`. Registered in Claude Code at user
  scope. The live test also exercised the refresh changed-path for real: 4
  re-embedded, 4 deleted, 7,771 unchanged after a day's edits. *Superseded
  2026-08-07 by the multi-graph work below: the server was configured by env
  vars (`GUFFIN_ROAM_GRAPH_NAME` / `ROAM_SEMANTIC_SEARCH_DB`) and served exactly
  one graph; it now reads Roam's registry and requires a `graph` argument.*
- **Phase 4a — additional graphs: DONE 2026-08-07.** New `graph_registry.py`
  resolves a nickname through Roam's own config files to a canonical name,
  the shared Local API port, and that graph's token. Every MCP tool takes a
  required `graph`; the CLI's `--graph` accepts a nickname or canonical name and
  derives port and token from it, leaving `--port`/`--token` as overrides. Added
  `list_indexes` (MCP) and `graphs` (CLI) so callers can discover the legal
  values. See the decision log for the three findings that shaped it, and for
  why offline graphs are excluded.
- **Phase 4b — scheduled refresh (brain): DONE 2026-08-07.** LaunchAgent
  `dev.roam-semantic-search.refresh-brain` (`~/Library/LaunchAgents/`), hourly
  (`StartInterval` 3600) plus on load: runs `refresh --graph brain`, logging to
  `~/Library/Logs/roam-semantic-search/refresh-brain.log`. No env in the plist —
  credentials resolve from Roam's registry (Phase 4a), which is what makes the
  agent this small. No `KeepAlive`: a run without Roam Desktop or Ollama fails
  cleanly and the next interval catches up. A sibling agent
  `dev.roam-semantic-search.refresh-scfh` (added later the same day) refreshes
  SCFH on the same hourly cadence, keeping the two indexes' staleness bounds in
  parity.
- **Phase 4c — don't wake a closed graph: DONE 2026-08-31.** Both agents now
  pass `--require-open-window`. The Local API is addressed per graph
  (`/api/<name>`), so a request naming a graph Roam Desktop has no window open
  on makes Roam *open* one — for SCFH, an encrypted graph, an unlock prompt
  awaiting a human. SCFH had been closed since 2026-08-29, so the hourly agent
  had raised 88 consecutive prompts, each run then failing with a Local API 500
  and hanging for tens of seconds. `graph_windows.py` reads Roam's own record of
  its open windows (`user-config.edn`, an EDN file listing each window's
  `#/app/<graph>` URL) straight off disk, so the check cannot itself provoke the
  prompt it is avoiding; the run skips with exit 0 and the next interval
  retries. `WindowState.UNKNOWN` is kept distinct from `CLOSED`, but both skip:
  the flag's promise is that a gated run never provokes a prompt, so only a
  confirmed-open window satisfies it. The same gate is available to the MCP
  server's automatic pre-search refresh via
  `ROAM_SEMANTIC_SEARCH_REQUIRE_OPEN_WINDOW` (outcome `window-closed`); the
  explicit `refresh_index` tool is deliberately never gated.
- **Phase 4d — the open-window gate was not sufficient; SCFH's agent retired:
  DONE 2026-09-07.** Phase 4c rested on a premise that is false: that a graph named in
  `:roam/windows` can be served. It cannot be relied on for two independent reasons. The
  file is a *restore* list — Roam writes it when the window set changes and then carries
  it across a relaunch — so a listing outlives the session that created it; and it records
  only that a **window** exists, never whether the encrypted graph behind one is
  **unlocked**, which is the fact that actually decides whether the Local API can serve it
  (a relaunch re-locks the graph while faithfully restoring its window). Observed: the gate
  skipped 27 runs correctly, then SCFH was opened on 2026-09-05 21:02 and entered the list;
  Roam relaunched 2026-09-06 08:34 without rewriting the file; from then on every hourly
  run read `SCFH -> open`, issued the request, and raised the unlock prompt the gate
  existed to prevent — 93 failures. Since no filesystem-observable fact distinguishes a
  servable graph from a locked one, and the request itself is the only test, **a scheduled
  refresh of a habitually-locked graph cannot be made prompt-free**. The
  `dev.roam-semantic-search.refresh-scfh` agent was therefore unloaded and its plist
  deleted (reconstructible from its `refresh-brain` sibling: same plist, `--graph scfh`,
  own log path); SCFH's index is now refreshed on demand — the MCP server's pre-search
  auto-refresh when a search finds it stale, or an explicit `refresh` / `refresh_index`.
  The gate itself is kept and its documentation corrected to claim only what it proves: an
  unlisted graph is certainly unservable (skipping is always right), a listed one merely
  might be servable. `brain`/hippo keeps its hourly agent, the graph being habitually open.
- **Scheduled refresh retired for brain too: 2026-09-24.** The same failure reached hippo:
  Roam relaunched 2026-09-24 05:18 with hippo's window in the restore list, and every hourly
  run from then on read `hippo -> open`, issued the request, and had Roam raise the hippo
  window (HTTP 500 "Promise error", six failures) — the periodic pop-up Joe noticed. Rule
  set by Joe the same day: **hippo's index is refreshed only when hippo is already open in a
  Roam window, and a refresh must never make Roam open one**; stale `brain` results are
  expected whenever hippo isn't open. Since the restore-list gate cannot tell a loaded,
  unlocked graph from a restored-but-locked window, and no permission-free live signal
  exists (`osascript` window titles need Accessibility), the only unattended caller was
  removed: `dev.roam-semantic-search.refresh-brain` unloaded via `launchctl bootout`, its
  plist moved to `~/.local/share/roam-semantic-search/refresh-brain.plist.disabled-20260924`
  for reference. Both graphs now refresh on demand only — the MCP server's pre-search
  auto-refresh (still gated by `ROAM_SEMANTIC_SEARCH_REQUIRE_OPEN_WINDOW=1` in
  `~/.claude.json`, so an unlisted graph is never touched) or an explicit `refresh` /
  `refresh_index` — which happen only while the user is present to see why a listed window
  came forward. No launchd agent remains.
- **Phase 4 (optional, demand-driven):** result reranking.
- **Phase 5 — PDF attachment text: PLANNED 2026-11-13; 5a (measurement), 5b
  (extraction), and 5c (records and search) DONE 2026-11-13; 5d (budgets) and 5e (OCR)
  DONE 2026-11-14.** Index the text inside attached PDFs — about 1,400 in
  hippo, most of them encrypted. Measurements, design, the 5a–5e sub-phases, and 5a's
  results are in *PDF attachments* below.
- **Phase 6 — image attachment text: PLANNED 2026-11-14; 6a (SVG text), 6b (raster
  measurement), and 6c (raster OCR) DONE 2026-11-14.** Index the text inside
  attached images: SVGs first (6a — their text is in the XML, no OCR needed), then,
  only if a measurement justifies it, raster images by OCR (6b measure, 6c build).
  Measurements, design, and sub-phases are in *Image attachments* below.

## Phase 0 results (2026-08-04, live SCFH)

Spike script: `spike/phase0_full_graph_pull.py` (run with guffin's venv).

| Measurement | Result |
|---|---|
| Full-graph pull (aliased `[*]` over every `:block/uid` entity) | **0.91 s**, ~4.5 MB re-serialized JSON |
| Entities | 9,965 — 633 pages, 9,315 blocks, 17 other |
| Text content | ~1.92 M chars; 70 empty-string blocks |
| Uid-only sweep (deletion detection) | 0.04 s |
| `[(> ?t ?since)]` edit-time candidates (7 days) | 0.02 s, 272 hits — the predicate works through the Local API |

Findings beyond the raw numbers:

- **Every design assumption holds with huge margin.** A full pull is
  sub-second, so even the fallback posture — full rebuild candidate scan on
  every refresh — is cheap; incrementality only has to save embedding calls,
  which the content-hash gate already does.
- **`:edit/time` alone cannot select candidates.** Coverage on SCFH:
  `:create/time` on 9,949 of 9,965 entities, `:edit/time` on only **638** —
  a block created and never re-edited carries no `:edit/time` datom. The
  incremental selector must take candidates where *either* timestamp exceeds
  the watermark (client-side union of two simple queries is fine at these
  sizes). The 11 entities carrying neither are system entities (e.g.
  graph-token records) — skip-list material regardless.
- **~2 M chars ≈ a few thousand embed calls** (block-level chunks): a full
  index build is minutes of local Ollama work, not hours.

## PDF attachments (Phase 5 plan, 2026-11-13)

The index holds no PDF content today. A PDF sits in a block as `{{[[pdf]]: <url>}}` (or
the older bare `{{pdf: <url>}}`); the full-graph pull carries only that markup, never the
file, and normalize's widget pass strips it. A block holding nothing but a PDF therefore
yields no record, and a PDF is findable only through the text around it — its block's
own words, its children, its ancestors, its page title and tags.

### Measured (2026-11-13, live hippo)

Datalog over `:block/string` with `clojure.string/includes?`, parsed with guffin's
`PDF_EMBED_RE`:

| Measurement | Result |
|---|---|
| Blocks embedding `{{[[pdf]]: …}}` | 1,388 — 1,382 distinct URLs among the 1,385 guffin parses |
| Blocks embedding `{{pdf: …}}` | 22 — unencrypted, under the graph's older `joesbrain` storage path |
| Encrypted (`.pdf.enc`) | 1,285 of those 1,382 URLs |
| On daily-note pages | 988 of 1,385 |
| Daily-note PDFs by year | 2022: 130 · 2023: 196 · 2024: 149 · 2025: 186 · 2026 (to Nov 13): 270 |
| Heaviest standing pages | Portugal 2026 (37), 2024 tax (30), Thailand 2025 (27), 2025 tax (24) |
| Missed by `PDF_EMBED_RE` | 3 (`JAhyOK80q`, `BT-cJ9znD`, `xY5Q9jDbo`): upload names carrying `'`, `(`, `)` or `!`, which the `object_path` class `[\w%.-]+` of guffin's `FIREBASE_STORAGE_URL_PATTERN` rejects |

Not measured: SCFH (not open in Roam that day; the query would have made Roam open it),
and for every graph the file sizes, page counts, text-layer yield, and password
protection — Phase 5a's job.

### Design

- **Fetch.** guffin's `LocalApiGraphSource.fetch_file` — the Local API `file.get`
  action, which returns an encrypted graph's assets decrypted. Never with a `cache_dir`:
  no decrypted PDF bytes are written to disk. A file fetch is a Local API request like
  the graph pull, so it runs only inside a build or refresh that already reached the
  graph, under the same open-window gate.
- **Text cache.** Extraction results keyed by the SHA-256 of the URL. A Firebase Storage
  URL names one immutable upload, so an entry never goes stale. It lives in its own
  SQLite file, `~/.cache/roam-semantic-search/<graph>.pdf-text.db`, because
  `write_store` deletes and recreates the store on every `build`, and a rebuild must not
  refetch ~1,400 files. Per entry: status (`ok` / `no-text` / `password` /
  `fetch-failed` / `parse-failed`), per-page text, page count, upload name
  (`recovered_upload_name`), extractor version, extracted-at. Bumping the extractor
  version re-extracts.
- **Extract.** pypdfium2 (Apache-2.0 / BSD-3), text layer only in the first cut.
  PyMuPDF is ruled out: AGPL, against this repo's MIT. *5e* adds OCR of pages without a
  text layer: macOS Vision through `pyobjc-framework-Vision` (MIT), on-device.
- **Records.** One record per chunk: a PDF page, split past ~2,000 chars with a small
  overlap. Embed input: `breadcrumb › block text › PDF "<upload name>" p. N | <chunk>`.
  Chunk text does **not** fold into ancestors' `descendant_text` — it would swamp every
  ancestor and page record and overrun `EMBED_INPUT_MAX_CHARS`.
- **Schema v3.** `records` gains a `record_id` primary key: the Roam uid for page and
  block records, `<uid>/pdf/<urlhash8>/<page>.<chunk>` for chunks. `uid` stays the host
  block's Roam uid (Constraint 3: every hit one `((uid))` away), joined by `kind` and
  `pdf_page` columns; the FTS mirror keys by `record_id`. The refresh diff keys by
  `record_id` and is otherwise unchanged — content-hash selection covers chunks as-is.
  *As built (5c):* no rebuild. A v2 record is a page or block record whose
  `record_id` is its uid, so `store.upgrade_store` migrates a v2 store in place — rows
  and FTS entries copied, embeddings kept, in one immediate transaction — and needs
  neither Roam Desktop nor Ollama. Refresh and search both run it first, so SCFH's
  index upgrades the next time it is touched, even while the graph stays closed.
- **Module placement.** `normalize.py` stays pure. A new `pdf_text.py` finds PDF embeds
  in the pulled rows, consults the cache, and fetches and extracts the misses; a pure
  `pdf_chunk_records(rows, texts)` turns cached texts into `IndexRecord`s. *As built:*
  `pdf_chunk_records` lives in `normalize.py` beside the breadcrumb machinery it reuses,
  and in 5c `build` / `refresh` only *read* the cache. Since 5d they also fetch what the
  cache lacks, within a budget (*Refresh budget*), filtered by
  `normalize.indexable_pdf_embeds` so a `--no-daily-notes` index spends no budget on PDFs
  it would not index.
- **Query.** Collapse to the best chunk per (host uid, PDF) before taking the top *k*,
  so one long PDF cannot fill the results. `SearchHit` and the MCP response gain
  `kind`, `pdf_name`, `pdf_page`. Each leg's candidate pool widens to
  `max(10k, 200)` (from `max(4k, 50)`), so the collapse still leaves *k* distinct hits
  when one PDF's chunks crowd the top of a ranking.
- **Refresh budget.** The pre-search auto-refresh must stay quick: it fetches at most N
  new PDFs or T seconds' worth, and `RefreshSummary` and the MCP `refresh` field report
  `pdf_pending`. A new `extract-pdfs --graph` command (and `build`) works the backlog.
  *As built (5d):* a `PdfFetchBudget` bounds PDFs, bytes, and seconds, checked before each
  fetch; `refresh_store` takes it plus `max_pdf_embeds`, which caps PDF chunk embeddings
  (page and block records are never deferred — an edit always lands; deferred chunks
  stay due and a later refresh embeds them). Every refresh the MCP server runs, automatic
  or `refresh_index`, uses the interactive budget — 10 PDFs, 50 MB, 15 s, 250 chunk
  embeddings — and `AutoRefresh.summary` carries `pdf_pending_count` and
  `embed_deferred_count`. The CLI's `build`, `refresh`, and `extract-pdfs` default to a
  500 MB byte budget per run (`--max-pdfs`, `--max-pdf-mb`; `refresh` also
  `--max-pdf-embeds`). A PDF fetch that fails mid-refresh — Roam gone, five failures in a
  row — is reported as `pdf_error`, and the refresh goes on with the cached text.
- **Failure posture.** A bad PDF never fails a refresh: its status is cached and decides
  when it comes back. `password` never does (an upload never changes); `fetch-failed`
  only with `--retry-failed`; `parse-failed` with `--retry-failed` or a newer extractor;
  `ok` / `no-text` with a newer extractor alone. A run of five fetch failures in a row is
  different in kind — the graph is failing (closed, or locked mid-run), not its files —
  so it aborts the run *without* caching that run, leaving those PDFs to be retried.
- **Privacy.** Fully local, like everything else. The new exposure is the text cache:
  decrypted text of tax, medical, and estate documents at rest. It is the same class as
  the store, which already holds decrypted block text — and both stores are `0644`
  today. Create the text cache `0600`, and tighten the stores to match. *Done in 5b
  (cache) and 5c (stores).* OCR (5e) keeps the posture: Vision runs on-device, a
  separate framework whose network behaviour can be checked like Ollama's, and a page
  reaches it as PNG bytes rendered in memory — no image is written to disk.
- **Daily notes.** PDF chunks follow `--no-daily-notes` like their host blocks.

### Sub-phases

- **5a — measure (spike): DONE 2026-11-13 for hippo** (results below). Fetch every PDF
  into memory only; record bytes, fetch time, pages, text-layer yield, and the
  password-protected count. Decides whether OCR belongs in the first cut and what a first
  build costs. SCFH is still to be measured, the next time it is open.
- **5b — extraction: DONE 2026-11-13.** guffin's `FIREBASE_STORAGE_URL_PATTERN` now
  admits the characters `encodeURIComponent` leaves literal (`!~*'()`), so all 1,411
  hippo embeds parse. `pdf_text.py` (embed discovery, pypdfium2 extraction, the
  owner-only text cache) and `extract-pdfs`; tests over generated fixture PDFs plus a
  qpdf-encrypted one. Live on hippo, 1,246 of 1,408 PDFs were cached (1,085 `ok`, 144
  `no-text`, 16 `password`, 1 `fetch-failed`; a 15 MB cache file) before **Roam Desktop
  crashed mid-run** — 18:34, `EXC_BREAKPOINT` in V8 string creation on Electron's main
  thread, in a process up for 7 days. The run had moved ~1.3 GB through `file.get`, on top
  of 5a's ~1.5 GB through the same process an hour earlier, and the largest PDF still
  pending is 84.6 MB. Cause unconfirmed; plausibly the main process building a ~110 MB
  base64 string under accumulated memory pressure. The cache kept every completed entry,
  as designed. After a Roam relaunch, one resumed `--limit 200` run (21:49; the option is `--max-pdfs` since 5d) fetched the
  remaining 162 in 121 s without incident, the 84.6 MB PDF included — so the file alone
  is not the trigger, and accumulated state in a long-lived process is the likelier one.
  The finished cache matches 5a exactly: 1,408 PDFs — 1,233 `ok`, 157 `no-text`, 17
  `password`, 1 `fetch-failed` — in a 16 MB file.
- **5c — records and search: DONE 2026-11-13.** Chunk records (`normalize.pdf_chunk_records`),
  schema v3 with the in-place v2 upgrade instead of a rebuild, the best-chunk collapse
  and wider candidate pools, and `kind` / `pdf_name` / `pdf_page` on CLI and MCP hits;
  `build` and `refresh` take `--pdfs/--no-pdfs` and `--pdf-cache` (now `extract-pdfs`'
  spelling too). Stores are created owner-only and an older world-readable store is
  tightened on its next refresh or search, since a store now holds PDF text as well. Live
  on hippo, one `refresh` upgraded the store and embedded **11,451 chunks in 648 s**
  (25,612 records; the store grew from 70 to 180 MB). That is more than 5a's projected
  10,433, because cutting at a word boundary in the chunk's second half yields shorter
  chunks than 5a's character arithmetic assumed. A search, query embedding included,
  answers in ~120 ms. Spot checks found the right PDF passage for an appliance-manual
  question, a hotel booking, and tax forms, one hit per PDF alongside block hits.
- **5d — refresh budget: DONE 2026-11-14.** The per-refresh fetch cap and `pdf_pending`
  reporting — and, since 5c, a cap on chunk *embeddings* per auto-refresh: after an
  `extract-pdfs` run adds many PDFs, the next pre-search refresh would otherwise embed
  every new chunk before answering (hippo's ~10,400 take ~10 minutes). Built as described
  under *Refresh budget*. Live on hippo: with nothing pending, a refresh under the
  interactive budget took 3.2 s. Against scratch copies of the store and cache with 30
  PDFs removed, one refresh fetched 10 (8.3 MB) and reported 20 pending in 15.7 s; a
  second fetched 10 more and, under a 5-chunk embed cap, embedded 5 and deferred 19.
- **5e — OCR, following 5c: DONE 2026-11-14.** macOS Vision, on-device, for pages without
  a text layer; an extractor-version bump re-extracts just those entries. 5a settled the
  "only if": 157 PDFs are fully scanned and unfindable without it. *As built:* not an
  extractor-version bump, which would have refetched every PDF, but an OCR version of
  its own. The cache (schema v2, migrated in place) records which `OCR_VERSION` read
  each PDF's blank pages, and a run with OCR on refetches exactly the cached PDFs that
  still have blank pages no current OCR has read — a page OCR could not read either
  stays blank, and its PDF does not come back. A page renders at 216 dpi (longest side
  capped at 4,000 px), goes to Vision's accurate recognizer as in-memory PNG bytes, and
  its lines are ordered top to bottom, then left to right. OCR is opt-in per run because
  it costs about a second a page: the CLI's `build`, `refresh`, and `extract-pdfs` do it
  by default (`--ocr/--no-ocr`), and the MCP server's refreshes never do. Tests cover it
  on a committed image-only fixture (`tests/fixtures/scanned.pdf`, made with `sips`).
  Live on hippo, two `extract-pdfs` runs under the 500 MB default read all 246 PDFs with
  blank pages: 145 (506 MB, 442 pages by OCR, 487 s), then the last 101 (370 MB, 548
  pages, 543 s). Every one now has text — no `no-text` PDF remains — and 990 of the
  1,140 blank pages were readable; Roam Desktop stayed up through the 876 MB. One
  `refresh` then embedded the 1,466 new passages in 91 s (27,079 records). Paraphrased
  queries found a scanned church giving letter, two scanned wills, and a scanned
  immunization form first. A PDF embedded in two blocks surfaces once per block — by
  design, since each block is its own citation.

### Phase 5a results (2026-11-13, live hippo)

Spike script: `spike/phase5a_pdf_measure.py` (project venv; resumable; refuses to run
unless the window-restore list names the graph). The full-graph pull found 1,411 blocks
embedding a PDF, 1,408 distinct URLs.

| Measurement | Result |
|---|---|
| PDFs | 1,408 — 1,288 encrypted, 1,002 on daily-note pages |
| Outcome | 1,233 with text · 157 no text layer · 17 password-protected · 1 fetch failed |
| Pages | 9,068; **7,928 (87%) with a text layer**; median 2 per PDF, p90 11, max 338 |
| Decrypted size | 1.51 GB; median 0.19 MB, p90 1.94 MB, max 117.9 MB |
| `file.get` fetch | **1,026 s total (~17 min)**; median 0.55 s, p90 0.99 s, max 27.1 s; ~1.5 MB/s |
| pypdfium2 extraction | 26.6 s total; median 0.01 s |
| Text-layer chars | 11.34 M |
| Projected chunk records | **10,433** (2,000-char chunks, 200-char overlap) |

Findings beyond the raw numbers:

- **OCR is not a prerequisite, but not optional for coverage.** 1,140 pages carry no
  text layer: 957 across the 157 fully scanned PDFs (101 of them on daily notes), 183 in
  89 partly scanned ones. A text-layer-only first cut covers 87% of pages and leaves
  11% of PDFs unfindable, so 5e follows 5c rather than waiting on demand.
- **Chunks concentrate in a few large PDFs.** The 10 largest yield 25% of all chunks,
  the top 50 yield 52%; the longest (338 pages) alone yields 542. The best-chunk-per-PDF
  collapse is required, not a refinement.
- **Index growth is as predicted:** hippo goes from 14,162 to ~24,600 records, and the
  search-time matrix from ~44 MB to ~75 MB.
- **The refresh budget matters only for the first build.** Daily-note PDFs arrive at
  under one a day in 2026, at ~0.5 s each; the ~17-minute backlog is a one-off for
  `build` / `extract-pdfs`.
- **guffin's pattern is the only obstacle for the three odd upload names.** `file.get`
  served all three (`ok`), and pydantic's `HttpUrl` left every one of the 1,408 URLs
  unchanged.
- **The one fetch failure** (`Ahg2ca1kK`) is a Local API 500, "Unable to get file
  metadata" — most likely an upload that no longer exists; `fetch-failed` covers it.
- **Password protection is terminal.** All 17 report PDFium's "Incorrect password";
  `password` is a final status, not a retry candidate.

### Risks

- **`file.get` can crash Roam Desktop** (5b's live run). Each fetch hands the whole file
  to Roam's Electron main process as one base64 string. The crash followed ~2.8 GB of such
  traffic through a week-old process, while a fresh process then served the largest
  pending file without trouble, so volume per process looks like the trigger rather than
  any one file. Until the cause is known, work a backlog in batches, ideally soon after a
  Roam relaunch. *5d:* every fetching command now caps bytes per run — 500 MB from the
  CLI, 50 MB from the MCP server — so batching is the default rather than advice.
- **Search-time matrix load.** Every search loads the whole embedding matrix — hippo's
  is ~44 MB at 14,162 records today; 5a's 10,433 chunks would take it to ~75 MB. If
  that shows in search latency, keep the matrix resident in the MCP server between
  searches.
- **Embedding time** is now the largest unmeasured cost (5a measured fetching). Chunks
  average ~1,100 chars, far longer than a block, so records/s understates it; scaling
  SCFH's Phase 1 build (~1.9 M chars in ~100 s) puts 11.3 M chars on the order of ten
  minutes.

## Image attachments (Phase 6 plan, 2026-11-14)

The index holds no text from attached images. An image embeds as `![alt](<url>)`, and
normalize keeps only the alt text, so an image with empty alt text — the common case —
leaves its block with no text and no record; a bare Firebase URL in a block is indexed
as the URL string itself. No image file is fetched, and Phase 5's OCR runs on PDF pages
only.

### Measured (2026-11-14, live hippo and SCFH)

Every file was fetched decrypted into memory and discarded; only counts were kept.

| Measurement | Result |
|---|---|
| SVG uploads | hippo none; SCFH 9 distinct, in 10 blocks — 8 image embeds with empty alt text (no record today), 2 bare URLs (indexed as URL text) |
| SVG fetches | 8 of 9; the ninth failed in Roam (`file.get` 500, "Failed to fetch") |
| SVG structure | all 8 carry real `<text>` elements (4–52 each, some with `<tspan>`), most a `<title>` and `<desc>`; none outline their text, none use `<foreignObject>` |
| SVG text | 184–2,937 chars each, ~10,000 in all; 5–23 KB per file |
| Raster images embedded | hippo 1,354 (742 JPEG, 357 PNG, 246 JPG, 9 other); SCFH 352 (146 JPEG, 104 JPG, 91 PNG, 11 other) |

### Design (6a — SVG text)

- **Discover.** Any Firebase Storage URL in a block whose object path ends `.svg` or
  `.svg.enc` — as an image embed or bare. Fetched like a PDF (decrypting `file.get`,
  in memory only), under the same `PdfFetchBudget`.
- **Cache.** The PDF text cache becomes an attachment text cache: schema v3, migrated
  in place, adds a `media` column (`pdf` / `svg`); the file keeps its name.
- **Extract.** Parse the XML with the standard library (`xml.etree.ElementTree`; it
  resolves no external entities, and the expat bundled with Python 3.14 guards against
  entity-expansion bombs). Collect, in document order, the text of `<title>`, `<desc>`,
  `<text>` (with its `<tspan>` / `<textPath>`), and `<foreignObject>` — the last for
  Mermaid and draw.io exports, which put labels there — skipping `<style>`, `<script>`,
  and `<metadata>`. Cleaned like a PDF page and kept as a one-page document; no text is
  `no-text`, unparseable XML `parse-failed`. Outlined-text SVGs would need rasterizing
  for OCR; none of SCFH's do, so that stays out of scope.
- **Records.** A new `RecordKind.IMAGE` ("image") with record ids
  `<uid>/img/<key8>/1.<chunk>`; embed input is the host block's breadcrumb, then
  `SVG "<upload name>"`, then the text. Search collapses to one hit per image, as per PDF.
- **Hit fields.** Hits gain a general `attachment_name` (and `attachment_page`, PDFs
  only); `pdf_name` / `pdf_page` stay as duplicates so no consumer breaks.
- **Bare URLs.** Normalize strips bare Firebase Storage URLs from block text, so the two
  URL-only blocks stop embedding a URL string — and an access token stops reaching the
  index's text.
- **Cost.** Nine fetches (~60 KB) and about ten records — within the interactive budget.

### Sub-phases

- **6a — SVG text: DONE 2026-11-14.** As designed above; tests over fixture SVGs (plain
  `<text>`, `<tspan>`, `<foreignObject>`, no text, malformed); live on SCFH. *As built:*
  SVGs ride the PDF pipeline under its existing names — `pdf_text.svg_embeds` and
  `extracted_svg_text`, a `Media` value on embeds and cache entries, the cache's schema
  v3 adding `media` (migrated in place), `extract-pdfs` fetching SVGs too — while
  normalize's `attachment_records` and `indexable_attachment_embeds` replace the
  PDF-only names. An SVG cache entry never counts as due for OCR. A test pins that an
  external entity in an SVG is not resolved. Live on SCFH: 8 of the 9 SVGs cached
  (~10,400 chars; the ninth again `fetch-failed` in Roam), yielding 12 image records;
  the refresh embedded 51 records and removed 19 — blocks that held nothing but a bare
  Firebase URL. hippo's refresh removed 103 such blocks. Queries about the
  stored-program-computer diagram and the programmer–AI triangle found the SVGs' own
  text, as `image` hits on the blocks that carry them.
- **6b — measure raster OCR: DONE 2026-11-14** (results below). A sample of ~50 images
  per graph: how many carry readable text, bytes per image, OCR time. Photos often carry
  none, and hippo's 1,354 images could be gigabytes through `file.get` (the 5b crash
  risk).
- **6c — raster OCR, only if 6b justifies it.** Reuse `ocr.py`'s Vision path on the
  image itself, with a recognition-confidence threshold so photos do not index junk,
  under the existing byte budgets. *6b justifies it for hippo* (below); the junk filter
  must be word-likeness, not confidence alone. *DONE 2026-11-14, as built:*
  `pdf_text.raster_embeds` finds any Firebase upload with a raster extension (PNG, JPEG,
  WebP, GIF, HEIC, AVIF, TIFF; embedded, linked, or bare) as `Media.RASTER`;
  `ocr.readable_image_text` runs Vision on the image's own bytes, drops lines below
  confidence 0.5, and keeps the rest only when `word_likeness` — the share of 3+-letter
  tokens in `/usr/share/dict/words` — reaches 0.5 (that list, web2, lacks many inflected
  forms, so it understates real text; 6b calibrated the 0.5 against it). A raster image
  is fetched only by a run with OCR on — the CLI's default, never the MCP server's —
  and comes back only for a newer `OCR_VERSION`; its text is an `image` record labelled
  `Image "<upload name>"`. *Live, after a Roam relaunch:* hippo's 1,355 raster uploads
  took two budgeted runs (500 MB + 352 MB, 735 s + 494 s): 798 carry readable text, 556
  none, and one `fetch-failed` ("Unable to get file metadata") — against 6b's projection
  of ~680 worded images, ~0.87 GB, and ~22 min. SCFH's 353 (183 MB): 73 with text, 279
  none, one failed. That failure is the run's lesson: one `file.get` hung for 3,600 s
  until Roam answered "Response timeout", so SCFH's run took 64 min where its other 352
  images took about 4. The fetch budget's `max_seconds` is checked only between fetches,
  and guffin's `invoke_action` set no client timeout, so a stuck fetch was bounded only by
  Roam's hour (fixed the same day: decision log, "Every fetch has a timeout"). Roam
  Desktop stayed up through ~1.04 GB of `file.get`. The refreshes
  embedded 906 image records for hippo (41 s; long screenshots chunk) and 78 for SCFH
  (4 s). The junk filter held: of the 871 images kept, about 20 carry two or fewer
  dictionary words, and nearly all of those are real labels — part numbers,
  measurements, a product name, "Shoggoth (Artificial Intelligence)"; outright noise
  (a Midjourney hedge maze read as `all/c1 Hlel008 …`) is a case or two, so the gate
  stays as built rather than costing the short labels. Searches found what only the
  pixels held: "MacBook Air 15-inch" ranked four screenshots and photos of About-This-Mac
  and order pages first through fourth; "health care proxy and living trust paperwork"
  ranked a photo of a labelled folder second, between two scanned PDFs; and on SCFH a
  screenshot's caption ranked second for "shoggoth monster meme about AI".

### Phase 6b results (2026-11-14, live hippo and SCFH)

Spike script: `spike/phase6b_image_ocr_measure.py` (project venv; resumable; in-memory
fetches; Vision on the image bytes, no rendering; `--assume-open` skips the window-list
check, which lagged SCFH). An evenly spaced sample of 50 embedded raster images per
graph. An image "has words" when Vision recognizes at least 20 non-whitespace chars at
confidence ≥ 0.5 *and* at least half of its alphabetic tokens (3+ letters) are in the
system word list (`/usr/share/dict/words`); text failing only the second test is
"noise".

| Measurement | hippo (1,354 images) | SCFH (352 images) |
|---|---|---|
| Words / noise / no text | 25 / 2 / 23 (50% / 4% / 46%) | 8 / 5 / 37 (16% / 10% / 74%) |
| Text in a "words" image | median 975 chars (p10 59), word-likeness 0.77 | median 68 chars (p10 39), word-likeness 0.79 |
| Size | median 0.32 MB, p90 1.68, max 3.60 | median 0.23 MB, p90 0.83, max 1.86 |
| Fetch / OCR | median 0.60 s / 0.20 s | median 0.55 s / 0.09 s |
| Whole graph, projected | ~0.87 GB, ~22 min, ~680 images with words | ~0.12 GB, ~4 min, ~56 images with words |

No fetch or OCR failed in either sample.

Findings:

- **hippo justifies 6c; SCFH comes along cheaply.** Half of hippo's images are
  screenshots and scanned documents carrying page-length text (median ~1,000 chars) —
  roughly 680 images now unfindable but by their surroundings. SCFH's are mostly the
  book's illustrations; its ~56 worded images carry short labels, but at 0.12 GB and four
  minutes there is no reason to exclude them.
- **Confidence does not separate noise; word-likeness does.** Noise images reached
  mean confidence 0.6–1.0 — Vision is confidently wrong on textures and scenes — but
  their word-likeness was 0.0–0.4, against 0.5–0.9 for real text. 6c should gate on
  word-likeness (with a confidence floor per line), which on macOS can use the same
  system word list.
- **Cost is fetch, not OCR.** Vision on the raw bytes takes a fraction of a second
  (median 0.1–0.2 s), so a full pass is the ~1 GB of fetches — two 500 MB budgeted runs.
- **Mind the per-process crash risk.** This Roam Desktop process has moved ~1.5–2 GB
  since its relaunch; another ~1 GB approaches the ~2.8 GB at which the 5b run crashed
  it. Relaunch Roam before 6c's bulk pass.

6a stands on its own: if raster images are not worth it, 6b and 6c are skipped.

## Open questions

1. **Model choice.** `nomic-embed-text` is the default candidate; worth a
   small side-by-side (e.g. vs `mxbai-embed-large`) on real SCFH queries
   during Phase 1.
2. **Daily notes in or out?** RESOLVED 2026-08-05: in by default
   (`--no-daily-notes` opts out). Daily notes turned out to hold real project
   content — e.g. Emi's thumbnail-sketch reviews — not just journal noise, and
   their absence made well-matched blocks unfindable.
3. **Scope of block-ref resolution in normalize** — resolve one level only,
   or recursively with a depth cap?

## Decision log

- **2026-11-14 — Every fetch has a timeout.** In 6c's SCFH run one `file.get` went
  unanswered for 3,600 s, until Roam Desktop gave up with "Response timeout": guffin's
  `invoke_action` posted with no client timeout, and the fetch budget's `max_seconds` was
  checked only between fetches, so neither bounded a stuck fetch — an MCP refresh's "15 s"
  included. guffin's `invoke_action` and `GraphSource.fetch_file` now take a `timeout`
  (default `None`, the old behavior; the snapshot replay source ignores it), and every
  attachment fetch passes `FETCH_TIMEOUT_SECONDS` = 120 — over four times the slowest fetch
  seen, a 118 MB PDF at ~27 s — or the budget's time left if that is less
  (`PdfFetchBudget.fetch_timeout`). A fetch that outwaits the full 120 s is cached as
  `fetch-failed` (refetched by `--retry-failed`) and counts toward the five-in-a-row abort;
  one the budget cuts short stops the run with that attachment left pending, not failed, so
  a large PDF the 15 s interactive budget cannot fetch waits for a CLI run rather than
  being written off. Roam may still be working on a request the client abandoned; whether
  that slows the requests after it is not yet observed.
- **2026-11-14 — A server refuses to serve once its code changes.** A running MCP server
  keeps the code it started with, and the Claude app keeps reporting it healthy (✓, no
  Reconnect in `/mcp`). After 5c changed the store schema, the app's still-running copies
  failed every hippo search with `no such column: uid` — the error naming a symptom, not
  the cause. The server now fingerprints every source file it loaded from this package
  and guffin — modification time, size, and SHA-256 of the content — re-checks them
  before each tool call, and once any file's content has changed fails the call with a
  message saying to restart it (quit and reopen Claude, or the terminal `claude`).
  Refusing rather than warning is deliberate: old code reading an index or cache newer
  code wrote can answer wrongly as well as fail. The first version compared only
  `(mtime, size)` and tripped within minutes: fast-forwarding `main` from a feature branch
  rewrote `mcp_server.py` with identical content and a new time. A file whose time or size
  moved is now read and its content compared, so only a real change counts. Found the same day:
  the server was registered both in the Claude app's config and in `~/.claude.json`, so
  each Code-tab session started an extra copy that never answered; the user-scope
  registrations were removed and the README now says to register once per client.

- **2026-08-07 — Search-time staleness bound (auto-refresh), not refresh-on-every-search.**
  `semantic_search` now refreshes the store first *only* when its last capture is older
  than `ROAM_SEMANTIC_SEARCH_MAX_STALENESS` (default 3600 s; negative disables). Refreshing
  on every search was rejected: it would add seconds to every query and make search
  *require* Roam Desktop, whereas the snapshot answers in milliseconds without it. A failed
  auto-refresh degrades to the snapshot rather than erroring — search's availability must
  not inherit refresh's dependencies — and every response carries a `refresh` field
  (`fresh` / `refreshed` / `refresh-failed` + error / `disabled`) so staleness is visible
  rather than silent. An unknown capture age (unparseable meta timestamp) is treated as
  stale. The CLI `search` is unchanged: scripted use composes `refresh` explicitly.

- **2026-08-07 — Multi-graph, addressed by nickname; no local config.** Several
  graphs are connected at once (SCFH, hippo, Apple), so a single-graph server
  was the wrong shape. Graph identity, the shared Local API port, and per-graph
  bearer tokens are read from Roam's own files — `~/.roam-local-api.json`
  (`{"port": 3333}`) and `~/.roam-tools.json` (a `graphs` array) — via the new
  `graph_registry` module, so this project stores no graph configuration and
  cannot drift from Roam's. Three findings drove the shape:
  - **One port serves every graph.** The port lives in its own file and is
    app-wide; the graph is named by the request path (`/api/<name>`), not by a
    port of its own. An earlier assumption of per-graph ports was wrong.
  - **Registry tokens are ordinary Local API tokens**, interchangeable with
    `GUFFIN_ROAM_API_TOKEN` (a graph may have several minted; all work).
    Verified by `data.q` returning HTTP 200 under each, against a 401 control.
    So the registry is preferred over the environment variable, which names one
    default graph and whose token is the wrong credential for any other.
  - **Stores key off the canonical name, never the nickname**
    (`~/.cache/roam-semantic-search/hippo.db` for nickname `brain`), so the CLI
    and MCP server address the same file for the same graph.
- **2026-08-07 — The MCP tools require an explicit `graph`; no default.** A
  server-wide default would silently answer from whichever graph the process was
  configured for — a wrong answer indistinguishable from a right one, and the
  connected graphs are comparable in size (SCFH 10,127 entities, hippo 15,334),
  so nothing about a result would reveal the mistake. Requiring the argument
  makes the failure mode "you must choose" instead of "you were answered from
  the wrong graph". `list_indexes` (MCP) and `graphs` (CLI) exist so a caller can
  discover the legal values rather than guess. Consequence: the server no longer
  reads `GUFFIN_ROAM_GRAPH_NAME` or `ROAM_SEMANTIC_SEARCH_DB`.
- **2026-08-07 — Offline graphs are out of scope for now.** A registry entry of
  `type: offline` (local-only storage) rejects the `/api/<name>` path with
  *"Token is valid for offline graph … not hosted"*; the correct path is
  undocumented and probing `/api/offline/<name>` and neighbours returned 404.
  `graph_registry.api_endpoint_for` refuses such a graph with that explanation
  rather than letting it surface as an opaque 401. Roam's own MCP server reaches
  offline graphs, so the answer is recoverable from its source if this becomes
  worth supporting.
- **2026-08-05 — Concept-weighted indexing (schema v2).** Words are not equal:
  a `[[Page]]`/`#tag` reference in a block's own text names a well-identified
  concept (highest weight), a direct-child `tags::` value is the user's explicit
  classification of the block (middle), and everything else — own text plus the
  whole descendant subtree's folded words, full depth — is base weight. The
  keyword leg realizes the tiers as FTS5 per-column BM25 weights (4/2/1); the
  vector leg by embed-input composition (labeled `concepts:`/`tags:` segments,
  descendant text last so the length cap cuts it first). Motivating case: the
  Hedge Maze thumbnail-sketch review `((JOqXHpSTE))` — `[[Emi]]` ref +
  `tags:: #Illustration` child — ranked #239 vector / #218 keyword for
  "Emi illustration illustrator" and was unfindable; after the change it fuses
  into the top dozen. Requires a full `build` (the refresh guard rejects a
  mismatched `schema_version`).
- **2026-08-04 — Roam server-side embeddings rejected.** Roam's opt-in
  embeddings feature would have Roam's servers compute embeddings over
  decrypted SCFH content. Declined on privacy grounds; this project exists
  because of that decision.
- **2026-08-04 — Sibling project, not a guffin sub-package.** Guffin's charter
  is export; the index has its own lifecycle and store. Depends on guffin as a
  library, reusing the `roam/` fetch layer.
- **2026-08-04 — Ollama over in-process sentence-transformers.** A separate
  localhost process is independently observable (network monitoring can prove
  no egress) and models swap without touching the Python env.
- **2026-08-04 — sqlite-vec over Chroma/LanceDB.** One file, no server, FTS5
  in the same store enables hybrid search with a single join.
- **2026-08-04 — Hybrid retrieval (RRF) from the start.** Author queries mix
  paraphrase ("where do I argue…") with exact identifiers; neither pure
  vector nor pure BM25 covers both.
- **2026-08-04 — Brute-force numpy KNN instead of sqlite-vec.** The host
  Python 3.14 build ships `sqlite3` without loadable-extension support
  (`Connection.enable_load_extension` absent), so `sqlite-vec` cannot load.
  At ~10k vectors a single float32 matrix product answers in milliseconds;
  embeddings live as blobs in the `records` table. Revisit only if the index
  grows orders of magnitude.
- **2026-08-04 — Loopback enforcement in `embed.py`.** The embedding client
  refuses any non-loopback server URL (`ValueError`), so the no-egress
  guarantee is enforced in code, not merely assumed.
- **2026-08-04 — nomic-embed task prefixes.** Documents embed under
  `search_document:` and queries under `search_query:`, per the model's
  retrieval convention; omitting them measurably degrades nomic-embed
  retrieval quality.
- **2026-08-04 — Page titles: raw for display, cleaned for embedding.** A
  record keeps the raw title as its display identity, but breadcrumbs and
  embed inputs use the markup-cleaned form (`[[Programmer]]` → `Programmer`)
  so link chrome never pollutes the vectors.
- **2026-08-04 — Refresh selects by content hash alone; timestamps dropped.**
  The planned `create/edit`-time candidate selection was abandoned, for two
  measured reasons. It is *unnecessary*: Phase 0 showed a full fetch +
  normalize costs ~1 s, so there is nothing worth skipping ahead of the
  embed step. And it is *insufficient*: a record's embeddable input contains
  other entities' text (ancestors reach descendants' breadcrumbs; a
  referenced block's text reaches its referrers), so an edit moves hashes on
  records whose own timestamps never changed — a timestamp selector would
  silently leave their embeddings stale. The hash diff over the full record
  set is both simpler and the only correct selector; timestamps remain in
  the store as bookkeeping only.
