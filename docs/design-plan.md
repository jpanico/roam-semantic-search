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
│   ├── mcp_server.py         # stdio MCP front end
│   └── cli.py                # typer front end (build / refresh / search)
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
  SDK 2.0, `MCPServer`, stdio; entry point `roam-semantic-search-mcp`),
  configured by env vars (`GUFFIN_ROAM_GRAPH_NAME` or
  `ROAM_SEMANTIC_SEARCH_DB`). Three tools: `semantic_search` (hits + index
  meta, so callers can judge staleness), `refresh_index`, `index_stats`.
  Registered in Claude Code at user scope. The live test also exercised the
  refresh changed-path for real: 4 re-embedded, 4 deleted, 7,771 unchanged
  after a day's edits.
- **Phase 4 (optional, demand-driven):** scheduled refresh; result reranking;
  additional graphs.

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
