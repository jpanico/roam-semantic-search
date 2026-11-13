"""Phase 5a spike: measure a graph's PDF attachments before building PDF indexing.

Settles the unknowns of docs/design-plan.md, *PDF attachments*: per PDF, the decrypted size,
the Local API fetch time, the page count, how much of it carries a text layer (versus scanned
images that would need OCR), password protection, and the chunk records it would yield.

Privacy: every PDF is fetched decrypted through the Local API ``file.get`` action and held in
memory only — its bytes and extracted text are discarded as soon as they are measured. The
per-PDF results file carries numbers and the host block uid, never text, titles, or URLs (a
Firebase URL embeds an access token, which for an unencrypted upload is a read capability).

Refuses to run unless Roam Desktop's window-restore list names the graph, so it never makes
Roam open (or ask to unlock) a graph that is closed.

Run with the project venv; resumable (already-measured PDFs in ``--out`` are skipped):

    .venv/bin/python spike/phase5a_pdf_measure.py --graph brain --out <results.jsonl>
    .venv/bin/python spike/phase5a_pdf_measure.py --graph brain --out <results.jsonl> --limit 15
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pypdfium2
import regex
from guffin.model.primitives import DAILY_NOTE_UID_PATTERN
from guffin.roam.local_api_source import LocalApiGraphSource
from guffin.roam.markdown import PDF_EMBED_RE
from pydantic import HttpUrl

from roam_semantic_search.fetch import fetch_graph
from roam_semantic_search.graph_registry import api_endpoint_for, resolve_graph
from roam_semantic_search.graph_windows import open_graph_names

# Looser than guffin's PDF_EMBED_RE: any non-space, non-brace run up to the closing braces, so
# upload names carrying ' ( ) ! still match (the three misses recorded in the design plan).
LOOSE_PDF_EMBED_RE: Final[regex.Pattern[str]] = regex.compile(
    r"\{\{(?:pdf|\[\[pdf\]\]):\s*(?P<url>https://firebasestorage\.googleapis\.com/[^\s{}]+)\s*\}\}"
)
DAILY_NOTE_UID_RE: Final[regex.Pattern[str]] = regex.compile(rf"^{DAILY_NOTE_UID_PATTERN}$")

NO_TEXT_PAGE_MAX_CHARS: Final[int] = 20
"""A page with fewer non-whitespace text-layer chars than this counts as having no text layer."""

CHUNK_CHARS: Final[int] = 2000
"""Planned chunk size (design plan: a page, split past ~2,000 chars)."""

CHUNK_OVERLAP: Final[int] = 200
"""Planned overlap between consecutive chunks of one page."""


def url_key(url: str) -> str:
    """A short, non-reversible identifier for a PDF URL (never write the URL itself)."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def chunk_count(chars: int) -> int:
    """How many chunk records a page of *chars* text-layer chars would yield."""
    if chars <= CHUNK_CHARS:
        return 1
    return 1 + math.ceil((chars - CHUNK_CHARS) / (CHUNK_CHARS - CHUNK_OVERLAP))


def pdf_hosts(rows: Sequence[dict[str, object]]) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    """Every distinct PDF URL in the graph, with its first host block, plus embed-form counts."""
    page_uid_by_dbid: Final[dict[int, str]] = {}
    for row in rows:
        row_id: object = row.get("id")
        uid: object = row.get("uid")
        if "title" in row and isinstance(row_id, int) and isinstance(uid, str):
            page_uid_by_dbid[row_id] = uid

    hosts: Final[dict[str, dict[str, object]]] = {}
    counts: Final[Counter[str]] = Counter()
    for row in rows:
        string: object = row.get("string")
        uid = row.get("uid")
        if not isinstance(string, str) or not isinstance(uid, str):
            continue
        matches: list[regex.Match[str]] = list(LOOSE_PDF_EMBED_RE.finditer(string))
        if not matches:
            continue
        counts["blocks"] += 1
        counts["guffin_parsed_blocks"] += 1 if PDF_EMBED_RE.search(string) else 0
        page_stub: object = row.get("page")
        page_dbid: object = page_stub.get("id") if isinstance(page_stub, dict) else None
        page_uid: str = page_uid_by_dbid.get(page_dbid, "") if isinstance(page_dbid, int) else ""
        for match in matches:
            counts["embeds"] += 1
            url: str = match.group("url")
            if url in hosts:
                continue
            hosts[url] = {
                "host_uid": uid,
                "is_daily_note": DAILY_NOTE_UID_RE.match(page_uid) is not None,
                "encrypted": url.split("?", 1)[0].endswith(".enc"),
            }
    return hosts, dict(counts)


def measure_pdf(source: LocalApiGraphSource, url: str) -> dict[str, object]:
    """Fetch one PDF into memory, measure it, and discard it."""
    result: dict[str, object] = {}
    http_url: Final[HttpUrl] = HttpUrl(url)
    result["url_normalized"] = str(http_url) != url
    started: float = time.perf_counter()
    try:
        asset = source.fetch_file(http_url)
    except Exception as error:  # noqa: BLE001 — a spike records every failure kind
        result.update(status="fetch-failed", error=f"{type(error).__name__}: {str(error)[:200]}")
        result["fetch_s"] = round(time.perf_counter() - started, 3)
        return result
    result["fetch_s"] = round(time.perf_counter() - started, 3)
    contents: bytes = asset.contents
    result["bytes"] = len(contents)
    result["media_type"] = str(asset.media_type) if asset.media_type is not None else None

    started = time.perf_counter()
    try:
        document = pypdfium2.PdfDocument(contents)
    except pypdfium2.PdfiumError as error:
        message: str = str(error)
        result["status"] = "password" if "password" in message.lower() else "parse-failed"
        result["error"] = message[:200]
        result["extract_s"] = round(time.perf_counter() - started, 3)
        return result
    page_chars: list[int] = []
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            textpage = page.get_textpage()
            text: str = textpage.get_text_bounded()
            page_chars.append(len(text) - sum(1 for char in text if char.isspace()))
            textpage.close()
            page.close()
    except pypdfium2.PdfiumError as error:
        result.update(status="parse-failed", error=str(error)[:200])
    finally:
        document.close()
    del contents
    result["extract_s"] = round(time.perf_counter() - started, 3)
    text_pages: Final[list[int]] = [chars for chars in page_chars if chars >= NO_TEXT_PAGE_MAX_CHARS]
    result["pages"] = len(page_chars)
    result["text_pages"] = len(text_pages)
    result["chars"] = sum(text_pages)
    result["chunks"] = sum(chunk_count(chars) for chars in text_pages)
    result.setdefault("status", "ok" if text_pages else "no-text")
    return result


def evenly_spaced(items: Sequence[str], limit: int) -> list[str]:
    """*limit* items spread evenly across *items* (all of them when *limit* is 0 or larger)."""
    if limit <= 0 or limit >= len(items):
        return list(items)
    step: Final[float] = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def percentile(values: Sequence[float], fraction: float) -> float:
    """The *fraction* percentile of *values* (nearest rank)."""
    ordered: Final[list[float]] = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def summarize(results: Sequence[dict[str, object]]) -> None:
    """Print the aggregate measurements."""

    def numbers(key: str) -> list[float]:
        return [float(value) for result in results if isinstance(value := result.get(key), int | float)]

    statuses: Final[Counter[str]] = Counter(str(result.get("status")) for result in results)
    print(f"\n=== {len(results)} PDFs measured ===")
    for status, count in statuses.most_common():
        print(f"  {status:13} {count}")
    for label, key, scale, unit in (
        ("size", "bytes", 1 / 1_000_000, "MB"),
        ("fetch", "fetch_s", 1.0, "s"),
        ("extract", "extract_s", 1.0, "s"),
        ("pages", "pages", 1.0, ""),
    ):
        values: list[float] = numbers(key)
        if not values:
            continue
        print(
            f"{label:8} total {sum(values) * scale:10.1f}{unit}   median {statistics.median(values) * scale:.2f}{unit}"
            f"   p90 {percentile(values, 0.9) * scale:.2f}{unit}   max {max(values) * scale:.2f}{unit}"
        )
    pages: Final[float] = sum(numbers("pages"))
    text_pages: Final[float] = sum(numbers("text_pages"))
    measured: Final[list[dict[str, object]]] = [result for result in results if "pages" in result]
    partial: Final[int] = sum(
        1 for result in measured if 0 < int(str(result.get("text_pages", 0))) < int(str(result.get("pages", 0)))
    )
    print(f"text-layer pages  {int(text_pages)} of {int(pages)} ({100 * text_pages / pages if pages else 0:.0f}%)")
    print(f"partly scanned    {partial} PDFs (some pages without a text layer)")
    print(f"text chars        {int(sum(numbers('chars'))):,}")
    print(f"projected chunks  {int(sum(numbers('chunks'))):,}")
    print(f"url normalized    {sum(1 for result in results if result.get('url_normalized'))}")
    for flag in ("encrypted", "is_daily_note"):
        subset: list[dict[str, object]] = [result for result in results if result.get(flag)]
        scanned: int = sum(1 for result in subset if result.get("status") == "no-text")
        print(f"{flag:17} {len(subset)} PDFs, {scanned} with no text layer")


def main() -> None:
    """Measure the graph's PDFs and print the aggregate results."""
    parser: Final[argparse.ArgumentParser] = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph", required=True, help="nickname or canonical name")
    parser.add_argument("--out", type=Path, required=True, help="per-PDF results, JSON Lines (no text or URLs)")
    parser.add_argument("--limit", type=int, default=0, help="measure only this many, evenly spaced (0 = all)")
    args: Final[argparse.Namespace] = parser.parse_args()

    graph = resolve_graph(args.graph)
    open_names: Final[tuple[str, ...] | None] = open_graph_names()
    if open_names is None or graph.name not in open_names:
        sys.exit(f"{graph.name} is not in Roam Desktop's window list — refusing to make Roam open it")
    endpoint = api_endpoint_for(graph)
    source: Final[LocalApiGraphSource] = LocalApiGraphSource(api_endpoint=endpoint)

    started: Final[float] = time.perf_counter()
    rows: Final[list[dict[str, object]]] = fetch_graph(endpoint)
    hosts, counts = pdf_hosts(rows)
    print(f"graph {graph.name}: {len(rows):,} entities pulled in {time.perf_counter() - started:.1f}s")
    print(f"PDF embeds {counts.get('embeds', 0)} in {counts.get('blocks', 0)} blocks; {len(hosts)} distinct URLs")
    print(f"blocks guffin's PDF_EMBED_RE parses: {counts.get('guffin_parsed_blocks', 0)}")

    done: Final[dict[str, dict[str, object]]] = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record: dict[str, object] = json.loads(line)
                done[str(record["key"])] = record
    targets: Final[list[str]] = evenly_spaced(sorted(hosts, key=url_key), args.limit)
    pending: Final[list[str]] = [url for url in targets if url_key(url) not in done]
    print(f"measuring {len(pending)} ({len(targets) - len(pending)} already in {args.out})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_started: Final[float] = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as out:
        for position, url in enumerate(pending, start=1):
            result: dict[str, object] = {"key": url_key(url), **hosts[url], **measure_pdf(source, url)}
            out.write(json.dumps(result) + "\n")
            out.flush()
            done[str(result["key"])] = result
            if position % 25 == 0 or position == len(pending):
                elapsed: float = time.perf_counter() - run_started
                eta: float = elapsed / position * (len(pending) - position)
                print(f"  {position}/{len(pending)}  {elapsed:.0f}s elapsed, ~{eta:.0f}s left", flush=True)

    summarize([done[url_key(url)] for url in targets if url_key(url) in done])


if __name__ == "__main__":
    main()
