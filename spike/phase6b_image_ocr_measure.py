"""Phase 6b spike: measure what OCR would recover from a graph's raster image attachments.

Settles the unknowns of docs/design-plan.md, *Image attachments*, before building raster OCR
(6c): on an evenly spaced sample of a graph's embedded images (``![alt](<firebase url>)``,
PNG/JPEG/WebP/GIF/HEIC/AVIF/TIFF), how large they are, how long a fetch and an OCR take,
how much text macOS Vision recognizes, how confident it is, and whether that text reads as
words or as noise.  "Word-likeness" is the share of recognized alphabetic tokens (3+
letters) found in the system word list (/usr/share/dict/words), so nothing about the
text itself has to be read to judge it.

Privacy: every image is fetched decrypted through the Local API ``file.get`` action and held
in memory only; Vision runs on-device on the image bytes.  The per-image results file
carries numbers, the host block uid, and the file type — never recognized text, titles, or
URLs.

Refuses to run unless Roam Desktop's window-restore list names the graph, unless
``--assume-open`` is given (the list can lag a graph opened since Roam last wrote it).

Run with the project venv; resumable (already-measured images in ``--out`` are skipped):

    .venv/bin/python spike/phase6b_image_ocr_measure.py --graph brain --out <results.jsonl>
    .venv/bin/python spike/phase6b_image_ocr_measure.py --graph scfh --out <results.jsonl> --assume-open
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false

import argparse
import hashlib
import io
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import regex
import Vision
from Foundation import NSData
from guffin.model.primitives import DAILY_NOTE_UID_PATTERN
from guffin.roam.local_api_source import LocalApiGraphSource
from guffin.roam.markdown import IMAGE_LINK_RE
from PIL import Image
from pydantic import HttpUrl

from roam_semantic_search.fetch import fetch_graph
from roam_semantic_search.graph_registry import api_endpoint_for, resolve_graph
from roam_semantic_search.graph_windows import open_graph_names

RASTER_SUFFIXES: Final[tuple[str, ...]] = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".avif", ".tif", ".tiff")
CONFIDENCE_FLOOR: Final[float] = 0.5
"""A recognized line counts as text only at or above this Vision confidence."""
TEXT_MIN_CHARS: Final[int] = 20
"""Non-whitespace chars of confident text an image needs to count as carrying text."""
WORDLIKE_FLOOR: Final[float] = 0.5
"""Share of tokens found in the word list above which confident text reads as words."""
DAILY_NOTE_UID_RE: Final[regex.Pattern[str]] = regex.compile(rf"^{DAILY_NOTE_UID_PATTERN}$")
TOKEN_RE: Final[regex.Pattern[str]] = regex.compile(r"\p{L}{3,}")


def url_key(url: str) -> str:
    """A short, non-reversible identifier for an image URL (never write the URL itself)."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def suffix_of(url: str) -> str:
    """The image's file type from its object path (``.enc`` dropped), e.g. ``.jpeg``."""
    path: Final[str] = url.split("?", 1)[0].lower().removesuffix(".enc")
    return "." + path.rsplit(".", 1)[-1] if "." in path.rsplit("%2f", 1)[-1] else ""


def image_hosts(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Every distinct raster image URL the graph embeds, with its first host block."""
    page_uid_by_dbid: Final[dict[int, str]] = {}
    for row in rows:
        row_id: object = row.get("id")
        uid: object = row.get("uid")
        if "title" in row and isinstance(row_id, int) and isinstance(uid, str):
            page_uid_by_dbid[row_id] = uid
    hosts: Final[dict[str, dict[str, object]]] = {}
    for row in rows:
        string: object = row.get("string")
        uid = row.get("uid")
        if not isinstance(string, str) or not isinstance(uid, str) or "![" not in string:
            continue
        page_stub: object = row.get("page")
        page_dbid: object = page_stub.get("id") if isinstance(page_stub, dict) else None
        page_uid: str = page_uid_by_dbid.get(page_dbid, "") if isinstance(page_dbid, int) else ""
        for match in IMAGE_LINK_RE.finditer(string):
            url: str = match.group("url")
            if url in hosts or not suffix_of(url).endswith(RASTER_SUFFIXES):
                continue
            hosts[url] = {
                "host_uid": uid,
                "suffix": suffix_of(url),
                "encrypted": url.split("?", 1)[0].endswith(".enc"),
                "is_daily_note": DAILY_NOTE_UID_RE.match(page_uid) is not None,
            }
    return hosts


def recognized_lines(data: bytes) -> list[tuple[str, float]]:
    """Vision's accurate recognition of an image's bytes: each line's text and confidence."""
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
        NSData.dataWithBytes_length_(data, len(data)), None
    )
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    succeeded, _ = handler.performRequests_error_([request], None)
    if not succeeded:
        raise RuntimeError("Vision could not read the image")
    lines: Final[list[tuple[str, float]]] = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if candidates:
            lines.append((str(candidates[0].string()), float(candidates[0].confidence())))
    return lines


def measure_image(source: LocalApiGraphSource, url: str, words: frozenset[str]) -> dict[str, object]:
    """Fetch one image into memory, OCR it, measure the result, and discard it all."""
    result: dict[str, object] = {}
    started: float = time.perf_counter()
    try:
        asset = source.fetch_file(HttpUrl(url))
    except Exception as error:  # noqa: BLE001 — a spike records every failure kind
        result.update(status="fetch-failed", error=f"{type(error).__name__}: {str(error)[:160]}")
        return result
    result["fetch_s"] = round(time.perf_counter() - started, 3)
    data: bytes = asset.contents
    result["bytes"] = len(data)
    try:
        with Image.open(io.BytesIO(data)) as image:
            result["pixels"] = image.width * image.height
    except Exception:  # noqa: BLE001 — Pillow lacks HEIC/AVIF here; Vision still reads them
        result["pixels"] = None
    started = time.perf_counter()
    try:
        lines: list[tuple[str, float]] = recognized_lines(data)
    except RuntimeError as error:
        result.update(status="ocr-failed", error=str(error))
        return result
    finally:
        del data
    result["ocr_s"] = round(time.perf_counter() - started, 3)
    confident: Final[list[str]] = [text for text, confidence in lines if confidence >= CONFIDENCE_FLOOR]
    tokens: Final[list[str]] = [token.lower() for text in confident for token in TOKEN_RE.findall(text)]
    result["lines"] = len(lines)
    result["chars"] = sum(len(text.replace(" ", "")) for text, _ in lines)
    result["confident_chars"] = sum(len(text.replace(" ", "")) for text in confident)
    result["mean_confidence"] = round(statistics.fmean(c for _, c in lines), 3) if lines else None
    result["tokens"] = len(tokens)
    result["wordlike"] = round(sum(1 for token in tokens if token in words) / len(tokens), 3) if tokens else None
    has_text: Final[bool] = int(str(result["confident_chars"])) >= TEXT_MIN_CHARS
    reads_as_words: Final[bool] = has_text and (result["wordlike"] or 0) >= WORDLIKE_FLOOR
    result["status"] = "words" if reads_as_words else "noise" if has_text else "no-text"
    return result


def evenly_spaced(items: Sequence[str], count: int) -> list[str]:
    """*count* items spread evenly across *items* (all of them when there are fewer)."""
    if count >= len(items):
        return list(items)
    step: Final[float] = len(items) / count
    return [items[int(index * step)] for index in range(count)]


def percentile(values: Sequence[float], fraction: float) -> float:
    """The *fraction* percentile of *values* (nearest rank)."""
    ordered: Final[list[float]] = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def summarize(results: Sequence[dict[str, object]], population: int) -> None:
    """Print the sample's measurements and their extrapolation to the whole population."""

    def numbers(key: str, among: Sequence[dict[str, object]] = results) -> list[float]:
        return [float(value) for result in among if isinstance(value := result.get(key), int | float)]

    statuses: Final[Counter[str]] = Counter(str(result.get("status")) for result in results)
    print(f"\n=== {len(results)} images sampled of {population} ===")
    for status, count in statuses.most_common():
        print(f"  {status:13} {count:3}  ({100 * count / len(results):.0f}%)")
    for label, key, scale, unit in (
        ("size", "bytes", 1 / 1_000_000, "MB"),
        ("fetch", "fetch_s", 1.0, "s"),
        ("ocr", "ocr_s", 1.0, "s"),
        ("pixels", "pixels", 1 / 1_000_000, "MP"),
    ):
        values: list[float] = numbers(key)
        if values:
            print(
                f"{label:7} median {statistics.median(values) * scale:7.2f}{unit}"
                f"   p90 {percentile(values, 0.9) * scale:7.2f}{unit}"
                f"   max {max(values) * scale:7.2f}{unit}   mean {statistics.fmean(values) * scale:7.2f}{unit}"
            )
    worded: Final[list[dict[str, object]]] = [result for result in results if result.get("status") == "words"]
    if worded:
        print(
            f"text in 'words' images: median {statistics.median(numbers('confident_chars', worded)):.0f} chars,"
            f" median word-likeness {statistics.median(numbers('wordlike', worded)):.2f}"
        )
    measured: Final[list[dict[str, object]]] = [result for result in results if "bytes" in result]
    if measured:
        mean_bytes: Final[float] = statistics.fmean(numbers("bytes", measured))
        mean_seconds: Final[float] = statistics.fmean(numbers("fetch_s", measured)) + statistics.fmean(
            numbers("ocr_s", measured) or [0.0]
        )
        print(
            f"whole population ({population}): ~{population * mean_bytes / 1e9:.2f} GB through file.get,"
            f" ~{population * mean_seconds / 60:.0f} min fetch+OCR,"
            f" ~{population * len(worded) / len(results):.0f} images with words"
        )
    for flag in ("is_daily_note", "encrypted"):
        subset: list[dict[str, object]] = [result for result in results if result.get(flag)]
        print(f"{flag:13} {len(subset)} sampled, {sum(1 for r in subset if r.get('status') == 'words')} with words")
    print("by type:", dict(Counter(f"{r.get('suffix')}:{r.get('status')}" for r in results).most_common()))


def main() -> None:
    """Measure a sample of the graph's raster images and print the aggregate results."""
    parser: Final[argparse.ArgumentParser] = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--graph", required=True, help="nickname or canonical name")
    parser.add_argument("--out", type=Path, required=True, help="per-image results, JSON Lines (no text or URLs)")
    parser.add_argument("--sample", type=int, default=50, help="images to measure, evenly spaced")
    parser.add_argument("--assume-open", action="store_true", help="skip the window-list check")
    args: Final[argparse.Namespace] = parser.parse_args()

    graph = resolve_graph(args.graph)
    open_names: Final[tuple[str, ...] | None] = open_graph_names()
    if not args.assume_open and (open_names is None or graph.name not in open_names):
        sys.exit(f"{graph.name} is not in Roam Desktop's window list — refusing to make Roam open it")
    endpoint = api_endpoint_for(graph)
    source: Final[LocalApiGraphSource] = LocalApiGraphSource(api_endpoint=endpoint)
    words: Final[frozenset[str]] = frozenset(
        line.strip().lower() for line in Path("/usr/share/dict/words").read_text().splitlines() if line.strip()
    )

    hosts: Final[dict[str, dict[str, object]]] = image_hosts(fetch_graph(endpoint))
    print(f"graph {graph.name}: {len(hosts)} distinct raster images embedded")
    done: Final[dict[str, dict[str, object]]] = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record: dict[str, object] = json.loads(line)
                done[str(record["key"])] = record
    targets: Final[list[str]] = evenly_spaced(sorted(hosts, key=url_key), args.sample)
    pending: Final[list[str]] = [url for url in targets if url_key(url) not in done]
    print(f"measuring {len(pending)} ({len(targets) - len(pending)} already in {args.out})", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as out:
        for position, url in enumerate(pending, start=1):
            result: dict[str, object] = {"key": url_key(url), **hosts[url], **measure_image(source, url, words)}
            out.write(json.dumps(result) + "\n")
            out.flush()
            done[str(result["key"])] = result
            if position % 10 == 0:
                print(f"  {position}/{len(pending)}", flush=True)
    summarize([done[url_key(url)] for url in targets if url_key(url) in done], len(hosts))


if __name__ == "__main__":
    main()
