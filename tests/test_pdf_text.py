"""Tests for roam_semantic_search.pdf_text."""

import sqlite3
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Final

import pytest
import requests
from guffin.roam.asset import RoamAsset
from guffin.roam.graph_source import DatalogQuery, DatalogRows, GraphSource
from pydantic import ConfigDict, Field, HttpUrl

from roam_semantic_search import pdf_text
from roam_semantic_search.ocr import OCR_VERSION, ocr_available
from roam_semantic_search.pdf_text import (
    CONSECUTIVE_FETCH_FAILURE_LIMIT,
    ExtractedText,
    ExtractionSummary,
    PdfEmbed,
    PdfFetchBudget,
    PdfStatus,
    PdfText,
    cached_pdf_texts,
    extract_pdf_texts,
    extracted_text,
    pdf_embeds,
    url_key,
)

FIXTURES: Final[Path] = Path(__file__).parent / "fixtures"

_URL_PREFIX: Final[str] = "https://firebasestorage.googleapis.com/v0/b/test.appspot.com/o/imgs%2Fapp%2Ftest%2F"
_URL_QUERY: Final[str] = "?alt=media&token=abc123"

_LINE_ONE: Final[str] = "Invoice for the garden shed, paid in full"
_LINE_TWO: Final[str] = "Warranty covers the roof panels for ten years"


def pdf_bytes(page_texts: Sequence[str]) -> bytes:
    """A minimal valid PDF: one Helvetica line per page, a page with ``""`` left blank.

    Texts must be plain ASCII without parentheses or backslashes (no string escaping).
    """
    page_count: Final[int] = len(page_texts)
    kids: Final[str] = " ".join(f"{4 + 2 * index} 0 R" for index in range(page_count))
    bodies: Final[list[bytes]] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, text in enumerate(page_texts):
        content: bytes = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        bodies.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            f" /Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * index} 0 R >>".encode()
        )
        bodies.append(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
    out: Final[bytearray] = bytearray(b"%PDF-1.4\n")
    offsets: Final[list[int]] = []
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_offset: Final[int] = len(out)
    out.extend(f"xref\n0 {len(bodies) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode())
    return bytes(out)


def url(name: str) -> str:
    return f"{_URL_PREFIX}{name}{_URL_QUERY}"


class FakeSource(GraphSource):
    """A graph source serving PDFs from memory, recording every fetch."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    files: Mapping[str, bytes]
    failing: frozenset[str] = frozenset()
    unreachable: bool = False
    fetched: list[str] = Field(default_factory=list[str])

    @property
    def graph_name(self) -> str:
        return "test"

    def query(self, query: DatalogQuery) -> DatalogRows:
        raise NotImplementedError

    def fetch_file(self, url: HttpUrl) -> RoamAsset:
        self.fetched.append(str(url))
        if self.unreachable:
            raise requests.exceptions.ConnectionError("Local API unreachable")
        if str(url) in self.failing:
            raise requests.exceptions.HTTPError("Status Code: 500, Unable to get file metadata")
        return RoamAsset(
            file_name="stored.pdf.enc",
            last_modified=datetime.now(),
            media_type=None,
            contents=self.files[str(url)],
            original_file_name="upload.pdf",
        )


def embeds(*urls: str) -> list[PdfEmbed]:
    return [PdfEmbed(host_uid=f"host{index:05d}", url=pdf_url) for index, pdf_url in enumerate(urls)]


class TestPdfEmbeds:
    def test_both_component_forms_and_every_occurrence(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            {"uid": "aaaaaaaaa", "string": f"manual: {{{{pdf: {url('a.pdf')}}}}}"},
            {"uid": "bbbbbbbbb", "string": f"{{{{[[pdf]]: {url('b.pdf.enc')}}}}} and {{{{pdf: {url('c.pdf')}}}}}"},
        ]
        assert pdf_embeds(rows) == [
            PdfEmbed(host_uid="aaaaaaaaa", url=url("a.pdf")),
            PdfEmbed(host_uid="bbbbbbbbb", url=url("b.pdf.enc")),
            PdfEmbed(host_uid="bbbbbbbbb", url=url("c.pdf")),
        ]

    def test_upload_names_with_unescaped_punctuation(self) -> None:
        odd: Final[str] = url("Ab3dE.%20It's%20(final)%20v2!.pdf.enc")
        assert pdf_embeds([{"uid": "aaaaaaaaa", "string": f"{{{{[[pdf]]: {odd}}}}}"}]) == [
            PdfEmbed(host_uid="aaaaaaaaa", url=odd)
        ]

    def test_ignores_pages_other_hosts_and_plain_text(self) -> None:
        rows: Final[list[dict[str, object]]] = [
            {"uid": "ppppppppp", "title": "A page"},
            {"uid": "aaaaaaaaa", "string": "{{pdf: https://example.com/paper.pdf}}"},
            {"uid": "bbbbbbbbb", "string": "no attachment here"},
        ]
        assert pdf_embeds(rows) == []


class TestExtractedText:
    def test_text_layer_per_page(self) -> None:
        status, pages, error, _ = extracted_text(pdf_bytes([_LINE_ONE, _LINE_TWO]))
        assert status is PdfStatus.OK
        assert pages == (_LINE_ONE, _LINE_TWO)
        assert error is None

    def test_short_or_blank_page_counts_as_no_text(self) -> None:
        status, pages, _, _ = extracted_text(pdf_bytes([_LINE_ONE, "", "p. 3"]))
        assert status is PdfStatus.OK
        assert pages == (_LINE_ONE, "", "")

    def test_scan_without_any_text_layer(self) -> None:
        status, pages, error, _ = extracted_text(pdf_bytes(["", ""]))
        assert status is PdfStatus.NO_TEXT
        assert pages == ("", "")
        assert error is None

    def test_password_protected(self) -> None:
        status, pages, error, _ = extracted_text((FIXTURES / "password-protected.pdf").read_bytes())
        assert status is PdfStatus.PASSWORD
        assert pages == ()
        assert error is not None and "password" in error.lower()

    def test_unreadable_bytes(self) -> None:
        status, pages, error, _ = extracted_text(b"not a pdf at all")
        assert status is PdfStatus.PARSE_FAILED
        assert pages == ()
        assert error is not None


class TestExtractPdfTexts:
    def test_caches_each_distinct_pdf_owner_only(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        source: Final[FakeSource] = FakeSource(
            files={url("a.pdf"): pdf_bytes([_LINE_ONE]), url("b.pdf"): pdf_bytes(["", ""])}
        )
        summary: Final[ExtractionSummary] = extract_pdf_texts(
            cache, embeds(url("a.pdf"), url("b.pdf"), url("a.pdf")), source
        )
        assert summary.distinct_count == 2
        assert summary.outcomes == {PdfStatus.OK: 1, PdfStatus.NO_TEXT: 1}
        assert source.fetched == [url("a.pdf"), url("b.pdf")]
        assert stat.S_IMODE(cache.stat().st_mode) == 0o600
        cached: Final[dict[str, PdfText]] = cached_pdf_texts(cache)
        assert cached[url_key(url("a.pdf"))].page_texts == (_LINE_ONE,)
        assert cached[url_key(url("a.pdf"))].upload_name == "upload.pdf"
        assert cached[url_key(url("b.pdf"))].status is PdfStatus.NO_TEXT

    def test_never_stores_the_url(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        extract_pdf_texts(cache, embeds(url("a.pdf")), FakeSource(files={url("a.pdf"): pdf_bytes([_LINE_ONE])}))
        assert b"abc123" not in cache.read_bytes()

    def test_second_run_fetches_nothing(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        files: Final[dict[str, bytes]] = {url("a.pdf"): pdf_bytes([_LINE_ONE])}
        extract_pdf_texts(cache, embeds(url("a.pdf")), FakeSource(files=files))
        again: Final[FakeSource] = FakeSource(files=files)
        summary: Final[ExtractionSummary] = extract_pdf_texts(cache, embeds(url("a.pdf")), again)
        assert again.fetched == []
        assert summary.current_count == 1
        assert summary.extracted_count == 0

    def test_fetch_failure_is_cached_and_retried_only_on_request(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        files: Final[dict[str, bytes]] = {url("a.pdf"): pdf_bytes([_LINE_ONE]), url("gone.pdf"): b""}
        extract_pdf_texts(
            cache, embeds(url("gone.pdf"), url("a.pdf")), FakeSource(files=files, failing=frozenset({url("gone.pdf")}))
        )
        assert cached_pdf_texts(cache)[url_key(url("gone.pdf"))].status is PdfStatus.FETCH_FAILED

        plain: Final[FakeSource] = FakeSource(files=files)
        extract_pdf_texts(cache, embeds(url("gone.pdf"), url("a.pdf")), plain)
        assert plain.fetched == []

        retrying: Final[FakeSource] = FakeSource(files={**files, url("gone.pdf"): pdf_bytes([_LINE_TWO])})
        extract_pdf_texts(cache, embeds(url("gone.pdf"), url("a.pdf")), retrying, retry_failed=True)
        assert retrying.fetched == [url("gone.pdf")]
        assert cached_pdf_texts(cache)[url_key(url("gone.pdf"))].status is PdfStatus.OK

    def test_password_protected_is_never_refetched(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        files: Final[dict[str, bytes]] = {url("locked.pdf"): (FIXTURES / "password-protected.pdf").read_bytes()}
        extract_pdf_texts(cache, embeds(url("locked.pdf")), FakeSource(files=files))
        assert cached_pdf_texts(cache)[url_key(url("locked.pdf"))].status is PdfStatus.PASSWORD
        again: Final[FakeSource] = FakeSource(files=files)
        extract_pdf_texts(cache, embeds(url("locked.pdf")), again, retry_failed=True)
        assert again.fetched == []

    def test_newer_extractor_reextracts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        files: Final[dict[str, bytes]] = {url("a.pdf"): pdf_bytes([_LINE_ONE])}
        extract_pdf_texts(cache, embeds(url("a.pdf")), FakeSource(files=files))
        monkeypatch.setattr(pdf_text, "EXTRACTOR_VERSION", pdf_text.EXTRACTOR_VERSION + 1)
        again: Final[FakeSource] = FakeSource(files=files)
        extract_pdf_texts(cache, embeds(url("a.pdf")), again)
        assert again.fetched == [url("a.pdf")]
        assert cached_pdf_texts(cache)[url_key(url("a.pdf"))].extractor_version == pdf_text.EXTRACTOR_VERSION

    def test_limit_leaves_the_rest_pending(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        names: Final[list[str]] = ["a.pdf", "b.pdf", "c.pdf"]
        source: Final[FakeSource] = FakeSource(files={url(name): pdf_bytes([_LINE_ONE]) for name in names})
        summary: Final[ExtractionSummary] = extract_pdf_texts(
            cache, embeds(*(url(name) for name in names)), source, budget=PdfFetchBudget(max_pdfs=2)
        )
        assert source.fetched == [url("a.pdf"), url("b.pdf")]
        assert summary.pending_count == 1

    def test_a_run_of_fetch_failures_aborts_without_caching_them(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        names: Final[list[str]] = ["ok.pdf"] + [f"f{index}.pdf" for index in range(CONSECUTIVE_FETCH_FAILURE_LIMIT)]
        source: Final[FakeSource] = FakeSource(
            files={url("ok.pdf"): pdf_bytes([_LINE_ONE])},
            failing=frozenset(url(name) for name in names[1:]),
        )
        with pytest.raises(RuntimeError, match="in a row"):
            extract_pdf_texts(cache, embeds(*(url(name) for name in names)), source)
        assert set(cached_pdf_texts(cache)) == {url_key(url("ok.pdf"))}

    def test_isolated_fetch_failures_below_the_limit_are_cached(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        failing: Final[list[str]] = [f"f{index}.pdf" for index in range(CONSECUTIVE_FETCH_FAILURE_LIMIT - 1)]
        source: Final[FakeSource] = FakeSource(files={}, failing=frozenset(url(name) for name in failing))
        summary: Final[ExtractionSummary] = extract_pdf_texts(cache, embeds(*(url(name) for name in failing)), source)
        assert summary.outcomes == {PdfStatus.FETCH_FAILED: len(failing)}

    def test_unreachable_local_api_propagates(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        source: Final[FakeSource] = FakeSource(files={}, unreachable=True)
        with pytest.raises(requests.exceptions.ConnectionError):
            extract_pdf_texts(cache, embeds(url("a.pdf")), source)
        assert cached_pdf_texts(cache) == {}

    def test_existing_world_readable_cache_is_tightened(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        extract_pdf_texts(cache, [], FakeSource(files={}))
        cache.chmod(0o644)
        extract_pdf_texts(cache, [], FakeSource(files={}))
        assert stat.S_IMODE(cache.stat().st_mode) == 0o600


class TestFetchBudget:
    def test_byte_budget_stops_after_the_crossing_fetch(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        names: Final[list[str]] = ["a.pdf", "b.pdf", "c.pdf"]
        files: Final[dict[str, bytes]] = {url(name): pdf_bytes([_LINE_ONE]) for name in names}
        one_file: Final[int] = len(files[url("a.pdf")])
        source: Final[FakeSource] = FakeSource(files=files)
        summary: Final[ExtractionSummary] = extract_pdf_texts(
            cache, embeds(*(url(name) for name in names)), source, budget=PdfFetchBudget(max_bytes=one_file + 1)
        )
        assert source.fetched == [url("a.pdf"), url("b.pdf")]
        assert summary.fetched_bytes == 2 * one_file
        assert summary.pending_count == 1

    def test_spent_time_budget_fetches_nothing(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        source: Final[FakeSource] = FakeSource(files={url("a.pdf"): pdf_bytes([_LINE_ONE])})
        summary: Final[ExtractionSummary] = extract_pdf_texts(
            cache, embeds(url("a.pdf")), source, budget=PdfFetchBudget(max_seconds=0)
        )
        assert source.fetched == []
        assert summary.pending_count == 1

    def test_exhausted_checks_each_bound(self) -> None:
        assert not PdfFetchBudget().exhausted(10_000, 10**12, 10**6)
        assert PdfFetchBudget(max_pdfs=3).exhausted(3, 0, 0)
        assert not PdfFetchBudget(max_pdfs=3).exhausted(2, 0, 0)
        assert PdfFetchBudget(max_bytes=100).exhausted(1, 100, 0)
        assert PdfFetchBudget(max_seconds=5).exhausted(1, 0, 5.0)


_SCANNED: Final[str] = "Receipt for the garden shed paint, total 42 dollars"
"""The text rendered into fixtures/scanned.pdf, an image-only PDF (no text layer)."""


def stub_ocr(monkeypatch: pytest.MonkeyPatch, text: str = "Text recognized from the scanned page image") -> list[int]:
    """Make OCR available and canned; return the list each OCR'd page appends to."""
    calls: Final[list[int]] = []

    def recognized_page_text(page: object) -> str:
        calls.append(1)
        return text

    monkeypatch.setattr(pdf_text, "ocr_available", lambda: True)
    monkeypatch.setattr(pdf_text, "recognized_page_text", recognized_page_text)
    return calls


class TestOcr:
    def test_a_scan_has_no_text_layer_without_ocr(self) -> None:
        found: Final[ExtractedText] = extracted_text((FIXTURES / "scanned.pdf").read_bytes())
        assert (found.status, found.page_texts, found.ocr_page_count) == (PdfStatus.NO_TEXT, ("",), 0)

    @pytest.mark.skipif(not ocr_available(), reason="on-device OCR (macOS Vision via pyobjc) is unavailable")
    def test_vision_reads_a_scanned_page(self) -> None:
        found: Final[ExtractedText] = extracted_text((FIXTURES / "scanned.pdf").read_bytes(), ocr=True)
        assert found.status is PdfStatus.OK
        assert found.page_texts == (_SCANNED,)
        assert found.ocr_page_count == 1

    def test_ocr_reads_only_pages_without_a_text_layer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: Final[list[int]] = stub_ocr(monkeypatch)
        found: Final[ExtractedText] = extracted_text(pdf_bytes([_LINE_ONE, ""]), ocr=True)
        assert found.page_texts == (_LINE_ONE, "Text recognized from the scanned page image")
        assert (len(calls), found.ocr_page_count) == (1, 1)

    def test_ocr_that_finds_too_little_leaves_the_page_blank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_ocr(monkeypatch, text="p. 3")
        found: Final[ExtractedText] = extracted_text(pdf_bytes(["", ""]), ocr=True)
        assert (found.status, found.page_texts, found.ocr_page_count) == (PdfStatus.NO_TEXT, ("", ""), 0)

    def test_an_ocr_run_refetches_unread_scans_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        files: Final[dict[str, bytes]] = {url("scan.pdf"): pdf_bytes([""]), url("text.pdf"): pdf_bytes([_LINE_ONE])}
        extract_pdf_texts(cache, embeds(url("scan.pdf"), url("text.pdf")), FakeSource(files=files))
        assert cached_pdf_texts(cache)[url_key(url("scan.pdf"))].status is PdfStatus.NO_TEXT

        plain: Final[FakeSource] = FakeSource(files=files)
        extract_pdf_texts(cache, embeds(url("scan.pdf"), url("text.pdf")), plain)
        assert plain.fetched == []  # without OCR, a scan is not worth refetching

        stub_ocr(monkeypatch)
        reading: Final[FakeSource] = FakeSource(files=files)
        summary: Final[ExtractionSummary] = extract_pdf_texts(
            cache, embeds(url("scan.pdf"), url("text.pdf")), reading, ocr=True
        )
        assert reading.fetched == [url("scan.pdf")]  # the fully texted PDF needs no OCR
        assert summary.ocr_page_count == 1
        entry: Final[PdfText] = cached_pdf_texts(cache)[url_key(url("scan.pdf"))]
        assert (entry.status, entry.ocr_version, entry.ocr_page_count) == (PdfStatus.OK, OCR_VERSION, 1)

        again: Final[FakeSource] = FakeSource(files=files)
        extract_pdf_texts(cache, embeds(url("scan.pdf"), url("text.pdf")), again, ocr=True)
        assert again.fetched == []

    def test_a_scan_ocr_could_not_read_is_not_refetched(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        files: Final[dict[str, bytes]] = {url("blank.pdf"): pdf_bytes([""])}
        stub_ocr(monkeypatch, text="")
        extract_pdf_texts(cache, embeds(url("blank.pdf")), FakeSource(files=files), ocr=True)
        again: Final[FakeSource] = FakeSource(files=files)
        extract_pdf_texts(cache, embeds(url("blank.pdf")), again, ocr=True)
        assert again.fetched == []

    def test_ocr_asked_for_but_unavailable_changes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        files: Final[dict[str, bytes]] = {url("scan.pdf"): pdf_bytes([""])}
        extract_pdf_texts(cache, embeds(url("scan.pdf")), FakeSource(files=files))
        monkeypatch.setattr(pdf_text, "ocr_available", lambda: False)
        again: Final[FakeSource] = FakeSource(files=files)
        extract_pdf_texts(cache, embeds(url("scan.pdf")), again, ocr=True)
        assert again.fetched == []


class TestCacheSchemaUpgrade:
    def test_a_v1_cache_gains_the_ocr_columns_keeping_its_entries(self, tmp_path: Path) -> None:
        cache: Final[Path] = tmp_path / "test.pdf-text.db"
        connection: Final[sqlite3.Connection] = sqlite3.connect(cache)
        with connection:
            connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE pdf_texts (url_key TEXT PRIMARY KEY, status TEXT NOT NULL,"
                " extractor_version INTEGER NOT NULL, upload_name TEXT, byte_count INTEGER,"
                " page_texts TEXT NOT NULL, error TEXT, extracted_at TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO meta VALUES ('schema_version', '1')")
            connection.execute(
                "INSERT INTO pdf_texts VALUES (?, 'no-text', 1, 'scan.pdf', 10, '[\"\"]', NULL, '2026-11-13')",
                (url_key(url("scan.pdf")),),
            )
        connection.close()
        entry: Final[PdfText] = cached_pdf_texts(cache)[url_key(url("scan.pdf"))]
        assert (entry.status, entry.ocr_version, entry.ocr_page_count) == (PdfStatus.NO_TEXT, 0, 0)
        check: Final[sqlite3.Connection] = sqlite3.connect(cache)
        assert check.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == "2"
        check.close()
