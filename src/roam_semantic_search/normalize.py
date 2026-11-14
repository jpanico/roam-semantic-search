"""Normalization of raw pull-block rows into embeddable index records.

Turns the full-graph fetch's pull-block rows into :class:`IndexRecord` values: breadcrumb
context assembled from each block's ancestor chain, source markup cleaned down to plain
prose, block references resolved to their target's text (one level), retrieval emphasis
extracted — the page names the entity's own text references (its *concepts*) and its
direct-child ``tags::`` values (its *tags*) — descendant block text folded in document
order, and skip rules applied (system pages, daily notes on request, blocks left empty
by cleanup).

Attached PDFs contribute records of their own: :func:`pdf_chunk_records` turns the cached
text of each PDF a block embeds (:mod:`roam_semantic_search.pdf_text`) into one record per
chunk of a page, set in its host block's context.  Chunk text never folds into any other
record's descendant text — a long PDF would swamp every ancestor and page record.

Public symbols:

- :class:`RecordKind` — what a record represents: a page, a block, or a PDF chunk.
- :class:`PdfLocator` — where a PDF chunk record sits: which PDF, page, and chunk.
- :class:`IndexRecord` — one embeddable record: identity, context, text, hash, bookkeeping.
- :func:`normalized_records` — pull-block rows → the records worth indexing.
- :func:`pdf_chunk_records` — pull-block rows + cached PDF text → one record per chunk.
- :data:`PDF_CHUNK_MAX_CHARS` / :data:`PDF_CHUNK_OVERLAP_CHARS` — the PDF chunking sizes.
- :data:`SKIPPED_PAGE_PREFIXES` — page-title prefixes excluded from the index (with their blocks).
- :data:`REF_PREVIEW_MAX_CHARS` — length cap on a resolved block reference's inserted text.
- :data:`BREADCRUMB_ANCESTOR_MAX_CHARS` — length cap on one ancestor's breadcrumb segment.
- :data:`EMBED_INPUT_MAX_CHARS` — length cap on a record's embeddable input.
"""

import enum
import hashlib
from collections.abc import Mapping, Sequence
from typing import Final

import regex
from guffin.model.primitives import DAILY_NOTE_UID_PATTERN, UID_PATTERN
from pydantic import BaseModel, ConfigDict, validate_call

from roam_semantic_search.json_narrowing import is_json_list, is_json_object
from roam_semantic_search.pdf_text import PdfStatus, PdfText, pdf_embeds, url_key

SKIPPED_PAGE_PREFIXES: Final[tuple[str, ...]] = ("roam/",)
"""Page-title prefixes whose pages (and all their blocks) are excluded from the index."""

REF_PREVIEW_MAX_CHARS: Final[int] = 300
"""Length cap on the text a resolved ``((uid))`` block reference inserts."""

BREADCRUMB_ANCESTOR_MAX_CHARS: Final[int] = 60
"""Length cap on one ancestor's segment of a breadcrumb."""

EMBED_INPUT_MAX_CHARS: Final[int] = 8000
"""Length cap on a record's embeddable input (breadcrumb + text)."""

PDF_CHUNK_MAX_CHARS: Final[int] = 2000
"""Length cap on one PDF chunk; a longer page splits into several chunks."""

PDF_CHUNK_OVERLAP_CHARS: Final[int] = 200
"""How much of a split page's text consecutive chunks share, so no passage is cut in two."""

_PDF_KEY_CHARS: Final[int] = 8

_DAILY_NOTE_UID_RE: Final[regex.Pattern[str]] = regex.compile(rf"^{DAILY_NOTE_UID_PATTERN}$")

_EMBED_RE: Final[regex.Pattern[str]] = regex.compile(
    rf"\{{\{{\[?\[?embed(?:-children|-path)?\]?\]?:?\s*(\(\({UID_PATTERN}\)\))\s*\}}\}}"
)
_TODO_MARKER_RE: Final[regex.Pattern[str]] = regex.compile(r"\{\{\[\[(?:TODO|DONE)\]\]\}\}")
_WIDGET_RE: Final[regex.Pattern[str]] = regex.compile(r"\{\{[^{}]*\}\}")
_BLOCK_REF_RE: Final[regex.Pattern[str]] = regex.compile(rf"\(\(({UID_PATTERN})\)\)")
_IMAGE_RE: Final[regex.Pattern[str]] = regex.compile(r"!\[([^\]]*)\]\([^()]*(?:\([^()]*\)[^()]*)*\)")
_MD_LINK_RE: Final[regex.Pattern[str]] = regex.compile(r"\[([^\[\]]+)\]\([^()]*(?:\([^()]*\)[^()]*)*\)")
_COLOR_TAG_RE: Final[regex.Pattern[str]] = regex.compile(r"#c:[A-Za-z0-9-]+")
_HASH_BRACKET_RE: Final[regex.Pattern[str]] = regex.compile(r"#(?=\[\[)")
_PAGE_REF_RE: Final[regex.Pattern[str]] = regex.compile(r"\[\[([^\[\]]*)\]\]")
_HASHTAG_RE: Final[regex.Pattern[str]] = regex.compile(r"#([\w-]+)")
_ATTRIBUTE_HEAD_RE: Final[regex.Pattern[str]] = regex.compile(r"^([^:\n`]+)::\s*")
_TAGS_ATTRIBUTE_RE: Final[regex.Pattern[str]] = regex.compile(r"^tags::\s*(?P<values>.*)$", regex.DOTALL)
_CODE_FENCE_RE: Final[regex.Pattern[str]] = regex.compile(r"```[\w+-]*")
_WHITESPACE_RUN_RE: Final[regex.Pattern[str]] = regex.compile(r"\s+")

_STYLE_DELIMITERS: Final[tuple[str, ...]] = ("**", "__", "^^", "~~", "`")


class RecordKind(enum.StrEnum):
    """What an index record represents."""

    PAGE = "page"
    BLOCK = "block"
    PDF = "pdf"
    """One chunk of a page of a PDF a block embeds."""


class PdfLocator(BaseModel):
    """Where a PDF chunk record sits.

    Attributes:
        key: The first characters of the PDF URL's cache key (:func:`~roam_semantic_search.pdf_text.url_key`)
            — enough to tell apart the PDFs one block embeds, never the URL itself.
        page: The 1-based page number the chunk comes from.
        chunk: The chunk's 1-based position within its page.
        name: The name the PDF was uploaded under, when known.
    """

    model_config = ConfigDict(frozen=True)

    key: str
    page: int
    chunk: int
    name: str | None = None


class IndexRecord(BaseModel):
    """One embeddable index record derived from a page or block entity.

    Attributes:
        uid: The entity's stable identifier.
        page_title: Title of the page the entity belongs to (the page's own title for a page).
        breadcrumb: Context path — page title, then ancestor block texts root-first.
        text: The entity's own cleaned plain text (the title, for a page).
        concepts: Names of the pages the entity's own text references — ``[[Page Name]]``
            and ``#tag`` spellings alike, every nesting level of a nested reference —
            in encounter order, deduplicated.
        tags: Values of the entity's direct-child ``tags::`` attribute blocks — the
            entity's explicit classification — in document order, deduplicated.
        descendant_text: Every descendant block's cleaned text, depth-first in document
            order, joined.
        embed_input: What gets embedded and content-hashed: context + own text, the
            concept and tag emphasis segments, then descendant text — priority-ordered
            so the length cap cuts the lowest-priority content first.
        content_hash: SHA-256 hex digest of ``embed_input``.
        edited_at: The entity's latest bookkeeping timestamp (epoch ms; create/edit maximum).
        is_page: Whether the record represents a page rather than a block.
        pdf: For a PDF chunk record, which PDF, page, and chunk it is; ``uid`` is then the
            uid of the block that embeds the PDF.
    """

    model_config = ConfigDict(frozen=True)

    uid: str
    page_title: str
    breadcrumb: str
    text: str
    concepts: tuple[str, ...]
    tags: tuple[str, ...]
    descendant_text: str
    embed_input: str
    content_hash: str
    edited_at: int
    is_page: bool
    pdf: PdfLocator | None = None

    @property
    def kind(self) -> RecordKind:
        """What the record represents: a page, a block, or a PDF chunk."""
        if self.pdf is not None:
            return RecordKind.PDF
        return RecordKind.PAGE if self.is_page else RecordKind.BLOCK

    @property
    def record_id(self) -> str:
        """The record's unique identity in the store.

        The Roam uid for a page or block record; ``<uid>/pdf/<key>/<page>.<chunk>`` for a
        PDF chunk, since every chunk of every PDF a block embeds shares the block's uid.
        """
        if self.pdf is None:
            return self.uid
        return f"{self.uid}/pdf/{self.pdf.key}/{self.pdf.page}.{self.pdf.chunk}"


def _str_field(row: Mapping[str, object], key: str) -> str | None:
    """Return the string value of *key* in *row*, else ``None``."""
    value: Final[object] = row.get(key)
    return value if isinstance(value, str) else None


def _int_field(row: Mapping[str, object], key: str) -> int:
    """Return the int value of *key* in *row*, else ``0``."""
    value: Final[object] = row.get(key)
    return value if isinstance(value, int) else 0


def _stub_id(row: Mapping[str, object], key: str) -> int | None:
    """Return the entity id of the single ``{"id": n}`` stub at *key* in *row*, else ``None``."""
    value: Final[object] = row.get(key)
    if not is_json_object(value):
        return None
    stub_id: Final[object] = value.get("id")
    return stub_id if isinstance(stub_id, int) else None


def _stub_ids(row: Mapping[str, object], key: str) -> list[int]:
    """Return the entity ids of the ``{"id": n}`` stub list at *key* in *row*."""
    value: Final[object] = row.get(key)
    if not is_json_list(value):
        return []
    ids: Final[list[int]] = []
    for stub in value:
        if not is_json_object(stub):
            continue
        stub_id: object = stub.get("id")
        if isinstance(stub_id, int):
            ids.append(stub_id)
    return ids


def _plain_text(raw_text: str, ref_texts: Mapping[str, str]) -> str:
    """Reduce a source markup string to plain prose.

    Embeds collapse to their block reference; TODO/DONE markers and remaining ``{{...}}``
    widgets drop; a ``((uid))`` reference is replaced by its target's text from *ref_texts*
    (missing targets drop); images keep their alt text, links their display text; color
    tags drop; page references and hashtags keep their inner text (nesting resolved
    innermost-first); an attribute head's ``::`` becomes ``:``; code fences and styling
    delimiters drop; whitespace runs collapse to single spaces.

    Args:
        raw_text: The source markup string.
        ref_texts: Plain-text-by-uid map a block reference resolves through.

    Returns:
        The cleaned plain text (possibly empty).
    """
    text = _EMBED_RE.sub(r"\1", raw_text)
    text = _TODO_MARKER_RE.sub("", text)
    text = _BLOCK_REF_RE.sub(lambda match: ref_texts.get(match.group(1), ""), text)
    text = _WIDGET_RE.sub("", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _COLOR_TAG_RE.sub("", text)
    text = _HASH_BRACKET_RE.sub("", text)
    while _PAGE_REF_RE.search(text) is not None:
        text = _PAGE_REF_RE.sub(r"\1", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    text = _ATTRIBUTE_HEAD_RE.sub(r"\1: ", text)
    text = _CODE_FENCE_RE.sub("", text)
    for delimiter in _STYLE_DELIMITERS:
        text = text.replace(delimiter, "")
    return _WHITESPACE_RUN_RE.sub(" ", text).strip()


def _is_indexable_page(row: Mapping[str, object], include_daily_notes: bool) -> bool:
    """Whether a page row belongs in the index (skip rules for pages)."""
    title: Final[str | None] = _str_field(row, "title")
    uid: Final[str | None] = _str_field(row, "uid")
    if title is None or uid is None or not title.strip():
        return False
    if any(title.startswith(prefix) for prefix in SKIPPED_PAGE_PREFIXES):
        return False
    if not include_daily_notes and _DAILY_NOTE_UID_RE.match(uid) is not None:
        return False
    return True


def _reference_texts(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    """Plain-text-by-uid for every entity, for one-level block-reference resolution.

    Each entity's own raw text (block string, or page title) is cleaned with an empty
    reference map — so a reference nested inside a referenced block drops rather than
    recursing — and capped at :data:`REF_PREVIEW_MAX_CHARS`.
    """
    texts: Final[dict[str, str]] = {}
    empty: Final[dict[str, str]] = {}
    for row in rows:
        uid: str | None = _str_field(row, "uid")
        raw: str | None = _str_field(row, "string") or _str_field(row, "title")
        if uid is None or raw is None:
            continue
        texts[uid] = _plain_text(raw, empty)[:REF_PREVIEW_MAX_CHARS]
    return texts


def _breadcrumb(
    row: Mapping[str, object],
    page_title: str,
    by_dbid: Mapping[int, Mapping[str, object]],
    ref_texts: Mapping[str, str],
) -> str:
    """Assemble a block's context path: page title, then ancestor block texts root-first.

    Ancestors are ordered by their own ancestor count (the page carries none, a
    top-level block one, ...), which reconstructs root-first order without relying on
    the wire order of the ``parents`` stub list.
    """
    ancestor_rows: Final[list[Mapping[str, object]]] = [
        by_dbid[stub_id] for stub_id in _stub_ids(row, "parents") if stub_id in by_dbid
    ]
    ordered: Final[list[Mapping[str, object]]] = sorted(
        ancestor_rows, key=lambda ancestor: len(_stub_ids(ancestor, "parents"))
    )
    segments: Final[list[str]] = []
    for ancestor in ordered:
        raw: str | None = _str_field(ancestor, "string")
        if raw is None:
            continue  # The page ancestor contributes the title, already leading the breadcrumb.
        segment: str = _plain_text(raw, ref_texts)[:BREADCRUMB_ANCESTOR_MAX_CHARS]
        if segment:
            segments.append(segment)
    if not segments:
        return page_title
    return f"{page_title} › {' · '.join(segments)}"


def _page_ref_names(raw_text: str, ref_texts: Mapping[str, str]) -> tuple[str, ...]:
    """Every page name a source markup string references, in encounter order, deduplicated.

    Both reference spellings count — ``[[Page Name]]`` (with its ``#[[...]]`` variant) and
    ``#tag`` — since both reference a page.  A nested reference yields every level,
    innermost first (``[[[[A]] B]]`` names both ``A`` and ``A B``).  An attribute head and
    ``#c:`` color tags are not references; each name is cleaned to plain text, and names
    left empty by cleaning drop.

    Args:
        raw_text: The source markup string.
        ref_texts: Plain-text-by-uid map a block reference resolves through.

    Returns:
        The referenced page names, cleaned, deduplicated, in encounter order.
    """
    text = _COLOR_TAG_RE.sub("", raw_text)
    text = _ATTRIBUTE_HEAD_RE.sub("", text)
    text = _HASH_BRACKET_RE.sub("", text)
    names: Final[list[str]] = []
    while (match := _PAGE_REF_RE.search(text)) is not None:
        names.append(match.group(1))
        text = text[: match.start()] + match.group(1) + text[match.end() :]
    names.extend(_HASHTAG_RE.findall(text))
    cleaned: Final[list[str]] = [_plain_text(name, ref_texts) for name in names]
    return tuple(dict.fromkeys(name for name in cleaned if name))


def _ordered_children(
    row: Mapping[str, object], by_dbid: Mapping[int, Mapping[str, object]]
) -> list[Mapping[str, object]]:
    """A row's direct-child rows in sibling order (each child's ``order`` block property)."""
    children: Final[list[Mapping[str, object]]] = [
        by_dbid[stub_id] for stub_id in _stub_ids(row, "children") if stub_id in by_dbid
    ]
    return sorted(children, key=lambda child: _int_field(child, "order"))


def _tag_names(
    row: Mapping[str, object],
    by_dbid: Mapping[int, Mapping[str, object]],
    ref_texts: Mapping[str, str],
) -> tuple[str, ...]:
    """The values of a row's direct-child ``tags::`` attribute blocks, deduplicated.

    Each ``tags::`` child contributes its referenced page names (reference and hashtag
    spellings alike); a child whose value text references no page falls back to its
    comma-separated plain tokens.

    Args:
        row: The pull-block row whose tags are read.
        by_dbid: Every row keyed by its entity id.
        ref_texts: Plain-text-by-uid map a block reference resolves through.

    Returns:
        The tag values, deduplicated, in document order.
    """
    names: Final[list[str]] = []
    for child in _ordered_children(row, by_dbid):
        raw: str | None = _str_field(child, "string")
        if raw is None:
            continue
        match: regex.Match[str] | None = _TAGS_ATTRIBUTE_RE.match(raw)
        if match is None:
            continue
        values: str = match.group("values")
        ref_names: tuple[str, ...] = _page_ref_names(values, ref_texts)
        if ref_names:
            names.extend(ref_names)
            continue
        names.extend(token.strip() for token in values.split(",") if token.strip())
    return tuple(dict.fromkeys(names))


def _descendant_texts(
    row: Mapping[str, object],
    by_dbid: Mapping[int, Mapping[str, object]],
    cleaned_by_dbid: Mapping[int, str],
) -> list[str]:
    """Every descendant block's cleaned text under a row, depth-first in document order.

    Args:
        row: The pull-block row whose subtree is folded.
        by_dbid: Every row keyed by its entity id.
        cleaned_by_dbid: Each row's cleaned plain text, keyed by entity id.

    Returns:
        The descendant texts, document-ordered; descendants left empty by cleaning drop.
    """
    texts: Final[list[str]] = []
    for child in _ordered_children(row, by_dbid):
        child_id: object = child.get("id")
        child_text: str = cleaned_by_dbid.get(child_id, "") if isinstance(child_id, int) else ""
        if child_text:
            texts.append(child_text)
        texts.extend(_descendant_texts(child, by_dbid, cleaned_by_dbid))
    return texts


def _embed_input(lead: str, concepts: Sequence[str], tags: Sequence[str], descendant_text: str) -> str:
    """Assemble a record's embeddable input, priority-ordered under the length cap.

    The lead (context + own text) comes first, then the concept and tag emphasis
    segments, then the descendant text — so truncation at :data:`EMBED_INPUT_MAX_CHARS`
    cuts the lowest-priority content first.  Empty segments are absent.

    Args:
        lead: The record's context and own text.
        concepts: The record's referenced page names.
        tags: The record's ``tags::`` classification values.
        descendant_text: The record's folded descendant text.

    Returns:
        The embeddable input, capped at :data:`EMBED_INPUT_MAX_CHARS`.
    """
    segments: Final[list[str]] = [lead]
    if concepts:
        segments.append("concepts: " + " · ".join(concepts))
    if tags:
        segments.append("tags: " + " · ".join(tags))
    if descendant_text:
        segments.append(descendant_text)
    return " | ".join(segments)[:EMBED_INPUT_MAX_CHARS]


def _hashed(embed_input: str) -> str:
    """SHA-256 hex digest of an embeddable input."""
    return hashlib.sha256(embed_input.encode()).hexdigest()


@validate_call
def normalized_records(rows: Sequence[dict[str, object]], include_daily_notes: bool = True) -> list[IndexRecord]:
    """Derive the embeddable index records from full-graph pull-block rows.

    Produces one record per indexable page (its title as text) and one per indexable
    block: blocks of skipped pages are skipped with them, and a block whose cleanup
    leaves no text (pure-markup or empty) yields no record.  Each record also carries
    its retrieval emphasis — the page names its own text references (``concepts``) and
    its direct-child ``tags::`` values (``tags``) — plus its whole subtree's cleaned
    text (``descendant_text``), document-ordered.

    Args:
        rows: Pull-block rows, one per entity, as returned by the full-graph fetch.
        include_daily_notes: Whether daily-note pages (and their blocks) are indexed.

    Returns:
        The index records, pages first, then blocks, in row order within each kind.
    """
    by_dbid: Final[dict[int, dict[str, object]]] = {}
    for row in rows:
        row_id: object = row.get("id")
        if isinstance(row_id, int):
            by_dbid[row_id] = row

    ref_texts: Final[dict[str, str]] = _reference_texts(rows)

    cleaned_by_dbid: Final[dict[int, str]] = {}
    for row in rows:
        raw_string: str | None = _str_field(row, "string")
        cleaned_row_id: object = row.get("id")
        if raw_string is not None and isinstance(cleaned_row_id, int):
            cleaned_by_dbid[cleaned_row_id] = _plain_text(raw_string, ref_texts)

    page_records: Final[list[IndexRecord]] = []
    indexable_page_dbids: Final[set[int]] = set()
    for row in rows:
        if "title" not in row or not _is_indexable_page(row, include_daily_notes):
            continue
        title: str = _str_field(row, "title") or ""
        uid: str = _str_field(row, "uid") or ""
        row_id = row.get("id")
        if isinstance(row_id, int):
            indexable_page_dbids.add(row_id)
        # The raw title stays the display identity; markup is cleaned out of what embeds.
        title_text: str = _plain_text(title, ref_texts) or title
        page_concepts: tuple[str, ...] = _page_ref_names(title, ref_texts)
        page_tags: tuple[str, ...] = _tag_names(row, by_dbid, ref_texts)
        page_descendant_text: str = " · ".join(_descendant_texts(row, by_dbid, cleaned_by_dbid))
        embed_input: str = _embed_input(title_text, page_concepts, page_tags, page_descendant_text)
        page_records.append(
            IndexRecord(
                uid=uid,
                page_title=title,
                breadcrumb=title_text,
                text=title_text,
                concepts=page_concepts,
                tags=page_tags,
                descendant_text=page_descendant_text,
                embed_input=embed_input,
                content_hash=_hashed(embed_input),
                edited_at=max(_int_field(row, "time"), _int_field(row, "edit-time")),
                is_page=True,
            )
        )

    block_records: Final[list[IndexRecord]] = []
    for row in rows:
        raw: str | None = _str_field(row, "string")
        uid_value: str | None = _str_field(row, "uid")
        if raw is None or uid_value is None:
            continue
        page_id: int | None = _stub_id(row, "page")
        if page_id is None or page_id not in indexable_page_dbids:
            continue
        page_row: dict[str, object] = by_dbid[page_id]
        page_title: str = _str_field(page_row, "title") or ""
        page_title_text: str = _plain_text(page_title, ref_texts) or page_title
        block_row_id: object = row.get("id")
        text: str = cleaned_by_dbid[block_row_id] if isinstance(block_row_id, int) else _plain_text(raw, ref_texts)
        if not text:
            continue
        breadcrumb: str = _breadcrumb(row, page_title_text, by_dbid, ref_texts)
        block_concepts: tuple[str, ...] = _page_ref_names(raw, ref_texts)
        block_tags: tuple[str, ...] = _tag_names(row, by_dbid, ref_texts)
        block_descendant_text: str = " · ".join(_descendant_texts(row, by_dbid, cleaned_by_dbid))
        block_embed_input: str = _embed_input(
            f"{breadcrumb} › {text}", block_concepts, block_tags, block_descendant_text
        )
        block_records.append(
            IndexRecord(
                uid=uid_value,
                page_title=page_title,
                breadcrumb=breadcrumb,
                text=text,
                concepts=block_concepts,
                tags=block_tags,
                descendant_text=block_descendant_text,
                embed_input=block_embed_input,
                content_hash=_hashed(block_embed_input),
                edited_at=max(_int_field(row, "time"), _int_field(row, "edit-time")),
                is_page=False,
            )
        )

    return [*page_records, *block_records]


def _chunked(text: str) -> list[str]:
    """Split a page's (whitespace-collapsed) text into overlapping chunks at word boundaries.

    A text within :data:`PDF_CHUNK_MAX_CHARS` is one chunk.  A longer one is cut at the last
    space before the cap (or at the cap, for a run with no space in its second half), and the
    next chunk starts :data:`PDF_CHUNK_OVERLAP_CHARS` before the cut, at a word start.
    """
    if len(text) <= PDF_CHUNK_MAX_CHARS:
        return [text]
    chunks: Final[list[str]] = []
    start: int = 0
    while True:
        end: int = start + PDF_CHUNK_MAX_CHARS
        if end >= len(text):
            chunks.append(text[start:].strip())
            return chunks
        space: int = text.rfind(" ", start + PDF_CHUNK_MAX_CHARS // 2, end)
        cut: int = space if space != -1 else end
        chunks.append(text[start:cut].strip())
        next_start: int = cut - PDF_CHUNK_OVERLAP_CHARS
        word_gap: int = text.find(" ", next_start, cut)
        start = max(word_gap + 1 if word_gap != -1 else next_start, start + 1)


def _indexable_page_dbids(rows: Sequence[Mapping[str, object]], include_daily_notes: bool) -> set[int]:
    """The entity ids of every page the index covers (the page skip rules applied)."""
    dbids: Final[set[int]] = set()
    for row in rows:
        row_id: object = row.get("id")
        if "title" in row and isinstance(row_id, int) and _is_indexable_page(row, include_daily_notes):
            dbids.add(row_id)
    return dbids


@validate_call
def pdf_chunk_records(
    rows: Sequence[dict[str, object]],
    pdf_texts: Mapping[str, PdfText],
    include_daily_notes: bool = True,
) -> list[IndexRecord]:
    """Derive one index record per chunk of every cached PDF the graph's blocks embed.

    A block on an indexable page that embeds a PDF (``{{pdf: <url>}}`` / ``{{[[pdf]]: <url>}}``)
    whose cached extraction is :attr:`~roam_semantic_search.pdf_text.PdfStatus.OK` yields a
    record for each chunk of each page with a text layer (:func:`_chunked`).  A chunk record
    carries the host block's uid, so a hit is one ``((uid))`` away from the PDF in its live
    context; its breadcrumb is the block's context path, its own text, and the PDF's upload
    name, and that context leads its embed input ahead of the chunk.  A PDF without a cached
    entry, or whose entry failed or found no text, yields nothing.

    Args:
        rows: Pull-block rows, one per entity, as returned by the full-graph fetch.
        pdf_texts: Cached extraction results by URL key (:func:`~roam_semantic_search.pdf_text.cached_pdf_texts`).
        include_daily_notes: Whether PDFs embedded on daily-note pages are indexed.

    Returns:
        The chunk records, in row order, then page order, then chunk order.
    """
    if not pdf_texts:
        return []
    by_dbid: Final[dict[int, dict[str, object]]] = {}
    for row in rows:
        row_id: object = row.get("id")
        if isinstance(row_id, int):
            by_dbid[row_id] = row
    ref_texts: Final[dict[str, str]] = _reference_texts(rows)
    indexable_page_dbids: Final[set[int]] = _indexable_page_dbids(rows, include_daily_notes)

    records: Final[list[IndexRecord]] = []
    seen: Final[set[tuple[str, str]]] = set()
    for row in rows:
        raw: str | None = _str_field(row, "string")
        uid: str | None = _str_field(row, "uid")
        if raw is None or uid is None or "pdf" not in raw:
            continue
        page_id: int | None = _stub_id(row, "page")
        if page_id is None or page_id not in indexable_page_dbids:
            continue
        page_title: str = _str_field(by_dbid[page_id], "title") or ""
        context: str = _breadcrumb(row, _plain_text(page_title, ref_texts) or page_title, by_dbid, ref_texts)
        own_text: str = _plain_text(raw, ref_texts)
        if own_text:
            context = f"{context} › {own_text}"
        edited_at: int = max(_int_field(row, "time"), _int_field(row, "edit-time"))
        for embed in pdf_embeds([row]):
            if (uid, embed.url) in seen:
                continue
            seen.add((uid, embed.url))
            key: str = url_key(embed.url)
            entry: PdfText | None = pdf_texts.get(key)
            if entry is None or entry.status is not PdfStatus.OK:
                continue
            pdf_context: str = f'{context} › PDF "{entry.upload_name}"' if entry.upload_name else f"{context} › PDF"
            for page_number, page_text in enumerate(entry.page_texts, start=1):
                if not page_text:
                    continue
                for chunk_number, chunk in enumerate(_chunked(page_text), start=1):
                    embed_input: str = _embed_input(f"{pdf_context} p. {page_number} › {chunk}", (), (), "")
                    records.append(
                        IndexRecord(
                            uid=uid,
                            page_title=page_title,
                            breadcrumb=pdf_context,
                            text=chunk,
                            concepts=(),
                            tags=(),
                            descendant_text="",
                            embed_input=embed_input,
                            content_hash=_hashed(embed_input),
                            edited_at=edited_at,
                            is_page=False,
                            pdf=PdfLocator(
                                key=key[:_PDF_KEY_CHARS], page=page_number, chunk=chunk_number, name=entry.upload_name
                            ),
                        )
                    )
    return records
