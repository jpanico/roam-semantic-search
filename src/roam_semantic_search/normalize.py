"""Normalization of raw pull-block rows into embeddable index records.

Turns the full-graph fetch's pull-block rows into :class:`IndexRecord` values: breadcrumb
context assembled from each block's ancestor chain, source markup cleaned down to plain
prose, block references resolved to their target's text (one level), and skip rules
applied (system pages, daily notes on request, blocks left empty by cleanup).

Public symbols:

- :class:`IndexRecord` — one embeddable record: identity, context, text, hash, bookkeeping.
- :func:`normalized_records` — pull-block rows → the records worth indexing.
- :data:`SKIPPED_PAGE_PREFIXES` — page-title prefixes excluded from the index (with their blocks).
- :data:`REF_PREVIEW_MAX_CHARS` — length cap on a resolved block reference's inserted text.
- :data:`BREADCRUMB_ANCESTOR_MAX_CHARS` — length cap on one ancestor's breadcrumb segment.
- :data:`EMBED_INPUT_MAX_CHARS` — length cap on a record's embeddable input.
"""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Final

import regex
from guffin.model.primitives import DAILY_NOTE_UID_PATTERN, UID_PATTERN
from pydantic import BaseModel, ConfigDict, validate_call

from roam_semantic_search.json_narrowing import is_json_list, is_json_object

SKIPPED_PAGE_PREFIXES: Final[tuple[str, ...]] = ("roam/",)
"""Page-title prefixes whose pages (and all their blocks) are excluded from the index."""

REF_PREVIEW_MAX_CHARS: Final[int] = 300
"""Length cap on the text a resolved ``((uid))`` block reference inserts."""

BREADCRUMB_ANCESTOR_MAX_CHARS: Final[int] = 60
"""Length cap on one ancestor's segment of a breadcrumb."""

EMBED_INPUT_MAX_CHARS: Final[int] = 8000
"""Length cap on a record's embeddable input (breadcrumb + text)."""

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
_CODE_FENCE_RE: Final[regex.Pattern[str]] = regex.compile(r"```[\w+-]*")
_WHITESPACE_RUN_RE: Final[regex.Pattern[str]] = regex.compile(r"\s+")

_STYLE_DELIMITERS: Final[tuple[str, ...]] = ("**", "__", "^^", "~~", "`")


class IndexRecord(BaseModel):
    """One embeddable index record derived from a page or block entity.

    Attributes:
        uid: The entity's stable identifier.
        page_title: Title of the page the entity belongs to (the page's own title for a page).
        breadcrumb: Context path — page title, then ancestor block texts root-first.
        text: The entity's own cleaned plain text (the title, for a page).
        embed_input: What gets embedded and content-hashed: breadcrumb joined with text.
        content_hash: SHA-256 hex digest of ``embed_input``.
        edited_at: The entity's latest bookkeeping timestamp (epoch ms; create/edit maximum).
        is_page: Whether the record represents a page rather than a block.
    """

    model_config = ConfigDict(frozen=True)

    uid: str
    page_title: str
    breadcrumb: str
    text: str
    embed_input: str
    content_hash: str
    edited_at: int
    is_page: bool


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


def _hashed(embed_input: str) -> str:
    """SHA-256 hex digest of an embeddable input."""
    return hashlib.sha256(embed_input.encode()).hexdigest()


@validate_call
def normalized_records(rows: Sequence[dict[str, object]], include_daily_notes: bool = True) -> list[IndexRecord]:
    """Derive the embeddable index records from full-graph pull-block rows.

    Produces one record per indexable page (its title as text) and one per indexable
    block: blocks of skipped pages are skipped with them, and a block whose cleanup
    leaves no text (pure-markup or empty) yields no record.

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
        embed_input: str = title_text[:EMBED_INPUT_MAX_CHARS]
        page_records.append(
            IndexRecord(
                uid=uid,
                page_title=title,
                breadcrumb=title_text,
                text=title_text,
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
        text: str = _plain_text(raw, ref_texts)
        if not text:
            continue
        breadcrumb: str = _breadcrumb(row, page_title_text, by_dbid, ref_texts)
        block_embed_input: str = f"{breadcrumb} › {text}"[:EMBED_INPUT_MAX_CHARS]
        block_records.append(
            IndexRecord(
                uid=uid_value,
                page_title=page_title,
                breadcrumb=breadcrumb,
                text=text,
                embed_input=block_embed_input,
                content_hash=_hashed(block_embed_input),
                edited_at=max(_int_field(row, "time"), _int_field(row, "edit-time")),
                is_page=False,
            )
        )

    return [*page_records, *block_records]
