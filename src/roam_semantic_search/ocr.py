"""On-device OCR of PDF pages with macOS's Vision framework.

A scanned PDF page carries no text layer, so its words reach the index only through OCR.
Vision's text recognizer runs on this machine — no request leaves it, the same posture as
the local embedding server — and is reached through pyobjc.  A page is rendered with
pypdfium2, encoded as PNG in memory, and handed to Vision as bytes: nothing is written to
disk.

OCR is optional: on a machine without Vision (another OS, or pyobjc not installed)
:func:`ocr_available` is false and callers index text layers only.

An attached image (Phase 6c) is read from its own bytes, no rendering needed — but most
photos carry no text, and Vision is confidently wrong on textures and scenes (6b measured
noise at confidence 0.6–1.0).  :func:`readable_image_text` therefore keeps only confident
lines and only text that reads as words: at least half its alphabetic tokens must be in
the system word list (:func:`word_likeness`).

Public symbols:

- :data:`OCR_VERSION` — the OCR logic's version; cached PDFs OCR'd by an older one are redone.
- :data:`OCR_RENDER_SCALE` / :data:`OCR_MAX_RENDER_PIXELS` — how a page is rendered for OCR.
- :func:`ocr_available` — whether on-device OCR can run here.
- :data:`OCR_CONFIDENCE_FLOOR` / :data:`WORDLIKE_FLOOR` — an image's text filters.
- :func:`recognized_page_text` — OCR one rendered PDF page.
- :func:`readable_image_text` — OCR an image's bytes, keeping only confident, word-like text.
- :func:`word_likeness` — the share of a text's alphabetic tokens found in the word list.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# Rationale: pyobjc's Vision/Foundation bindings and pypdfium2 ship no type stubs, so their
# symbols are Unknown to pyright; these rules fire only on that propagation.  reportArgumentType
# too: pypdfium2 infers `render`'s `scale` as int but documents and accepts a float (as guffin notes).

import functools
import importlib.util
import io
import logging
from pathlib import Path
from typing import Final, NamedTuple

import pypdfium2
import regex

logger = logging.getLogger(__name__)

OCR_VERSION: Final[int] = 1
"""The OCR logic's version; a cached PDF OCR'd by an older one is fetched and OCR'd again."""

OCR_RENDER_SCALE: Final[float] = 3.0
"""Pixels per PDF point a page renders at for OCR (216 dpi)."""

OCR_MAX_RENDER_PIXELS: Final[int] = 4000
"""The longest side, in pixels, a page may render at; a larger page renders smaller."""

OCR_CONFIDENCE_FLOOR: Final[float] = 0.5
"""The Vision confidence below which a line of an image's text is dropped."""

WORDLIKE_FLOOR: Final[float] = 0.5
"""The share of word-list tokens an image's text needs to count as words, not noise.

6b: real text scored 0.5–0.9, noise 0.0–0.4."""

_WORD_LIST_PATH: Final[Path] = Path("/usr/share/dict/words")
_TOKEN_RE: Final[regex.Pattern[str]] = regex.compile(r"\p{L}{3,}")


class _Line(NamedTuple):
    """One line Vision recognized: where it sits, its text, and Vision's confidence."""

    top: float
    left: float
    text: str
    confidence: float


def ocr_available() -> bool:
    """Whether on-device OCR (macOS Vision, through pyobjc) can run here."""
    return importlib.util.find_spec("Vision") is not None and importlib.util.find_spec("Foundation") is not None


def _png_bytes(page: pypdfium2.PdfPage) -> bytes:
    """The page rendered for OCR, as PNG bytes held in memory."""
    width, height = page.get_size()
    scale: Final[float] = min(OCR_RENDER_SCALE, OCR_MAX_RENDER_PIXELS / max(width, height, 1.0))
    image = page.render(scale=scale).to_pil()
    buffer: Final[io.BytesIO] = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _recognized_lines(image: bytes) -> list[_Line]:
    """Vision's accurate recognition of an image's bytes, lines in reading order.

    Lines are ordered top to bottom, then left to right (Vision reports bounding boxes with
    a bottom-left origin).  A recognition failure is logged and yields no lines.
    """
    try:
        # Imported here, not at module level: a machine without pyobjc must still import this module.
        import Vision
        from Foundation import NSData
    except ImportError:
        return []
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
        NSData.dataWithBytes_length_(image, len(image)), None
    )
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    succeeded, error = handler.performRequests_error_([request], None)
    if not succeeded:
        logger.warning("Vision text recognition failed: %s", error)
        return []
    lines: Final[list[_Line]] = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        box = observation.boundingBox()
        lines.append(
            _Line(
                box.origin.y + box.size.height,
                box.origin.x,
                str(candidates[0].string()),
                float(candidates[0].confidence()),
            )
        )
    lines.sort(key=lambda line: (-line.top, line.left))
    return lines


def recognized_page_text(page: pypdfium2.PdfPage) -> str:
    """OCR one PDF page with Vision's accurate recognizer, lines in reading order.

    A recognition failure yields ``""``, so one unreadable page never sinks the PDF's
    other pages.

    Args:
        page: The page to recognize (rendered here at :data:`OCR_RENDER_SCALE`).

    Returns:
        The recognized lines joined by newlines; ``""`` when nothing was recognized or OCR
        is unavailable.
    """
    if not ocr_available():
        return ""
    return "\n".join(line.text for line in _recognized_lines(_png_bytes(page)))


@functools.cache
def _word_list() -> frozenset[str] | None:
    """The system word list, lower-cased, or ``None`` where the machine has none."""
    try:
        text: Final[str] = _WORD_LIST_PATH.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        logger.warning("no word list at %s; image text is kept unjudged", _WORD_LIST_PATH)
        return None
    return frozenset(word.strip().lower() for word in text.splitlines() if word.strip())


def word_likeness(text: str) -> float | None:
    """The share of *text*'s alphabetic tokens (3+ letters) found in the system word list.

    Args:
        text: The text to judge.

    Returns:
        A share in ``[0, 1]``; ``None`` when the text has no such tokens or the machine has
        no word list.
    """
    words: Final[frozenset[str] | None] = _word_list()
    tokens: Final[list[str]] = [token.lower() for token in _TOKEN_RE.findall(text)]
    if words is None or not tokens:
        return None
    return sum(1 for token in tokens if token in words) / len(tokens)


def readable_image_text(image: bytes) -> str:
    """OCR an image's bytes, keeping only confident lines, and only text that reads as words.

    Lines below :data:`OCR_CONFIDENCE_FLOOR` drop; what remains is kept only when its
    :func:`word_likeness` reaches :data:`WORDLIKE_FLOOR` (or cannot be judged — no word
    list), since a photo's texture yields confident non-words.

    Args:
        image: The image's (decrypted) bytes — any format ImageIO reads (PNG, JPEG, HEIC, …).

    Returns:
        The kept lines joined by newlines, or ``""``.
    """
    if not ocr_available():
        return ""
    text: Final[str] = "\n".join(
        line.text for line in _recognized_lines(image) if line.confidence >= OCR_CONFIDENCE_FLOOR
    )
    likeness: Final[float | None] = word_likeness(text)
    if not text or (likeness is not None and likeness < WORDLIKE_FLOOR):
        return ""
    return text
