"""On-device OCR of PDF pages with macOS's Vision framework.

A scanned PDF page carries no text layer, so its words reach the index only through OCR.
Vision's text recognizer runs on this machine — no request leaves it, the same posture as
the local embedding server — and is reached through pyobjc.  A page is rendered with
pypdfium2, encoded as PNG in memory, and handed to Vision as bytes: nothing is written to
disk.

OCR is optional: on a machine without Vision (another OS, or pyobjc not installed)
:func:`ocr_available` is false and callers index text layers only.

Public symbols:

- :data:`OCR_VERSION` — the OCR logic's version; cached PDFs OCR'd by an older one are redone.
- :data:`OCR_RENDER_SCALE` / :data:`OCR_MAX_RENDER_PIXELS` — how a page is rendered for OCR.
- :func:`ocr_available` — whether on-device OCR can run here.
- :func:`recognized_page_text` — OCR one rendered PDF page.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# Rationale: pyobjc's Vision/Foundation bindings and pypdfium2 ship no type stubs, so their
# symbols are Unknown to pyright; these rules fire only on that propagation.  reportArgumentType
# too: pypdfium2 infers `render`'s `scale` as int but documents and accepts a float (as guffin notes).

import importlib.util
import io
import logging
from typing import Final

import pypdfium2

logger = logging.getLogger(__name__)

OCR_VERSION: Final[int] = 1
"""The OCR logic's version; a cached PDF OCR'd by an older one is fetched and OCR'd again."""

OCR_RENDER_SCALE: Final[float] = 3.0
"""Pixels per PDF point a page renders at for OCR (216 dpi)."""

OCR_MAX_RENDER_PIXELS: Final[int] = 4000
"""The longest side, in pixels, a page may render at; a larger page renders smaller."""


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


def recognized_page_text(page: pypdfium2.PdfPage) -> str:
    """OCR one PDF page with Vision's accurate recognizer, lines in reading order.

    Lines are ordered top to bottom, then left to right (Vision reports bounding boxes
    with a bottom-left origin).  A recognition failure is logged and yields ``""``, so one
    unreadable page never sinks the PDF's other pages.

    Args:
        page: The page to recognize (rendered here at :data:`OCR_RENDER_SCALE`).

    Returns:
        The recognized lines joined by newlines; ``""`` when nothing was recognized or OCR
        is unavailable.
    """
    try:
        # Imported here, not at module level: a machine without pyobjc must still import this module.
        import Vision
        from Foundation import NSData
    except ImportError:
        return ""
    png: Final[bytes] = _png_bytes(page)
    data = NSData.dataWithBytes_length_(png, len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    succeeded, error = handler.performRequests_error_([request], None)
    if not succeeded:
        logger.warning("Vision text recognition failed: %s", error)
        return ""
    lines: Final[list[tuple[float, float, str]]] = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        box = observation.boundingBox()
        lines.append((box.origin.y + box.size.height, box.origin.x, str(candidates[0].string())))
    lines.sort(key=lambda line: (-line[0], line[1]))
    return "\n".join(text for _, _, text in lines)
