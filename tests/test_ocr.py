"""Tests for roam_semantic_search.ocr — the image-text filters."""

from typing import Final

import pytest

from roam_semantic_search import ocr
from roam_semantic_search.ocr import readable_image_text, word_likeness


def stub_recognition(monkeypatch: pytest.MonkeyPatch, lines: list[tuple[str, float]]) -> None:
    """Make OCR available and have it recognize *lines* (text, confidence) top to bottom."""
    recognized: Final[list[ocr._Line]] = [
        ocr._Line(1.0 - index / 10, 0.0, text, confidence) for index, (text, confidence) in enumerate(lines)
    ]

    def recognized_lines(image: bytes) -> list[ocr._Line]:
        return recognized

    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    monkeypatch.setattr(ocr, "_recognized_lines", recognized_lines)


class TestWordLikeness:
    def test_share_of_tokens_in_the_word_list(self) -> None:
        assert word_likeness("the garden shed") == 1.0
        assert word_likeness("the qxzvt") == 0.5

    def test_no_tokens_cannot_be_judged(self) -> None:
        assert word_likeness("42 % 7") is None
        assert word_likeness("") is None


class TestReadableImageText:
    def test_confident_word_like_text_is_kept_in_reading_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_recognition(monkeypatch, [("Invoice for the garden shed", 0.9), ("total due today", 0.8)])
        assert readable_image_text(b"image") == "Invoice for the garden shed\ntotal due today"

    def test_low_confidence_lines_drop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_recognition(monkeypatch, [("Invoice for the garden shed", 0.9), ("smudge", 0.2)])
        assert readable_image_text(b"image") == "Invoice for the garden shed"

    def test_confident_noise_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub_recognition(monkeypatch, [("qxzvt brrnk vvlop zzmt", 1.0)])
        assert readable_image_text(b"image") == ""

    def test_nothing_without_ocr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ocr, "ocr_available", lambda: False)
        assert readable_image_text(b"image") == ""
