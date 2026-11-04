"""Text acquisition: Tesseract OCR and the PDF's own text layer.

Both produce :class:`~polkadot.models.Word` objects in page-raster pixel
coordinates so that the rest of the pipeline does not care where text came
from.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from . import imaging as im
from .models import BBox, Word

# --------------------------------------------------------------------------
# Tesseract
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    return shutil.which("tesseract") is not None


@dataclass
class OcrConfig:
    lang: str = "eng"
    oem: int = 1
    text_psm: int = 6          # uniform block of text: listing columns
    ad_psm: int = 11           # sparse text: display advertising
    min_conf: float = 0.0      # words below this are dropped (-1 = keep all)
    pad: int = 12              # white border added around crops (px)
    upscale_below_dpi: int = 250
    try_rotations: bool = True


def _prep(crop: np.ndarray, pad: int) -> np.ndarray:
    g = im.to_gray(crop)
    return cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def ocr_image(crop: np.ndarray, psm: int, cfg: OcrConfig, dpi: int) -> list[Word]:
    """OCR a crop and return words in *crop* pixel coordinates."""
    import pytesseract

    if crop.size == 0 or min(crop.shape[:2]) < 6:
        return []
    scale = 1.0
    img = crop
    if dpi < cfg.upscale_below_dpi:
        scale = 300.0 / dpi
        img = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = _prep(img, cfg.pad)
    config = f"--oem {cfg.oem} --psm {psm} -c preserve_interword_spaces=1"
    try:
        data = pytesseract.image_to_data(img, lang=cfg.lang, config=config,
                                         output_type=pytesseract.Output.DICT)
    except Exception:  # pragma: no cover - tesseract crash on odd crops
        return []
    words: list[Word] = []
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt:
            continue
        conf = float(data["conf"][i])
        if conf < cfg.min_conf:
            continue
        x, y, w, h = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        b = BBox(x - cfg.pad, y - cfg.pad, x - cfg.pad + w, y - cfg.pad + h).scale(1.0 / scale)
        words.append(Word(txt, b, conf, "ocr"))
    return words


_GOOD_TOKEN = re.compile(
    r"^[\"'(“”]?(?:[A-Z]?[a-z]{2,}|[A-Z]{2,}|[A-Z]|\d{1,5}(?:-\d{1,5})?(?:st|nd|rd|th|d)?|&)[)'\".,;:”]*$")


def text_quality(words: list[Word]) -> float:
    """Share of tokens that look like real words / numbers (0..1)."""
    if not words:
        return 0.0
    good = sum(1 for w in words if _GOOD_TOKEN.match(w.text))
    return good / len(words)


def _score(words: list[Word]) -> float:
    s = 0.0
    for w in words:
        if w.conf < 30:
            continue
        alnum = sum(c.isalnum() for c in w.text)
        if alnum < 2 and not (alnum == 1 and w.conf > 80):
            continue
        s += (w.conf / 100.0) * alnum * (1.0 if _GOOD_TOKEN.match(w.text) else 0.4)
    return s


def ocr_region(page_img: np.ndarray, box: BBox, cfg: OcrConfig, dpi: int, psm: int,
               rotations: tuple[int, ...] = (0,)) -> tuple[list[Word], int, list[Word]]:
    """OCR ``box`` of the page, trying the given rotations (degrees CCW).

    Returns ``(page_words, rotation, frame_words)``: words mapped back to
    page coordinates, the rotation that read best, and the same words in
    the rotated reading frame (useful for grouping rotated text into lines).
    """
    H, W = page_img.shape[:2]
    pbox = box.pad(3, limit=BBox(0, 0, W, H))
    crop = im.crop(page_img, pbox)
    best: Optional[tuple[float, int, list[Word]]] = None
    for rot in rotations:
        fw = ocr_image(im.rotate_image(crop, rot), psm, cfg, dpi)
        sc = _score(fw) * (1.0 if rot == 0 else 0.97)   # prefer upright on ties
        if best is None or sc > best[0]:
            best = (sc, rot, fw)
        # upright text that reads well is good enough; skip other angles
        if rot == 0 and fw and text_quality(fw) > 0.85 and sc > 20:
            break
    if best is None:
        return [], 0, []
    _, rot, fwords = best
    ch, cw = crop.shape[:2]
    words = [Word(w.text, im.unrotate_bbox(w.bbox, rot, cw, ch).translate(pbox.x0, pbox.y0),
                  w.conf, w.source) for w in fwords]
    return words, rot, fwords


# --------------------------------------------------------------------------
# PDF text layer
# --------------------------------------------------------------------------


def pdf_words(page, scale: float, matrix: Optional[np.ndarray] = None) -> list[Word]:
    """Words of a PyMuPDF page, in raster pixels (after optional deskew)."""
    out = []
    try:
        raw = page.get_text("words")
    except Exception:  # pragma: no cover
        return out
    for x0, y0, x1, y1, txt, *_ in raw:
        txt = txt.strip()
        if not txt:
            continue
        b = BBox(x0, y0, x1, y1).scale(scale)
        if matrix is not None:
            b = im.transform_bbox(b, matrix)
        out.append(Word(txt, b, -1.0, "pdf"))
    return out


def words_in(words: list[Word], box: BBox) -> list[Word]:
    return [w for w in words if box.contains_point(w.bbox.cx, w.bbox.cy)]
