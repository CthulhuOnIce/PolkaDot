"""Text acquisition: Tesseract OCR and the PDF's own text layer.

Both produce :class:`~polkadot.models.Word` objects in page-raster pixel
coordinates so that the rest of the pipeline does not care where text came
from.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from . import imaging as im
from .models import BBox, Word

log = logging.getLogger("polkadot.ocr")

# --------------------------------------------------------------------------
# Tesseract
# --------------------------------------------------------------------------

# Tesseract's OpenMP threads busy-wait; with several pages OCR'd in parallel
# (or any other load) they oversubscribe the CPU and a single call can stall
# for minutes.  We parallelise across pages instead.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


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
    ad_psm: int = 6            # display advertising ...
    ad_fallback_psm: int = 3   # ... merged with a second, auto-segmented pass (0 = off)
    min_conf: float = 0.0      # words below this are dropped (-1 = keep all)
    pad: int = 12              # white border added around crops (px)
    upscale_below_dpi: int = 250
    try_rotations: bool = True
    timeout: float = 120.0     # seconds per Tesseract call (0 = no limit)


def _erase_rules(g: np.ndarray) -> np.ndarray:
    """Paint out long straight rules (box borders, column rules).

    A border running along the edge of a crop makes Tesseract's page
    segmentation drop whole lines of text, so rules spanning most of the
    crop are replaced by the surrounding paper tone."""
    h, w = g.shape[:2]
    if h < 8 or w < 8:
        return g
    _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    hk = max(20, int(w * 0.5))
    vk = max(20, int(h * 0.5))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    lines |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    if not lines.any():
        return g
    # keep thin rules only: a solid panel (reverse-video ad) is not a rule
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    lines[thick > 0] = 0
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    paper = int(np.median(g[ink == 0])) if (ink == 0).any() else 255
    out = g.copy()
    out[lines > 0] = paper
    return out


def _prep(crop: np.ndarray, pad: int, flatten: bool = True, dpi: int = 300) -> np.ndarray:
    """Greyscale + border for Tesseract.

    Tesseract is easily thrown by a white border around grey/tinted paper,
    so we either flatten the background to white first (``flatten``) or
    pad with the crop's own paper colour (keeps reverse-video panels, which
    flattening would wash out)."""
    g = _erase_rules(im.to_gray(crop))
    if flatten:
        g = im.normalize_background(g, dpi)
        val = 255
    else:
        edge = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
        val = int(np.median(edge)) if edge.size else 255
    return cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=val)


def ocr_image(crop: np.ndarray, psm: int, cfg: OcrConfig, dpi: int,
              invert: bool = False, flatten: bool = True) -> list[Word]:
    """OCR a crop and return words in *crop* pixel coordinates.

    ``invert`` reads white-on-black (reverse video) display type."""
    import pytesseract

    if invert:
        crop = 255 - im.to_gray(crop)

    if crop.size == 0 or min(crop.shape[:2]) < 6:
        return []
    scale = 1.0
    img = crop
    if dpi < cfg.upscale_below_dpi:
        scale = 300.0 / dpi
        img = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = _prep(img, cfg.pad, flatten=flatten and not invert, dpi=int(dpi * scale))
    config = f"--oem {cfg.oem} --psm {psm} -c preserve_interword_spaces=1"
    try:
        data = pytesseract.image_to_data(img, lang=cfg.lang, config=config,
                                         output_type=pytesseract.Output.DICT,
                                         timeout=cfg.timeout or 0)
    except RuntimeError as exc:  # pytesseract raises RuntimeError on timeout
        log.warning("tesseract gave up on a %dx%d crop: %s", img.shape[1], img.shape[0], exc)
        return []
    except Exception as exc:  # pragma: no cover - tesseract crash on odd crops
        log.warning("tesseract failed: %s", exc)
        return []
    words: list[Word] = []
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip().strip("|¦")   # column rules read as bars
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
    r"^[\"'(“”]?(?:[A-Z]?[a-z]{2,}|[A-Z]{2,}|[A-Za-z]|\d{1,5}(?:-\d{1,5})?(?:st|nd|rd|th|d)?|&)[)'\".,;:”]*$")


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
               rotations: tuple[int, ...] = (0,), flatten: bool = True,
               multiscale: bool = False) -> tuple[list[Word], int, list[Word]]:
    """OCR ``box`` of the page, trying the given rotations (degrees CCW).

    Returns ``(page_words, rotation, frame_words)``: words mapped back to
    page coordinates, the rotation that read best, and the same words in
    the rotated reading frame (useful for grouping rotated text into lines).
    ``flatten`` evens out the paper tone first - right for small body type,
    wrong for heavy display type whose strokes are wider than the
    background-estimation kernel.
    """
    H, W = page_img.shape[:2]
    pbox = box.pad(3, limit=BBox(0, 0, W, H))
    crop = im.crop(page_img, pbox)
    best: Optional[tuple[float, int, list[Word]]] = None
    g = im.to_gray(crop)
    dark = float((g < 100).mean()) if g.size else 0.0
    inverts = (False, True) if dark > 0.22 else (False,)
    for rot in rotations:
        for inv in inverts:
            fw = ocr_image(im.rotate_image(crop, rot), psm, cfg, dpi, invert=inv,
                           flatten=flatten and len(inverts) == 1)
            sc = _score(fw) * (1.0 if rot == 0 else 0.97)   # prefer upright on ties
            if inv and best is not None and len(inverts) > 1 and best[1] == rot:
                # merge: reverse-video panels often sit next to normal type
                prev = best[2]
                merged = prev + [w for w in fw if not any(
                    w.bbox.iou(p.bbox) > 0.3 for p in prev)]
                msc = _score(merged) * (1.0 if rot == 0 else 0.97)
                if msc > best[0]:
                    best = (msc, rot, merged)
                continue
            if best is None or sc > best[0]:
                best = (sc, rot, fw)
        # upright text that reads well is good enough; skip other angles
        if rot == 0 and best and best[2] and text_quality(best[2]) > 0.85 and best[0] > 20:
            break
    if best is None:
        return [], 0, []
    _, rot, fwords = best
    if multiscale and fwords is not None:
        fwords = _add_large_type(im.rotate_image(crop, rot), fwords, psm, cfg, dpi, flatten)
    ch, cw = crop.shape[:2]
    words = [Word(w.text, im.unrotate_bbox(w.bbox, rot, cw, ch).translate(pbox.x0, pbox.y0),
                  w.conf, w.source) for w in fwords]
    return words, rot, fwords


def merge_words(a: list[Word], a_frame: list[Word], b: list[Word],
                b_frame: list[Word]) -> tuple[list[Word], list[Word]]:
    """Union of two OCR passes over the same crop (aligned page/frame lists):
    words of ``b`` are added where ``a`` found nothing."""
    out, out_f = list(a), list(a_frame)
    for w, fw in zip(b, b_frame):
        if w.conf < 50 or not any(c.isalnum() for c in w.text):
            continue
        if any(w.bbox.iou(o.bbox) > 0.1 or o.bbox.overlap_ratio(w.bbox) > 0.4
               or w.bbox.overlap_ratio(o.bbox) > 0.4 for o in out):
            continue
        out.append(w)
        out_f.append(fw)
    return out, out_f


def _add_large_type(crop: np.ndarray, words: list[Word], psm: int, cfg: OcrConfig,
                    dpi: int, flatten: bool, scales: tuple[float, ...] = (0.45, 0.3)) -> list[Word]:
    """Re-read a display ad at reduced scale to catch very large type.

    Tesseract is tuned for text a few dozen pixels high; the huge letters
    of display advertising (often a firm's name) are missed at full
    resolution but read fine when shrunk.  Words found only at a smaller
    scale, and bigger than the ad's ordinary type, are merged in."""
    hs = sorted(w.bbox.height for w in words if w.conf > 40)
    base_h = hs[len(hs) // 2] if hs else 0.0
    out = list(words)
    for sc in scales:
        small = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        if min(small.shape[:2]) < 12:
            continue
        for w in ocr_image(small, psm, cfg, int(dpi * sc), flatten=flatten):
            b = w.bbox.scale(1.0 / sc)
            if w.conf < 55 or sum(c.isalnum() for c in w.text) == 0:
                continue
            if base_h and b.height < base_h * 1.6:
                continue
            if any(b.iou(o.bbox) > 0.15 or o.bbox.overlap_ratio(b) > 0.5 or b.overlap_ratio(o.bbox) > 0.5
                   for o in out if o.conf > 40):
                continue
            out.append(Word(w.text, b, w.conf, w.source))
    return out


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
