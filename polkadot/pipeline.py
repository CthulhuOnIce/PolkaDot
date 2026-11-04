"""Document pipeline.

Phase A (parallel, per page)
    render → deskew → layout → text (PDF text layer or OCR) → cache image
Phase B (sequential, whole document)
    section detection → entry assembly + field parsing → ad fields
    (headings such as the surname / street / category in force carry over
    from column to column and page to page, which is why this is serial)
Phase C
    crops, overlays and exports (see :mod:`polkadot.export`)
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from . import imaging as im
from . import ocr as ocrmod
from . import parse as P
from .layout import LayoutConfig, analyze_layout
from .models import (Ad, BBox, Entry, EntryKind, Line, PageResult, Region,
                     RegionKind, Word)
from .text import assemble_entries, group_lines, join_lines

log = logging.getLogger("polkadot")

SECTIONS = ("front", "ads", "misc", "street", "alphabetical", "classified")


@dataclass
class PipelineConfig:
    dpi: int = 300
    text_source: str = "auto"          # auto | pdf | ocr
    min_text_quality: float = 0.88     # pdf text layer quality needed in auto mode
    ocr_ads: bool = True               # OCR ads even when a text layer exists
    deskew: bool = True
    workers: int = 0                   # 0 = os.cpu_count()
    section_overrides: dict[int, str] = field(default_factory=dict)
    ocr: ocrmod.OcrConfig = field(default_factory=ocrmod.OcrConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    cache_dir: Optional[str] = None    # where deskewed page images are kept
    image_format: str = "jpg"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def parse_page_spec(spec: Optional[str], n_pages: int) -> list[int]:
    """'1-5,9,12-' (1-based, inclusive) -> sorted 0-based page indices."""
    if not spec:
        return list(range(n_pages))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo = int(a) if a.strip() else 1
            hi = int(b) if b.strip() else n_pages
            out.update(range(max(1, lo) - 1, min(n_pages, hi)))
        else:
            k = int(part)
            if 1 <= k <= n_pages:
                out.add(k - 1)
    return sorted(out)


def parse_section_map(spec: Optional[str], n_pages: int) -> dict[int, str]:
    """'1-18:ads,19-27:front,41-131:street' -> {page_index: section}."""
    out: dict[int, str] = {}
    if not spec:
        return out
    for part in spec.split(","):
        if ":" not in part:
            continue
        rng, sec = part.rsplit(":", 1)
        sec = sec.strip().lower()
        if sec not in SECTIONS:
            raise ValueError(f"unknown section {sec!r}; choose from {SECTIONS}")
        for p in parse_page_spec(rng, n_pages):
            out[p] = sec
    return out


def render_page(page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img[:, :, :3].copy()


def _page_words_from_pdf(page, dpi: int, matrix: Optional[np.ndarray]) -> list[Word]:
    scale = dpi / 72.0
    words = ocrmod.pdf_words(page, 1.0, None)
    if page.rotation:
        import pymupdf
        rm = page.rotation_matrix
        fixed = []
        for w in words:
            r = pymupdf.Rect(w.bbox.x0, w.bbox.y0, w.bbox.x1, w.bbox.y1) * rm
            fixed.append(Word(w.text, BBox(r.x0, r.y0, r.x1, r.y1), w.conf, w.source))
        words = fixed
    out = []
    for w in words:
        b = w.bbox.scale(scale)
        if matrix is not None:
            b = im.transform_bbox(b, matrix)
        out.append(Word(w.text, b, w.conf, "pdf"))
    return out


PAGE_NO_RE = re.compile(r"(?:^|\s)(\d{1,4})(?:\s|$)")


def _printed_page_number(header_text: str) -> Optional[str]:
    t = re.sub(r"\(\s*1[89]\d\d\s*\)", " ", header_text)   # drop "(1922)"
    nums = PAGE_NO_RE.findall(" " + t + " ")
    return nums[-1] if nums else None


# --------------------------------------------------------------------------
# Phase A: one page
# --------------------------------------------------------------------------


@dataclass
class PageWork:
    """Everything phase A produces for a page (picklable)."""
    result: PageResult
    image_path: Optional[str]
    matrix: Optional[list[list[float]]]     # deskew affine (orig px -> deskewed px)
    skew: float
    features: dict
    region_frames: dict                     # region id -> [(text, h, y0, y1, x0, conf)] in reading frame


def analyze_page(doc_path: str, page_no: int, cfg: PipelineConfig) -> PageWork:
    import pymupdf

    t0 = time.time()
    doc = pymupdf.open(doc_path)
    page = doc[page_no]
    dpi = cfg.dpi
    img = render_page(page, dpi)
    skew = 0.0
    matrix = None
    gray = im.to_gray(img)
    ink = im.binarize(gray, dpi)
    if cfg.deskew:
        skew = im.estimate_skew(ink)
        if abs(skew) >= 0.05:
            matrix = im.rotation_matrix(img.shape, skew)
            img = im.deskew(img, skew)
            gray = im.to_gray(img)
            ink = im.binarize(gray, dpi)
    layout = analyze_layout(img, dpi, page_no, cfg.layout, gray=gray, ink=ink)

    try:
        label = page.get_label() or None
    except Exception:  # pragma: no cover
        label = None
    res = PageResult(page=page_no, label=label, width=img.shape[1], height=img.shape[0],
                     dpi=dpi, section="", text_source="", regions=layout.regions,
                     content_bbox=layout.content)
    res.warnings = []

    pdf_all = _page_words_from_pdf(page, dpi, matrix) if cfg.text_source != "ocr" else []
    have_ocr = ocrmod.tesseract_available()
    if cfg.text_source == "ocr" and not have_ocr:
        raise RuntimeError("text_source='ocr' but tesseract is not installed")
    sources: set[str] = set()
    frames: dict[str, list] = {}

    out_regions: list[Region] = []
    for r in layout.regions:
        is_ad = r.kind.is_ad or r.kind is RegionKind.AD_LABEL
        pdf_words = ocrmod.words_in(pdf_all, r.bbox) if pdf_all else []
        q_pdf = ocrmod.text_quality(pdf_words)
        enough = len(pdf_words) >= max(3, r.bbox.area / (layout.body_h ** 2 * 400))
        if cfg.text_source == "pdf" or not have_ocr:
            use_pdf = True
        elif cfg.text_source == "ocr":
            use_pdf = False
        elif is_ad:
            use_pdf = not cfg.ocr_ads and q_pdf >= cfg.min_text_quality and enough
        else:
            use_pdf = q_pdf >= cfg.min_text_quality and enough

        rot = 0
        if use_pdf:
            words = fwords = pdf_words
            source = "pdf"
            if not have_ocr and cfg.text_source == "auto" and q_pdf < cfg.min_text_quality:
                res.warnings.append(f"{r.id}: low quality text layer and no OCR engine")
        else:
            if is_ad:
                tall = r.bbox.height > r.bbox.width * 1.6
                rots = (0, 270, 90) if (tall and cfg.ocr.try_rotations) else (0,)
                words, rot, fwords = _ocr_with_frame(img, r.bbox, cfg, rots, cfg.ocr.ad_psm, False)
                if cfg.ocr.ad_fallback_psm:
                    # psm 6 assumes one type size and can drop lines set smaller
                    # or larger than the rest; a second segmentation mode fills in
                    alt_w, _, alt_f = _ocr_with_frame(img, r.bbox, cfg, (rot,),
                                                      cfg.ocr.ad_fallback_psm, False)
                    words, fwords = ocrmod.merge_words(words, fwords, alt_w, alt_f)
            else:
                H, W = img.shape[:2]
                box = r.bbox.pad(dpi * 0.03, dpi * 0.01, limit=BBox(0, 0, W, H))
                words, rot, fwords = _ocr_with_frame(img, box, cfg, (0,), cfg.ocr.text_psm)
            source = "ocr"
            # a so-so text layer can still beat OCR of a poor (low-res) scan
            if cfg.text_source == "auto" and enough and rot == 0 \
                    and q_pdf > ocrmod.text_quality(words) + 0.05:
                words = fwords = pdf_words
                source = "pdf"
        r.rotation = rot
        r.meta["text_source"] = source
        r.meta["text_quality"] = round(ocrmod.text_quality(words), 3)
        sources.add(source)

        if r.kind is RegionKind.TEXT and words and rot == 0:
            kept, newbox = trim_edge_slivers(words, r.bbox, layout.body_h * 1.6)
            if len(kept) < len(words):
                r.meta["trimmed_words"] = len(words) - len(kept)
                words = fwords = kept
                r.bbox = newbox
        pieces = [(r, words, fwords)]
        if r.kind is RegionKind.TEXT:
            # last line of defence against two columns merged into one region
            parts = split_words_into_columns(words, r.bbox, layout.body_h)
            if len(parts) > 1:
                pieces = []
                for k, (pb, pw) in enumerate(parts):
                    sub = Region(id=f"{r.id}{'abcdefgh'[k]}", kind=r.kind, bbox=pb,
                                 rotation=0, column=r.column, meta=dict(r.meta))
                    sub.meta["split_from"] = r.id
                    pieces.append((sub, pw, pw))
        for reg, ws, fws in pieces:
            # group lines in the (possibly rotated) reading frame
            flines = group_lines(fws)
            lookup = {id(fw): pw for fw, pw in zip(fws, ws)}
            reg.lines = [Line([lookup[id(w)] for w in fl.words]) for fl in flines]
            frames[reg.id] = [(fl.text, fl.height, fl.bbox.y0, fl.bbox.y1, fl.bbox.x0, fl.conf)
                              for fl in flines]
            out_regions.append(reg)
    # renumber columns left to right after any splits
    cols = sorted([g for g in out_regions if g.kind is RegionKind.TEXT and g.column >= 0],
                  key=lambda g: (g.column, g.bbox.x0))
    for i, g in enumerate(cols):
        g.column = i
    layout.regions = out_regions
    res.regions = out_regions
    res.text_source = "mixed" if len(sources) > 1 else (sources.pop() if sources else "none")

    # printed page number from the running head
    for r in layout.regions:
        if r.kind in (RegionKind.HEADER, RegionKind.FOOTER) and r.lines:
            n = _printed_page_number(r.text)
            if n:
                r.meta["printed_page"] = n
                res.label = res.label or n
                break

    img_path = None
    if cfg.cache_dir:
        os.makedirs(cfg.cache_dir, exist_ok=True)
        img_path = os.path.join(cfg.cache_dir, f"page_{page_no + 1:04d}.{cfg.image_format}")
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        params = [cv2.IMWRITE_JPEG_QUALITY, 88] if cfg.image_format == "jpg" else []
        cv2.imwrite(img_path, bgr, params)

    feats = section_features(res)
    log.debug("page %d analysed in %.1fs (%d regions)", page_no + 1, time.time() - t0,
              len(layout.regions))
    return PageWork(res, img_path, matrix.tolist() if matrix is not None else None,
                    skew, feats, frames)


def trim_edge_slivers(words: list[Word], box: BBox, em: float) -> tuple[list[Word], BBox]:
    """Drop print from the facing page that leaked into the edge of a column.

    A strip at the left/right edge, cut off from the column by a vertical
    channel no word crosses, and reading badly (fragments of clipped words),
    is a sliver of the neighbouring leaf."""
    if len(words) < 12:
        return words, box
    x0 = int(box.x0)
    n = int(box.width) + 2
    cover = np.zeros(n, np.int32)
    for w in words:
        cover[max(0, int(w.bbox.x0) - x0):max(0, int(w.bbox.x1) - x0)] += 1
    lim = int(box.width * 0.3)

    def channel(rng) -> Optional[int]:
        run = 0
        for i in rng:
            run = run + 1 if cover[i] == 0 else 0
            if run >= em * 0.35:
                return i
        return None

    out = list(words)
    nb = box
    cut = channel(range(lim, 0, -1))           # scan from 30% towards the left edge
    if cut is not None:
        side = [w for w in out if w.bbox.cx < x0 + cut]
        if side and len(side) < len(out) * 0.4 and _poor(side):
            out = [w for w in out if w not in side]
            nb = BBox(x0 + cut, nb.y0, nb.x1, nb.y1)
    cut = channel(range(n - lim, n))
    if cut is not None:
        side = [w for w in out if w.bbox.cx > x0 + cut]
        if side and len(side) < len(out) * 0.4 and _poor(side):
            out = [w for w in out if w not in side]
            nb = BBox(nb.x0, nb.y0, x0 + cut, nb.y1)
    return out, nb


def _poor(words: list[Word]) -> bool:
    confs = [w.conf for w in words if w.conf >= 0]
    mean_conf = sum(confs) / len(confs) if confs else 100.0
    return mean_conf < 65 or ocrmod.text_quality(words) < 0.55


def split_words_into_columns(words: list[Word], box: BBox, em: float,
                             min_lines: int = 6) -> list[tuple[BBox, list[Word]]]:
    """Split a text region whose words reveal more than one column.

    Looks for a vertical channel that no word crosses, well inside the
    region, with plenty of words on both sides."""
    if len(words) < min_lines * 4 or box.width < em * 24:
        return [(box, words)]
    x0 = int(box.x0)
    cover = np.zeros(int(box.width) + 2, np.int32)
    for w in words:
        a, b = int(w.bbox.x0) - x0, int(w.bbox.x1) - x0
        cover[max(0, a):max(0, b)] += 1
    gaps, i, n = [], 0, len(cover)
    lo_lim, hi_lim = int(em * 10), n - int(em * 10)
    while i < n:
        if cover[i] == 0:
            j = i
            while j < n and cover[j] == 0:
                j += 1
            if j - i >= em * 0.9 and i > lo_lim and j < hi_lim:
                gaps.append((i, j))
            i = j
        else:
            i += 1
    if not gaps:
        return [(box, words)]
    a, b = max(gaps, key=lambda g: g[1] - g[0])
    cut = x0 + (a + b) / 2.0
    left = [w for w in words if w.bbox.cx < cut]
    right = [w for w in words if w.bbox.cx >= cut]
    if len(left) < min_lines * 2 or len(right) < min_lines * 2:
        return [(box, words)]
    lb = BBox(box.x0, box.y0, cut, box.y1)
    rb = BBox(cut, box.y0, box.x1, box.y1)
    return split_words_into_columns(left, lb, em, min_lines) + \
        split_words_into_columns(right, rb, em, min_lines)


def _ocr_with_frame(img: np.ndarray, box: BBox, cfg: PipelineConfig,
                    rotations: tuple[int, ...], psm: int, flatten: bool = True):
    # display ads (flatten=False) also get a reduced-scale pass for huge type
    return ocrmod.ocr_region(img, box, cfg.ocr, cfg.dpi, psm, rotations, flatten,
                             multiscale=not flatten)


# --------------------------------------------------------------------------
# Section detection
# --------------------------------------------------------------------------

_RES_RE = re.compile(r"\s(?:h|r|bds)\s+(?:\d|same\b)")
_MARKERS = [
    (re.compile(r"classified\s+business", re.I), "classified"),
    (re.compile(r"buyers'?\s*guide", re.I), "classified"),
    (re.compile(r"street\s+(?:and\s+avenue\s+)?(?:guide|directory)", re.I), "street"),
    (re.compile(r"householders'?\s+directory", re.I), "street"),
    (re.compile(r"miscellaneous\s+(?:information|directory)", re.I), "misc"),
    (re.compile(r"alphabetical\s+(?:list|directory)", re.I), "alphabetical"),
]


_TITLED_RE = re.compile(r"""^[*•"“”'‘’]?\s*[A-Z][a-z]+(?:[ —-]+(?:[A-Z][a-z]+|and|of|&))*$""")
_DASH_RE = re.compile(r"[A-Za-z.)]\s?—\s?\d")


def section_features(res: PageResult) -> dict:
    lines = [l for r in res.regions if r.kind is RegionKind.TEXT for l in r.lines]
    n = len(lines)
    # centred, short, mixed-case lines without numbers = category headings
    titled = 0
    for r in res.regions:
        if r.kind is not RegionKind.TEXT:
            continue
        for l in r.lines:
            t = l.text.strip()
            b = l.bbox
            if (len(t.split()) <= 5 and not any(c.isdigit() for c in t) and not t.isupper()
                    and _TITLED_RE.match(t) and abs(b.cx - r.bbox.cx) < r.bbox.width * 0.12
                    and b.width < r.bbox.width * 0.7 and b.x0 > r.bbox.x0 + l.height * 2):
                titled += 1
    ad_area = sum(r.bbox.area for r in res.regions if r.kind.is_ad)
    content_area = res.content_bbox.area if res.content_bbox else 1.0
    texts = [l.text.strip() for l in lines]
    num_start = sum(1 for t in texts if re.match(r"^\d{1,5}\b", t))
    res_hits = sum(1 for t in texts if _RES_RE.search(" " + t))
    see = sum(1 for t in texts if "(See" in t or "(see" in t)
    star = sum(1 for t in texts if t.startswith("*"))
    dash = sum(1 for t in texts if _DASH_RE.search(t))
    # section titles only count on sparse pages (title / divider pages) and
    # only from the listing text - ads quote "the Classified Business Lists"
    markers: list[str] = []
    if n < 40:
        body_text = " ".join(texts)
        found = {sec for rx, sec in _MARKERS if rx.search(body_text)}
        if len(found) == 1:
            markers = sorted(found)
    return {"n_lines": n, "ad_frac": ad_area / max(1.0, content_area),
            "num_start": num_start / max(1, n), "res": res_hits / max(1, n),
            "see": see / max(1, n), "star": star, "titled": titled / max(1, n),
            "dash": dash / max(1, n), "markers": markers}


def guess_section(f: dict) -> tuple[str, float]:
    """Return (section, confidence 0..1) from one page's features."""
    if f["n_lines"] < 8:
        return ("ads" if f["ad_frac"] > 0.4 else "front"), 0.6
    if f["num_start"] > 0.35:
        return "street", min(1.0, f["num_start"] * 1.6)
    if f["res"] > 0.18:
        return "alphabetical", min(1.0, f["res"] * 2.5)
    titled, dash = f.get("titled", 0.0), f.get("dash", 0.0)
    if f["res"] < 0.1:
        cls = min(1.0, f["see"] * 8 + titled * 6 + f["star"] * 0.1)
        misc = min(1.0, dash * 4)
        if cls >= misc and cls > 0.15:
            return "classified", cls
        if misc > 0.15:
            return "misc", misc
    return "misc", 0.2


def assign_sections(works: list[PageWork], overrides: dict[int, str]) -> None:
    """Per-page guesses smoothed with document context.

    Directories are organised in long contiguous sections.  A weakly
    classified page takes the section of the confident pages around it
    when those agree (or follows an explicit section title page); confident
    pages and advertising pages keep their own classification.
    """
    guesses: list[tuple[str, float]] = []
    for w in works:
        p = w.result.page
        if p in overrides:
            guesses.append((overrides[p], 2.0))
        else:
            guesses.append(guess_section(w.features))
    n = len(works)
    strong = [g[1] >= 0.5 and g[0] not in ("ads", "front") for g in guesses]

    def neighbour(i: int, step: int) -> Optional[str]:
        j = i + step
        while 0 <= j < n:
            if strong[j]:
                return guesses[j][0]
            # a section title page also anchors what follows it
            if step < 0 and works[j].features.get("markers"):
                return works[j].features["markers"][0]
            j += step
        return None

    for i, w in enumerate(works):
        sec, conf = guesses[i]
        if conf < 0.5 and sec not in ("ads", "front"):
            before, after = neighbour(i, -1), neighbour(i, +1)
            if before is not None and before == after:
                sec = before
            elif after is None and before is not None and w.features["n_lines"] >= 20:
                sec = before          # tail of the book
        w.result.section = sec

    # real sections run for dozens of pages: a short island of another
    # listing section between two long stretches of the same one is a
    # misreading (advertising pages and explicit overrides excepted)
    listing = ("alphabetical", "street", "classified", "misc")
    runs: list[list[int]] = []
    for i, w in enumerate(works):
        if runs and works[runs[-1][0]].result.section == w.result.section:
            runs[-1].append(i)
        else:
            runs.append([i])
    ads_like = ("ads", "front")

    def neighbour_run(k: int, step: int) -> Optional[list[int]]:
        j = k + step
        while 0 <= j < len(runs) and works[runs[j][0]].result.section in ads_like:
            j += step
        return runs[j] if 0 <= j < len(runs) else None

    for k, run in enumerate(runs):
        sec = works[run[0]].result.section
        if sec not in listing or len(run) > 2:
            continue
        if any(works[i].result.page in overrides for i in run):
            continue
        prev, nxt = neighbour_run(k, -1), neighbour_run(k, +1)
        if prev and nxt and len(prev) >= 3 and len(nxt) >= 3:
            ps, ns = works[prev[0]].result.section, works[nxt[0]].result.section
            if ps == ns and ps in listing and ps != sec:
                for i in run:
                    works[i].result.section = ps


# --------------------------------------------------------------------------
# Phase B: entries & ads
# --------------------------------------------------------------------------


@dataclass
class Context:
    surname: Optional[str] = None
    street: Optional[str] = None
    house_number: Optional[str] = None
    category: Optional[str] = None
    section: Optional[str] = None


def build_entries(work: PageWork, ctx: Context) -> None:
    res = work.result
    sec = res.section
    if ctx.section != sec:
        ctx.section = sec
        if sec != "alphabetical":
            ctx.surname = None
        ctx.house_number = None
    n = 0
    text_regions = sorted([r for r in res.regions if r.kind is RegionKind.TEXT],
                          key=lambda r: (r.column if r.column >= 0 else 99, r.bbox.x0, r.bbox.y0))
    for r in text_regions:
        raws = assemble_entries(r.lines, r.bbox, numbered=(sec == "street"),
                                heading_style="caps" if sec in ("street", "alphabetical") else "any")
        for raw in raws:
            text = raw.text
            if not text:
                continue
            kind = EntryKind.OTHER
            fields: dict = {}
            advertiser = False
            heading = None
            if raw.is_heading or (sec == "street" and P.parse_street_heading(text)):
                kind = EntryKind.HEADING
                h = P.clean_heading(text)
                if sec == "alphabetical":
                    word = h.split()[0] if h.split() else h
                    ctx.surname = word.strip(",.—-").title() if word.isupper() else ctx.surname
                    fields = {"surname": ctx.surname}
                elif sec == "street":
                    sh = P.parse_street_heading(text) or {"street": h.title()}
                    ctx.street = sh["street"]
                    ctx.house_number = None
                    fields = sh
                elif sec in ("classified", "misc"):
                    ctx.category = h
                    fields = {"category": h}
                heading = h
            elif sec == "alphabetical":
                out = P.parse_alphabetical(text, ctx.surname)
                fields, advertiser = out["fields"], out.get("advertiser", False)
                kind = EntryKind(out["kind"])
                if fields.get("surname") and kind is not EntryKind.BUSINESS:
                    ctx.surname = fields["surname"]
                heading = ctx.surname
            elif sec == "street":
                out = P.parse_street_line(text, ctx.street, ctx.house_number)
                fields, advertiser = out["fields"], out.get("advertiser", False)
                kind = EntryKind(out["kind"]) if out["kind"] != "other" else EntryKind.STREET
                if out["kind"] == "person":
                    kind = EntryKind.STREET
                if fields.get("house_number") and not fields.get("location") and not fields.get("unit"):
                    ctx.house_number = fields["house_number"]
                heading = ctx.street
            elif sec == "classified":
                out = P.parse_classified(text, ctx.category)
                fields, advertiser = out["fields"], out.get("advertiser", False)
                kind = EntryKind.BUSINESS
                heading = ctx.category
            elif sec == "misc":
                out = P.parse_misc(text, ctx.category)
                fields, advertiser = out["fields"], out.get("advertiser", False)
                kind = EntryKind(out["kind"])
                heading = ctx.category
            if "phones" not in fields:
                phones = P.find_phones(text)
                if phones:
                    fields["phones"] = phones
            e = Entry(id=f"p{res.page + 1}_e{n}", page=res.page, region_id=r.id, section=sec,
                      kind=kind, text=text, bbox=raw.bbox, lines=raw.lines,
                      heading=heading, fields=fields, advertiser=bool(advertiser))
            res.entries.append(e)
            n += 1


def build_ads(work: PageWork) -> None:
    res = work.result
    labels = {r.id: r for r in res.regions if r.kind is RegionKind.AD_LABEL}
    for r in res.regions:
        if not r.kind.is_ad:
            continue
        frame = work.region_frames.get(r.id, [])
        text = join_lines([t for t, *_ in frame]) if frame else r.text
        if sum(c.isalpha() for c in text) < 3:
            continue          # a page number, an ornament or an unreadable cut
        f = P.ad_fields(frame, text)
        cat = None
        if r.meta.get("label_region") in labels:
            cat = P.clean_heading(labels[r.meta["label_region"]].text).title() or None
            if cat and sum(c.isalpha() for c in cat) < 3:
                cat = None
        ad = Ad(id=f"p{res.page + 1}_ad{len(res.ads)}", page=res.page, region_id=r.id,
                kind=r.kind, bbox=r.bbox, rotation=r.rotation, text=text,
                business_name=f.get("business_name") or None, category=cat,
                addresses=f.get("addresses", []), phones=f.get("phones", []),
                people=f.get("people", []),
                meta={"text_source": r.meta.get("text_source"),
                      "text_quality": r.meta.get("text_quality")})
        res.ads.append(ad)


# --------------------------------------------------------------------------
# Document driver
# --------------------------------------------------------------------------


def run_pages(doc_path: str, pages: list[int], cfg: PipelineConfig,
              progress: Optional[Callable[[int, int, int], None]] = None) -> list[PageWork]:
    workers = cfg.workers or (os.cpu_count() or 1)
    works: dict[int, PageWork] = {}
    if workers <= 1 or len(pages) <= 1:
        for i, p in enumerate(pages):
            works[p] = analyze_page(doc_path, p, cfg)
            if progress:
                progress(i + 1, len(pages), p)
    else:
        # tesseract is itself multi-threaded; one thread per process is
        # faster overall when we parallelise across pages
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        # "spawn" avoids fork-after-threads deadlocks (and matches macOS /
        # Windows behaviour)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(analyze_page, doc_path, p, cfg): p for p in pages}
            done = 0
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    works[p] = fut.result()
                except Exception as exc:  # keep going; report the page
                    log.error("page %d failed: %s", p + 1, exc)
                done += 1
                if progress:
                    progress(done, len(pages), p)
    return [works[p] for p in sorted(works)]


def process_document(doc_path: str, pages: Optional[Iterable[int]] = None,
                     cfg: Optional[PipelineConfig] = None,
                     progress: Optional[Callable[[int, int, int], None]] = None) -> list[PageWork]:
    """Run phases A and B over ``pages`` of ``doc_path``."""
    import pymupdf

    cfg = cfg or PipelineConfig()
    with pymupdf.open(doc_path) as doc:
        n = doc.page_count
    page_list = list(pages) if pages is not None else list(range(n))
    works = run_pages(str(doc_path), page_list, cfg, progress)
    assign_sections(works, cfg.section_overrides)
    ctx = Context()
    for w in works:
        build_entries(w, ctx)
        build_ads(w)
    return works


def reparse(works: list[PageWork], overrides: Optional[dict[int, str]] = None) -> list[PageWork]:
    """Re-run phase B (sections, entries, ads) on already analysed pages."""
    for w in works:
        w.result.entries = []
        w.result.ads = []
        w.result.section = ""
    assign_sections(works, overrides or {})
    ctx = Context()
    for w in works:
        build_entries(w, ctx)
        build_ads(w)
    return works


def save_works(works: list[PageWork], path: str) -> None:
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(works, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_works(path: str) -> list[PageWork]:
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)


def summarize(works: list[PageWork]) -> dict:
    c = Counter()
    secs = Counter()
    for w in works:
        secs[w.result.section] += 1
        for e in w.result.entries:
            c[e.kind.value] += 1
        c["ads"] += len(w.result.ads)
    return {"pages": len(works), "sections": dict(secs), "counts": dict(c)}
