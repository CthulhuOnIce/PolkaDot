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

    for r in layout.regions:
        use_pdf = False
        pdf_words = ocrmod.words_in(pdf_all, r.bbox) if pdf_all else []
        if cfg.text_source == "pdf":
            use_pdf = True
        elif cfg.text_source == "auto":
            q = ocrmod.text_quality(pdf_words)
            enough = len(pdf_words) >= max(3, r.bbox.area / (layout.body_h ** 2 * 400))
            if r.kind.is_ad or r.kind is RegionKind.AD_LABEL:
                use_pdf = (not have_ocr) or (not cfg.ocr_ads and q >= cfg.min_text_quality and enough)
            else:
                use_pdf = (q >= cfg.min_text_quality and enough) or not have_ocr
        if use_pdf:
            words = pdf_words
            r.lines = group_lines(words)
            frames[r.id] = [(l.text, l.height, l.bbox.y0, l.bbox.y1, l.bbox.x0, l.conf) for l in r.lines]
            r.meta["text_source"] = "pdf"
            r.meta["text_quality"] = round(ocrmod.text_quality(words), 3)
            sources.add("pdf")
            if not have_ocr and cfg.text_source == "auto" and r.meta["text_quality"] < cfg.min_text_quality:
                res.warnings.append(f"{r.id}: low quality text layer and no OCR engine")
            continue
        if r.kind.is_ad or r.kind is RegionKind.AD_LABEL:
            tall = r.bbox.height > r.bbox.width * 1.6
            rots = (0, 270, 90) if (tall and cfg.ocr.try_rotations) else (0,)
            words, rot, fwords = _ocr_with_frame(img, r.bbox, cfg, rots, cfg.ocr.ad_psm, False)
            if cfg.ocr.ad_fallback_psm and (len(words) < 3 or ocrmod.text_quality(words) < 0.6):
                alt = _ocr_with_frame(img, r.bbox, cfg, rots, cfg.ocr.ad_fallback_psm, False)
                if ocrmod._score(alt[0]) > ocrmod._score(words):
                    words, rot, fwords = alt
            r.rotation = rot
        else:
            H, W = img.shape[:2]
            box = r.bbox.pad(dpi * 0.03, dpi * 0.01, limit=BBox(0, 0, W, H))
            words, rot, fwords = _ocr_with_frame(img, box, cfg, (0,), cfg.ocr.text_psm)
        # group lines in the (possibly rotated) reading frame
        flines = group_lines(fwords)
        lookup = {id(fw): pw for fw, pw in zip(fwords, words)}
        r.lines = [Line([lookup[id(w)] for w in fl.words]) for fl in flines]
        frames[r.id] = [(fl.text, fl.height, fl.bbox.y0, fl.bbox.y1, fl.bbox.x0, fl.conf)
                        for fl in flines]
        r.meta["text_source"] = "ocr"
        r.meta["text_quality"] = round(ocrmod.text_quality(words), 3)
        sources.add("ocr")
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


def section_features(res: PageResult) -> dict:
    lines = [l for r in res.regions if r.kind is RegionKind.TEXT for l in r.lines]
    n = len(lines)
    ad_area = sum(r.bbox.area for r in res.regions if r.kind.is_ad)
    content_area = res.content_bbox.area if res.content_bbox else 1.0
    texts = [l.text for l in lines]
    num_start = sum(1 for t in texts if re.match(r"^\d{1,5}\b", t))
    res_hits = sum(1 for t in texts if _RES_RE.search(" " + t))
    see = sum(1 for t in texts if "(See" in t or "(see" in t)
    star = sum(1 for t in texts if t.lstrip().startswith("*"))
    all_text = " ".join(r.text for r in res.regions)
    markers = [sec for rx, sec in _MARKERS if rx.search(all_text)]
    return {"n_lines": n, "ad_frac": ad_area / max(1.0, content_area),
            "num_start": num_start / max(1, n), "res": res_hits / max(1, n),
            "see": see / max(1, n), "star": star, "markers": markers}


def guess_section(f: dict) -> tuple[str, float]:
    """Return (section, confidence 0..1) from one page's features."""
    if f["n_lines"] < 8:
        return ("ads" if f["ad_frac"] > 0.4 else "front"), 0.6
    if f["num_start"] > 0.35:
        return "street", min(1.0, f["num_start"] * 1.6)
    if f["res"] > 0.18:
        return "alphabetical", min(1.0, f["res"] * 2.5)
    if f["see"] > 0.02 or f["star"] >= 2:
        return "classified", 0.6
    return "misc", 0.3


def assign_sections(works: list[PageWork], overrides: dict[int, str]) -> None:
    """Per-page guesses smoothed with document context.

    Directories are organised in long contiguous sections; a weak guess is
    replaced by the section in force (from the previous confident page or
    an explicit section title such as "Classified Business Directory").
    """
    current: Optional[str] = None
    for w in works:
        p = w.result.page
        if p in overrides:
            w.result.section = overrides[p]
            current = overrides[p] if overrides[p] not in ("ads",) else current
            continue
        guess, conf = guess_section(w.features)
        for m in w.features.get("markers", []):
            if guess in ("front", "misc", "ads") or conf < 0.5:
                current = m
        if guess == "ads":
            w.result.section = "ads"
            continue
        if conf >= 0.5 or current is None:
            w.result.section = guess
            if guess not in ("front",):
                current = guess
        else:
            w.result.section = current if current in ("classified", "alphabetical", "street") \
                and w.features["n_lines"] >= 20 else guess


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
                if fields.get("house_number") and not fields.get("location"):
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
            fields.setdefault("phones", P.find_phones(text)) if "phones" not in fields else None
            if not fields.get("phones"):
                fields.pop("phones", None)
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
        f = P.ad_fields(frame, text)
        cat = None
        if r.meta.get("label_region") in labels:
            cat = P.clean_heading(labels[r.meta["label_region"]].text).title() or None
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
        with ProcessPoolExecutor(max_workers=workers) as ex:
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


def summarize(works: list[PageWork]) -> dict:
    c = Counter()
    secs = Counter()
    for w in works:
        secs[w.result.section] += 1
        for e in w.result.entries:
            c[e.kind.value] += 1
        c["ads"] += len(w.result.ads)
    return {"pages": len(works), "sections": dict(secs), "counts": dict(c)}
