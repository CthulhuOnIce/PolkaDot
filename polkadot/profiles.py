"""Directory-style profiles.

City directories and telephone books share a family resemblance but differ
in ways that pull layout heuristics in opposite directions: a 1950s
telephone book prints dot leaders that look like broken rules and sets four
columns almost edge to edge, while a 1920s Polk directory has worn printed
rules that look like dot leaders and wide gutters.  A profile bundles the
settings for one style so that tuning for one never disturbs another.

``general`` is the all-round default (adaptive, validated on both styles);
``auto`` samples a few pages and picks the best-fitting profile, falling
back to ``general`` when the evidence is weak.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import Optional

from .layout import LayoutConfig

log = logging.getLogger("polkadot.profiles")


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    # relabel automatically detected sections ("telephone" pages cannot
    # occur in a city directory, nor street guides in a telephone book)
    section_map: tuple[tuple[str, str], ...] = ()
    # how category headings are recognised on classified pages:
    # "auto" (decide per page), "any" (centred / starred / caps headings) or
    # "marked" (only "▶ Heading" / "Heading (cont'd)", telephone yellow pages)
    classified_headings: str = "auto"

    def map_section(self, sec: str) -> str:
        return dict(self.section_map).get(sec, sec)


PROFILES: dict[str, Profile] = {
    "general": Profile(
        "general",
        "All-round settings: adapts per page to city directories and telephone books."),
    "polk-1920s": Profile(
        "polk-1920s",
        "R. L. Polk-style city directory, c. 1890s-1940s: alphabetical / street "
        "guide / classified sections, three columns framed by strip, side and "
        "corner-card ads.",
        # layout: the general settings are what this style was validated
        # on; disabling narrow-gutter splitting made pages worse
        section_map=(("telephone", "alphabetical"),),
        classified_headings="any"),
    "telephone-1950s": Profile(
        "telephone-1950s",
        "Mid-century telephone directory: four dense columns of "
        "'Name Address Town----EXCHANGE N-NNNN' white pages, yellow pages with "
        "'▶ Category' headings and display ads.",
        # layout: the general settings; stricter leader rejection
        # (rule_min_cont=0.8, rule_max_gaps_per_inch=8) fixed some white pages
        # but lost ad-box borders on yellow pages - not a net win
        section_map=(("alphabetical", "telephone"), ("street", "telephone")),
        classified_headings="marked"),
}

PROFILE_NAMES = ("auto",) + tuple(PROFILES)


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {', '.join(PROFILE_NAMES)}")


# --------------------------------------------------------------------------
# auto-detection
# --------------------------------------------------------------------------

_PHONE_END = re.compile(r"[A-Z][A-Za-z]{0,7}\.?\s?\d\s?-\s?\d{4}\s*[A-Z]?\W*$")
_RESIDENCE = re.compile(r"\s(?:h|r|bds)\s+(?:\d|same\b)")
_HOUSE_NO = re.compile(r"^\d{1,5}\s+[A-Z]")


@dataclass
class Detection:
    profile: str
    scores: dict
    pages: list[int]
    reason: str


def detect_profile(doc_path: str, sample: int = 8, dpi: int = 200) -> Detection:
    """Pick a profile by sampling pages from the body of the book.

    Each sampled page is segmented and a strip of its first listing column
    is read (PDF text layer if usable, else OCR).  Telephone books end most
    lines in an exchange number; city directories mark residences with
    ``h``/``r``/``bds`` or (street guides) start lines with house numbers."""
    import pymupdf

    from . import imaging as im
    from . import ocr as ocrmod
    from .layout import analyze_layout
    from .models import BBox, RegionKind
    from .pipeline import render_page
    from .text import group_lines

    with pymupdf.open(doc_path) as doc:
        n = doc.page_count
        lo, hi = int(n * 0.2), max(int(n * 0.2) + 1, int(n * 0.8))
        step = max(1, (hi - lo) // sample)
        pages = list(range(lo, hi, step))[:sample] or [0]
        votes = {"telephone-1950s": 0, "polk-1920s": 0}
        cols: list[int] = []
        for p in pages:
            page = doc[p]
            img = render_page(page, dpi)
            img = im.deskew(img, im.estimate_skew(im.binarize(im.to_gray(img), dpi)))
            lay = analyze_layout(img, dpi, p)
            text = [r for r in lay.regions if r.kind is RegionKind.TEXT]
            cols.append(len(text))
            if not text:
                continue
            col = max(text, key=lambda r: r.bbox.area)
            strip = BBox(col.bbox.x0, col.bbox.y0, col.bbox.x1,
                         col.bbox.y0 + min(col.bbox.height, dpi * 2.5))
            words = ocrmod.pdf_words(page, dpi / 72.0)
            words = ocrmod.words_in(words, strip)
            if (len(words) < 20 or ocrmod.text_quality(words) < 0.8) and ocrmod.tesseract_available():
                words, _, _ = ocrmod.ocr_region(img, strip, ocrmod.OcrConfig(), dpi, 6)
            lines = [l.text for l in group_lines(words)]
            if len(lines) < 8:
                continue
            phone = sum(1 for t in lines if _PHONE_END.search(t)) / len(lines)
            city = sum(1 for t in lines if _RESIDENCE.search(" " + t) or _HOUSE_NO.match(t)) / len(lines)
            if phone >= 0.3 and phone > city:
                votes["telephone-1950s"] += 1
            elif city >= 0.15:
                votes["polk-1920s"] += 1
    read = sum(votes.values())
    best = max(votes, key=votes.get)
    scores = dict(votes, sampled=len(pages), median_columns=sorted(cols)[len(cols) // 2] if cols else 0)
    if read >= 3 and votes[best] >= 0.7 * read:
        return Detection(best, scores, [p + 1 for p in pages],
                         f"{votes[best]} of {read} readable sample pages look like {best}")
    return Detection("general", scores, [p + 1 for p in pages],
                     "no clear match among the sampled pages")


__all__ = ["Profile", "PROFILES", "PROFILE_NAMES", "get_profile", "detect_profile", "Detection",
           "replace"]
