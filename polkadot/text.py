"""Turning words into lines and lines into logical listings.

Directory listings use a *hanging indent*: the first line of an entry starts
at the column's left margin and every continuation line is indented.  Name
headings (``KARR``), street headings and business-category headings
(``*Floral Designs``) are set centred or in capitals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .models import BBox, Line, Word

DITTO_RE = re.compile(r'^(?:["“”„«»\'`]{1,2}|,,|\'\')$')


def group_lines(words: list[Word], merge_tol: float = 0.5) -> list[Line]:
    """Group words into text lines (top to bottom, words left to right).

    A word joins a line when their vertical extents overlap by more than
    ``merge_tol`` of the smaller height - robust to mixed type sizes on one
    line (display ads) and to slightly sloping baselines.
    """
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w.bbox.cy, w.bbox.x0))
    lines: list[list[Word]] = []
    spans: list[tuple[float, float]] = []
    for w in ws:
        best, best_ov = -1, 0.0
        for i in range(len(lines) - 1, max(-1, len(lines) - 6), -1):
            y0, y1 = spans[i]
            ov = min(y1, w.bbox.y1) - max(y0, w.bbox.y0)
            small = min(y1 - y0, w.bbox.height) or 1.0
            # do not let a word bridge into a line it overlaps horizontally
            if any(o.bbox.x_overlap(w.bbox) > 0.3 * min(o.bbox.width, w.bbox.width) for o in lines[i]):
                continue
            if ov / small > best_ov:
                best, best_ov = i, ov / small
        if best >= 0 and best_ov >= merge_tol:
            lines[best].append(w)
            # track the line's core band (median-ish) rather than its union
            hs = sorted(x.bbox.height for x in lines[best])
            cy = sorted(x.bbox.cy for x in lines[best])[len(lines[best]) // 2]
            hh = hs[len(hs) // 2] / 2.0
            spans[best] = (cy - hh, cy + hh)
        else:
            lines.append([w])
            spans.append((w.bbox.y0, w.bbox.y1))
    out = [Line(sorted(l, key=lambda w: w.bbox.x0)) for l in lines]
    out.sort(key=lambda l: sorted(w.bbox.cy for w in l.words)[len(l.words) // 2])
    return out


def _split_wide_gaps(lines: list[Line], med_h: float) -> list[Line]:
    """Keep lines intact; hook for splitting accidental cross-column joins."""
    return lines


# --------------------------------------------------------------------------
# Entry assembly
# --------------------------------------------------------------------------


@dataclass
class RawEntry:
    lines: list[Line]
    is_heading: bool = False
    indent_level: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def bbox(self) -> BBox:
        return BBox.enclosing(l.bbox for l in self.lines)  # type: ignore[return-value]

    @property
    def text(self) -> str:
        return join_lines([l.text for l in self.lines])


def join_lines(parts: list[str]) -> str:
    """Join printed lines, repairing words hyphenated across lines."""
    out = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not out:
            out = p
            continue
        if re.search(r"[A-Za-z]-$", out) and p[:1].islower():
            out = out[:-1] + p
        else:
            out = out + " " + p
    return re.sub(r"\s+", " ", out).strip()


_HEAD_WORD = re.compile(r"^[*•]?[A-Z][A-Za-z'’&.,\-—]*$")


def looks_like_heading(line: Line, col: BBox, margin: float, em: float,
                       style: str = "any") -> bool:
    """Centred or all-caps short line without numbers.

    ``style``: "caps" accepts only capitalised headings (surname / street
    headings: ``KARR``, ``ADAMS—Contd``); "any" also accepts centred title
    case (business categories: ``*Floral Designs``, ``Florists``)."""
    t = line.text.strip()
    if not t or any(ch.isdigit() for ch in t):
        return False
    toks = t.split()
    if len(toks) > 7:
        return False
    b = line.bbox
    letters = [c for c in t if c.isalpha()]
    caps = letters and sum(c.isupper() for c in letters) / len(letters) > 0.85
    centred = (abs(b.cx - col.cx) < col.width * 0.12 and b.width < col.width * 0.75
               and b.x0 > margin + em * 0.8)
    if caps and len(letters) >= 3 and (centred or len(toks) <= 3) and "—" not in t[:2]:
        return True
    if style == "caps":
        return False
    indented = b.x0 > margin + em * 0.8
    # OCR often reads the star that marks paid headings as a quote mark
    starred = re.match(r"""^[*•"“”'‘’]+\s*[A-Z]""", t) is not None
    if starred and (centred or indented) and len(toks) <= 6:
        return True
    loose_centre = abs(b.cx - col.cx) < col.width * 0.2 and b.width < col.width * 0.8 and indented
    if (centred or loose_centre) and all(_HEAD_WORD.match(x) or x.islower() and len(x) <= 4 for x in toks):
        return True
    return False


def assemble_entries(lines: list[Line], col: BBox, numbered: bool = False,
                     heading_style: str = "any") -> list[RawEntry]:
    """Split a column's lines into hanging-indent entries.

    ``numbered``: street-guide columns where house numbers sit in a gutter
    to the left of names; a line starting with a number starts an entry
    and unnumbered lines at the name margin are further residents of the
    same house (we still emit them as their own entries, flagged).
    """
    if not lines:
        return []
    hs = sorted(l.height for l in lines if l.words)
    em = hs[len(hs) // 2] if hs else 10.0
    x0s = np.array([l.bbox.x0 for l in lines])
    margin = float(np.percentile(x0s, 10))
    tol = max(em * 0.55, 3.0)

    name_margin = _name_margin(lines) if numbered else None
    entries: list[RawEntry] = []
    cur: Optional[RawEntry] = None
    for ln in lines:
        if not ln.words:
            continue
        if looks_like_heading(ln, col, margin, em, heading_style):
            if cur:
                entries.append(cur)
            entries.append(RawEntry([ln], is_heading=True))
            cur = None
            continue
        first = ln.words[0].text
        starts_new = ln.bbox.x0 <= margin + tol
        if numbered and re.match(r"^\d", first):
            starts_new = True
        if numbered and not starts_new:
            # unnumbered resident line aligned with the names column
            if name_margin is not None and abs(ln.bbox.x0 - name_margin) <= tol:
                starts_new = True
        if (DITTO_RE.match(first) or re.match(r"""^["“”‘’„]+[A-Z]""", first)) \
                and ln.bbox.x0 <= margin + tol * 2.5:
            starts_new = True
        if starts_new or cur is None:
            if cur:
                entries.append(cur)
            cur = RawEntry([ln])
        else:
            cur.lines.append(ln)
    if cur:
        entries.append(cur)
    return entries


def _name_margin(lines: list[Line]) -> Optional[float]:
    """x position where names start in a street-guide column."""
    xs = [l.words[1].bbox.x0 for l in lines
          if len(l.words) >= 2 and re.match(r"^\d", l.words[0].text)]
    return float(np.median(xs)) if xs else None
