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
    """Group words into text lines (top to bottom, words left to right)."""
    if not words:
        return []
    hs = np.array([w.bbox.height for w in words])
    med_h = float(np.median(hs)) if len(hs) else 10.0
    ws = sorted(words, key=lambda w: (w.bbox.cy, w.bbox.x0))
    lines: list[list[Word]] = []
    centers: list[float] = []
    for w in ws:
        best, best_d = -1, 1e9
        for i in range(len(lines) - 1, max(-1, len(lines) - 4), -1):
            d = abs(centers[i] - w.bbox.cy)
            if d < best_d:
                best, best_d = i, d
        if best >= 0 and best_d <= merge_tol * med_h:
            lines[best].append(w)
            n = len(lines[best])
            centers[best] = centers[best] + (w.bbox.cy - centers[best]) / n
        else:
            lines.append([w])
            centers.append(w.bbox.cy)
    out = [Line(sorted(l, key=lambda w: w.bbox.x0)) for l in lines]
    out.sort(key=lambda l: l.bbox.cy)
    return _split_wide_gaps(out, med_h)


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


def looks_like_heading(line: Line, col: BBox, margin: float, em: float) -> bool:
    """Centred or all-caps short line without numbers."""
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
    if t.startswith(("*", "•")) and centred:
        return True
    if caps and len(letters) >= 3 and (centred or len(toks) <= 3) and "—" not in t[:2]:
        return True
    if centred and all(_HEAD_WORD.match(x) or x.islower() and len(x) <= 4 for x in toks):
        return True
    return False


def assemble_entries(lines: list[Line], col: BBox, numbered: bool = False) -> list[RawEntry]:
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
        if looks_like_heading(ln, col, margin, em):
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
        if DITTO_RE.match(first) and ln.bbox.x0 <= margin + tol * 2.5:
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
