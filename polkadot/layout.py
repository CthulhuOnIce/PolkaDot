"""Page segmentation: split a directory page into listing columns, running
heads and the many kinds of advertising that Polk-style directories wrap
around the text (top/bottom strips, side columns, boxed inline ads, and whole
pages of boxed ads).

Strategy
--------
1.  Rules and fully enclosed boxes ("cells") are detected on the ink mask.
    Boxes whose content does not look like body type are ads.
2.  The remaining ink is split with a recursive XY-cut (whitespace gaps and
    ruling lines are cut candidates) into leaf blocks, which are classified
    as body text or display matter by glyph-size statistics.
3.  Body-text leaves are merged into columns.  Tall columns form the *body*
    of the page; everything in the margins around the body (above, below,
    left, right) that is not the running head is advertising, labelled by
    the margin it sits in.  Display matter inside the body is an inline ad.
4.  Pages without a body (advertising pages, title pages) become a set of
    boxed ads / text blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from . import imaging as im
from .models import BBox, Region, RegionKind


@dataclass
class LayoutConfig:
    min_column_height: float = 0.22     # fraction of content height
    min_column_width_em: float = 8.0    # in body x-heights


@dataclass
class Leaf:
    bbox: BBox
    stats: im.BlockStats
    orientation: str = "h"
    is_text: bool = False


@dataclass
class PageLayout:
    content: BBox
    body: Optional[BBox]
    body_h: float
    regions: list[Region] = field(default_factory=list)
    leaves: list[Leaf] = field(default_factory=list)
    rules: Optional[im.RuleLines] = None
    cells: list[im.Cell] = field(default_factory=list)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _ink_bbox(mask: np.ndarray, box: BBox) -> Optional[BBox]:
    sub = im.crop(mask, box)
    if sub.size == 0:
        return None
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return None
    return BBox(box.x0 + xs.min(), box.y0 + ys.min(),
                box.x0 + xs.max() + 1, box.y0 + ys.max() + 1)


def _gaps(profile: np.ndarray, min_len: int, noise: float = 0.0) -> list[tuple[int, int]]:
    """Runs of (near) empty positions, excluding the ends of the profile."""
    empty = profile <= noise
    out: list[tuple[int, int]] = []
    i, n = 0, len(profile)
    # skip leading / trailing empties
    while i < n and empty[i]:
        i += 1
    end = n
    while end > i and empty[end - 1]:
        end -= 1
    while i < end:
        if empty[i]:
            j = i
            while j < end and empty[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def text_orientation(mask: np.ndarray, box: BBox, body_h: float) -> str:
    """Guess whether the type in ``box`` runs horizontally or vertically."""
    sub = im.crop(mask, box)
    if sub.size == 0 or sub.max() == 0:
        return "h"
    k = max(3, int(body_h * 2.0))
    sh = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    sv = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))

    def elong(m: np.ndarray, horizontal: bool) -> float:
        n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return 0.0
        st = st[1:]
        big = st[:, cv2.CC_STAT_AREA] >= body_h * body_h
        st = st[big]
        if len(st) == 0:
            return 0.0
        w, h = st[:, 2].astype(float), st[:, 3].astype(float)
        a = st[:, 4].astype(float)
        r = (w / np.maximum(h, 1)) if horizontal else (h / np.maximum(w, 1))
        return float((np.minimum(r, 20) * a).sum() / a.sum())

    eh, ev = elong(sh, True), elong(sv, False)
    if ev > eh * 1.6 and ev > 3:
        return "v"
    return "h"


# --------------------------------------------------------------------------
# XY cut
# --------------------------------------------------------------------------


class _Cutter:
    def __init__(self, ink: np.ndarray, rules: im.RuleLines, body_h: float, dpi: int):
        self.ink = ink
        self.rules = rules
        self.body_h = body_h
        self.dpi = dpi
        self.leaves: list[BBox] = []

    def run(self, box: BBox, depth: int = 0) -> None:
        tight = _ink_bbox(self.ink, box)
        if tight is None:
            return
        box = tight
        if depth > 12 or (box.width < self.body_h * 3 and box.height < self.body_h * 3):
            self.leaves.append(box)
            return
        cut = self._best_cut(box)
        if cut is None:
            self.leaves.append(box)
            return
        axis, positions = cut
        edges = [0] + positions + [None]
        for a, b in zip(edges[:-1], edges[1:]):
            if axis == "x":
                sub = BBox(box.x0 + a, box.y0, box.x1 if b is None else box.x0 + b, box.y1)
            else:
                sub = BBox(box.x0, box.y0 + a, box.x1, box.y1 if b is None else box.y0 + b)
            if not sub.is_empty():
                self.run(sub, depth + 1)

    def _rule_cuts(self, box: BBox, axis: str, frac: float = 0.5) -> list[int]:
        out = []
        m = self.body_h * 0.6
        if axis == "y":
            for s in self.rules.h_segments:
                if box.y0 + m < s.cy < box.y1 - m and s.x_overlap(box) >= frac * box.width:
                    out.append(int(s.cy - box.y0))
        else:
            for s in self.rules.v_segments:
                if box.x0 + m < s.cx < box.x1 - m and s.y_overlap(box) >= frac * box.height:
                    out.append(int(s.cx - box.x0))
        return sorted(set(out))

    def _best_cut(self, box: BBox) -> Optional[tuple[str, list[int]]]:
        sub = im.crop(self.ink, box) > 0
        bh = self.body_h
        # ruling lines are the strongest separators
        # (a horizontal rule must cross most of the block - a boxed notice
        # in one corner must not slice through neighbouring columns)
        for axis, frac in (("y", 0.5), ("x", 0.5)):
            rc = self._rule_cuts(box, axis, frac)
            if rc:
                return axis, rc
        colp = sub.sum(axis=0)
        rowp = sub.sum(axis=1)
        cands: list[tuple[float, str, list[int]]] = []
        # vertical whitespace channels -> columns
        # allow a little noise in the gutter (faint dotted column rules,
        # specks) - a real gutter is still almost empty
        xg = _gaps(colp, max(2, int(bh * 1.0)))
        xg += [g for g in _gaps(colp, max(2, int(bh * 1.8)), noise=max(1, int(box.height * 0.04)))
               if not any(a <= g[0] < b or g[0] <= a < g[1] for a, b in xg)]
        xg = sorted(g for g in xg if g[0] >= bh * 8 and box.width - g[1] >= bh * 8)
        if not xg and box.width >= bh * 40 and box.height > bh * 30:
            # dense listings: very narrow but clean gutters between wide columns
            thin = _gaps(colp, max(2, int(bh * 0.2)), noise=max(1, int(box.height * 0.03)))
            xg = [g for g in thin if g[0] >= bh * 20 and box.width - g[1] >= bh * 20]
        if xg and box.height > bh * 6:
            score = max(b - a for a, b in xg) / bh
            cands.append((score, "x", [(a + b) // 2 for a, b in xg]))
        # horizontal whitespace bands -> stacked blocks
        yg = _gaps(rowp, max(2, int(bh * 1.6)))
        if yg:
            score = max(b - a for a, b in yg) / bh
            cands.append((score * 0.9, "y", [(a + b) // 2 for a, b in yg]))
        if not cands:
            # last resort: shorter rules (e.g. the top rule of a strip ad
            # that only spans part of the text width)
            for axis in ("y", "x"):
                rc = self._rule_cuts(box, axis, 0.3)
                if rc:
                    return axis, rc
            return None
        cands.sort(key=lambda c: -c[0])
        _, axis, pos = cands[0]
        return axis, pos


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------


LAYOUT_DPI = 200


def analyze_layout(img: np.ndarray, dpi: int, page_no: int = 0,
                   cfg: Optional[LayoutConfig] = None,
                   gray: Optional[np.ndarray] = None,
                   ink: Optional[np.ndarray] = None) -> PageLayout:
    """Segment a page.  Analysis runs at a fixed working resolution
    (``LAYOUT_DPI``) so that thresholds behave identically whatever DPI the
    page was rendered at; results are scaled back to ``img`` pixels."""
    cfg = cfg or LayoutConfig()
    if abs(dpi - LAYOUT_DPI) <= 5:
        return _analyze(img, dpi, page_no, cfg, gray, ink)
    f = LAYOUT_DPI / float(dpi)
    small = cv2.resize(img, (max(1, int(round(img.shape[1] * f))), max(1, int(round(img.shape[0] * f)))),
                       interpolation=cv2.INTER_AREA)
    lay = _analyze(small, LAYOUT_DPI, page_no, cfg, None, None)
    inv = 1.0 / f
    for r in lay.regions:
        r.bbox = r.bbox.scale(inv)
    for l in lay.leaves:
        l.bbox = l.bbox.scale(inv)
    lay.content = lay.content.scale(inv)
    lay.body = lay.body.scale(inv) if lay.body is not None else None
    lay.body_h *= inv
    lay.cells = []
    if lay.rules is not None:
        lay.rules = im.RuleLines(lay.rules.horizontal, lay.rules.vertical,
                                 [s.scale(inv) for s in lay.rules.h_segments],
                                 [s.scale(inv) for s in lay.rules.v_segments])
    return lay


def _analyze(img: np.ndarray, dpi: int, page_no: int, cfg: LayoutConfig,
             gray: Optional[np.ndarray], ink: Optional[np.ndarray]) -> PageLayout:
    gray = im.to_gray(img) if gray is None else gray
    ink = im.binarize(gray, dpi) if ink is None else ink
    content = im.find_content_bbox(gray, ink, dpi)

    H, W = ink.shape
    window = content.pad(dpi * 0.2, limit=BBox(0, 0, W, H))
    rules = im.find_rule_lines(ink, window, dpi)
    content = im.expand_to_rules(content, rules, dpi).intersect(window)
    content = im.trim_outside_frame(content, rules, ink, dpi)
    rules = im.find_rule_lines(ink, content.pad(2, limit=BBox(0, 0, W, H)), dpi)
    rule_mask = cv2.dilate(rules.combined, np.ones((3, 3), np.uint8))
    text_ink = cv2.bitwise_and(ink, cv2.bitwise_not(rule_mask))
    # restrict to content and drop speckle
    cmask = np.zeros_like(text_ink)
    x0, y0, x1, y1 = content.to_int()
    cmask[y0:y1, x0:x1] = 255
    text_ink = cv2.bitwise_and(text_ink, cmask)
    text_ink = cv2.morphologyEx(text_ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    comps = im.components(text_ink)
    body_h = im.estimate_text_height(comps, dpi)

    cells = im.find_cells(rules, content, dpi)
    layout = PageLayout(content=content, body=None, body_h=body_h, rules=rules, cells=cells)

    # ---- 1. classify enclosed boxes ------------------------------------
    ad_cells: list[BBox] = []
    for c in sorted(cells, key=lambda c: -c.bbox.area):
        if c.bbox.area > content.area * 0.85:
            continue
        if c.bbox.area > content.area * 0.3 and c.bbox.width > content.width * 0.45:
            continue          # a frame around (part of) the body, not an ad
        inner = _ink_bbox(text_ink, c.bbox)
        if inner is None:
            continue
        st = im.block_stats(comps, c.bbox, body_h, text_ink)
        orient = text_orientation(text_ink, c.bbox, body_h)
        lines_est = inner.height / max(1.0, body_h * 2.2)
        body_like = st.looks_like_body_text() and orient == "h" and lines_est >= 4
        if body_like and c.bbox.height > content.height * cfg.min_column_height:
            continue          # a ruled frame around listing columns
        if body_like and c.bbox.width > content.width * 0.25 and lines_est > 12:
            continue
        if any(a.contains(c.bbox, tol=body_h) for a in ad_cells):
            continue          # nested inside an ad already found
        ad_cells.append(c.bbox)

    ruled_cells = list(ad_cells)
    for b in im.find_solid_panels(ink, content, dpi):
        if b.area > content.area * 0.5:
            continue
        if any(a.contains(b, tol=body_h) for a in ad_cells):
            continue
        # a panel already inside a body-text frame is still an ad
        ad_cells = [a for a in ad_cells if not b.contains(a, tol=body_h)]
        ad_cells.append(b)

    work = text_ink.copy()
    for b in ad_cells:
        x0, y0, x1, y1 = b.pad(2).to_int()
        work[max(0, y0):y1, max(0, x0):x1] = 0

    # ---- 2. XY-cut the rest ---------------------------------------------
    cutter = _Cutter(work, rules, body_h, dpi)
    cutter.run(content)
    leaves: list[Leaf] = []
    for b in cutter.leaves:
        st = im.block_stats(comps, b, body_h, work)
        if st.n == 0 and b.area < (body_h * 3) ** 2:
            continue   # specks
        orient = text_orientation(work, b, body_h) if b.height > body_h * 6 else "h"
        is_text = (st.n >= 3 and st.body_frac >= 0.6 and st.large_frac < 0.35
                   and orient == "h")
        leaves.append(Leaf(b, st, orient, is_text))
    layout.leaves = leaves

    # ---- 3. columns ------------------------------------------------------
    text_leaves = [l for l in leaves if l.is_text]
    columns = _merge_columns([l.bbox for l in text_leaves], body_h, rules.h_segments)
    min_h = content.height * cfg.min_column_height
    min_w = body_h * cfg.min_column_width_em
    body_cols = []
    for c in columns:
        if c.height < min_h or c.width < min_w:
            continue
        st = im.block_stats(comps, c, body_h, work)
        # listing columns are dense, uniform small type; a tall ad set in
        # mixed sizes (common on advertising pages) is not
        if st.large_frac > 0.2 or st.body_frac < 0.75:
            continue
        body_cols.append(c)
    # second chance: a column of the same size as an accepted one, set in
    # small type but with many bold / capitalised listings (paid entries
    # such as "PHOENIX FIRE INSURANCE CO") is a listing column too
    for c in columns:
        if c in body_cols or c.height < min_h or c.width < min_w:
            continue
        # (a boxed ad at the head of the column may shorten it)
        twin = any(abs(c.width - b.width) <= 0.15 * b.width and c.height >= 0.5 * b.height
                   and c.y0 >= b.y0 - body_h * 3 and c.y1 <= b.y1 + body_h * 3
                   and c.x_overlap(b) == 0 for b in body_cols)
        if not twin:
            continue
        st = im.block_stats(comps, c, body_h, work)
        if st.large_frac <= 0.5 and st.body_frac >= 0.85:
            body_cols.append(c)
    # body columns must not be absurdly narrow compared to the widest one
    if body_cols:
        wmax = max(c.width for c in body_cols)
        body_cols = [c for c in body_cols if c.width >= wmax * 0.35]

    regions: list[Region] = []
    rid = 0

    def add(kind: RegionKind, box: BBox, **meta) -> Region:
        nonlocal rid
        r = Region(id=f"p{page_no + 1}_r{rid}", kind=kind, bbox=box, meta=dict(meta))
        rid += 1
        regions.append(r)
        return r

    if not body_cols:
        # Advertising page / title page.  Split the page into panels along
        # its ruling lines; each inked panel is one ad (thin ones become
        # captions of the ad below them).
        panels = _rule_panels(content, rules, body_h)
        for pnl in panels:
            inner = _ink_bbox(text_ink, pnl)
            if inner is None or min(inner.width, inner.height) < body_h * 1.2:
                continue
            for g in _cluster_keep_cells(
                    [b for b in ad_cells if pnl.overlap_ratio(b) > 0 and b.overlap_ratio(pnl) > 0.5]
                    + [l.bbox for l in leaves if l.bbox.overlap_ratio(pnl) > 0.5],
                    ruled_cells, body_h, body_h * 4.0):
                g = g.intersect(pnl) if not g.intersect(pnl).is_empty() else g
                if min(g.width, g.height) >= body_h * 1.2:
                    add(RegionKind.AD_BOX, g)
        _merge_overlapping(regions)
        _attach_labels(regions, body_h)
        _order(regions)
        layout.regions = regions
        return layout

    body_cols = [piece for c in body_cols for piece in _split_column(work, c, body_h)]
    # a much narrower "column" at the outer edge of the body is a side ad
    # set in small type (e.g. a transfer company's list of services)
    if len(body_cols) >= 3:
        ws = sorted(c.width for c in body_cols)
        med = ws[len(ws) // 2]
        body_cols.sort(key=lambda c: c.x0)
        for edge in (body_cols[0], body_cols[-1]):
            if edge.width < med * 0.62 and edge in body_cols:
                body_cols.remove(edge)
    body = BBox.enclosing(body_cols)
    assert body is not None
    layout.body = body

    # leaves/cells outside the body ---------------------------------------
    header: Optional[BBox] = None
    footer: Optional[BBox] = None
    items: list[BBox] = list(ad_cells)
    for l in leaves:
        b = l.bbox
        if any(c.contains(b, tol=2) for c in body_cols):
            continue
        covered = sum(b.intersection_area(c) for c in body_cols)
        if l.is_text and covered >= 0.7 * max(1.0, b.area):
            continue          # a leaf that was split into several columns
        items.append(b)

    # running head: single text line just above (or below) the body
    def _line_like(b: BBox) -> bool:
        return b.height <= body_h * 3.2 and b.width >= body.width * 0.3

    above = [b for b in items if b.y1 <= body.y0 + body_h and _line_like(b)
             and body.y0 - b.y1 < body_h * 4 and b not in ad_cells]
    if above:
        header = max(above, key=lambda b: b.y1)
        # also merge other line-like pieces on the same baseline
        same = [b for b in above if b.y_overlap(header) > 0.5 * min(b.height, header.height)]
        header = BBox.enclosing(same)
        items = [b for b in items if b not in same]
    below = [b for b in items if b.y0 >= body.y1 - body_h and _line_like(b)
             and b.y0 - body.y1 < body_h * 3 and b not in ad_cells
             and b.height <= body_h * 2.2]
    if below:
        footer = min(below, key=lambda b: b.y0)
        items = [b for b in items if b is not footer]

    # inside-body items: small display lines become part of the columns
    # (category headings), bigger ones are inline ads.
    inside, outside = [], []
    for b in items:
        if b.overlap_ratio(body) > 0.6:
            inside.append(b)
        else:
            outside.append(b)
    text_boxes = {l.bbox for l in text_leaves}
    inline_ads: list[BBox] = []
    for b in inside:
        if b in ad_cells or (b not in text_boxes and b.height > body_h * 4.5
                             and b.width > body_h * 8):
            inline_ads.append(b)
        else:
            # attach to the column it overlaps most
            best = min(body_cols, key=lambda c: (-c.x_overlap(b), c.gap_to(b)), default=None)
            if best is not None and b.width > best.width * 1.3:
                # spans several columns: a running head, or display matter
                if header is None and b.height <= body_h * 3.2 and b.y0 <= body.y0 + body_h * 6:
                    header = b
                else:
                    inline_ads.append(b)
                continue
            if best is not None and (best.x_overlap(b) > 0 or best.gap_to(b) < body_h * 3):
                i = body_cols.index(best)
                body_cols[i] = best.union(b)
    inline_ads = _cluster_keep_cells(inline_ads, ruled_cells, body_h)

    # outside: group by margin zone
    zone_items: dict[RegionKind, list[BBox]] = {}
    for b in outside:
        kind = _zone(b, body, content, cfg)
        zone_items.setdefault(kind, []).append(b)
    for kind, bs in zone_items.items():
        gap = body_h * (6.0 if kind is RegionKind.AD_SIDE else 2.5)
        for g in _cluster_keep_cells(bs, ruled_cells, body_h, gap):
            if min(g.width, g.height) < max(body_h * 1.5, dpi * 0.18) or g.area < (body_h * 3) ** 2:
                continue
            add(kind, g)
    for g in inline_ads:
        add(RegionKind.AD_INLINE, g)
    if header is not None:
        add(RegionKind.HEADER, header)
    if footer is not None:
        add(RegionKind.FOOTER, footer)
    body_cols.sort(key=lambda c: c.x0)
    for i, c in enumerate(body_cols):
        r = add(RegionKind.TEXT, c)
        r.column = i
    _drop_slivers(regions, rules, content, dpi)
    _merge_overlapping(regions)
    _attach_labels(regions, body_h)
    _order(regions)
    layout.regions = regions
    return layout


# --------------------------------------------------------------------------
# grouping helpers
# --------------------------------------------------------------------------


def _merge_columns(boxes: list[BBox], body_h: float,
                   hrules: Optional[list[BBox]] = None) -> list[BBox]:
    """Merge vertically stacked text blocks that share a column.

    Blocks separated by a horizontal rule are never merged (the rule is what
    separates listings from a strip ad below / above them)."""
    hrules = hrules or []
    groups = sorted(boxes, key=lambda b: (b.x0, b.y0))
    changed = True
    while changed:
        changed = False
        out: list[BBox] = []
        used = [False] * len(groups)
        for i, a in enumerate(groups):
            if used[i]:
                continue
            cur = a
            for j in range(i + 1, len(groups)):
                if used[j]:
                    continue
                b = groups[j]
                ov = cur.x_overlap(b)
                narrow = min(cur.width, b.width)
                wide = max(cur.width, b.width)
                vgap = max(0.0, max(cur.y0, b.y0) - min(cur.y1, b.y1))
                if narrow <= 0 or vgap > body_h * 8:
                    continue
                lo, hi = (cur, b) if cur.y0 <= b.y0 else (b, cur)
                blocked = any(lo.y1 - body_h <= r.cy <= hi.y0 + body_h
                              and r.x_overlap(BBox(max(cur.x0, b.x0), 0, min(cur.x1, b.x1), 1))
                              >= 0.5 * max(1.0, ov)
                              for r in hrules)
                if blocked:
                    continue
                same_col = ov >= 0.7 * narrow and narrow >= 0.7 * wide
                # a short fragment (heading, one-line entry) inside a column
                small = b if b.width <= cur.width else cur
                big = cur if small is b else b
                fragment = (small.height <= body_h * 5
                            and small.x0 >= big.x0 - body_h and small.x1 <= big.x1 + body_h)
                if same_col or fragment:
                    cur = cur.union(b)
                    used[j] = True
                    changed = True
            out.append(cur)
            used[i] = True
        groups = out
    return groups


def _split_column(ink: np.ndarray, col: BBox, body_h: float, depth: int = 0) -> list[BBox]:
    """Split a text block that really holds several columns.

    Inside known listing text we can afford to tolerate more noise in the
    gutter (faint dotted rules, a heading that pokes across) than the
    generic XY-cut does."""
    if depth > 4 or col.width < body_h * 20:
        return [col]
    colp = (im.crop(ink, col) > 0).sum(axis=0)
    gaps = _gaps(colp, max(2, int(body_h * 1.5)), noise=max(1, int(col.height * 0.08)))
    gaps = [g for g in gaps if g[0] >= body_h * 10 and col.width - g[1] >= body_h * 10]
    if not gaps and col.width >= body_h * 40:
        # dense listings (telephone books) set wide columns with a very
        # narrow gutter: accept a thin but nearly clean channel
        thin = _gaps(colp, max(2, int(body_h * 0.2)), noise=max(1, int(col.height * 0.03)))
        gaps = [g for g in thin if g[0] >= body_h * 20 and col.width - g[1] >= body_h * 20]
    if not gaps:
        return [col]
    # choose the widest gap
    a, b = max(gaps, key=lambda g: g[1] - g[0])
    left = BBox(col.x0, col.y0, col.x0 + a, col.y1)
    right = BBox(col.x0 + b, col.y0, col.x1, col.y1)
    out = []
    for part in (left, right):
        t = _ink_bbox(ink, part)
        if t is not None:
            out.extend(_split_column(ink, t, body_h, depth + 1))
    return out


def _cluster(boxes: list[BBox], gap: float) -> list[BBox]:
    groups = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[BBox] = []
        for b in groups:
            for i, g in enumerate(out):
                if g.gap_to(b) <= gap:
                    out[i] = g.union(b)
                    changed = True
                    break
            else:
                out.append(b)
        groups = out
    return groups


def _cluster_keep_cells(boxes: list[BBox], cells: list[BBox], body_h: float,
                        gap: Optional[float] = None) -> list[BBox]:
    """Group loose display fragments into ads; ruled boxes stay as they are
    (but absorb fragments lying inside them)."""
    cell_set = [b for b in boxes if b in cells]
    loose = [b for b in boxes if b not in cells]
    # loose fragments that overlap a cell belong to that cell
    rest = []
    for b in loose:
        hit = None
        for i, c in enumerate(cell_set):
            if c.overlap_ratio(b) > 0.3 or b.overlap_ratio(c) > 0.3:
                hit = i
                break
        if hit is None:
            rest.append(b)
        else:
            cell_set[hit] = cell_set[hit].union(b)
    return cell_set + _cluster(rest, body_h * 2.5 if gap is None else gap)


def _rule_panels(box: BBox, rules: im.RuleLines, body_h: float, depth: int = 0) -> list[BBox]:
    """Recursively split ``box`` along rules that cross most of it."""
    if depth > 8 or box.width < body_h * 4 or box.height < body_h * 3:
        return [box]
    margin = body_h * 1.5
    for s in sorted(rules.h_segments, key=lambda s: -s.width):
        if box.y0 + margin < s.cy < box.y1 - margin and s.x_overlap(box) >= 0.8 * box.width:
            return (_rule_panels(BBox(box.x0, box.y0, box.x1, s.y0), rules, body_h, depth + 1)
                    + _rule_panels(BBox(box.x0, s.y1, box.x1, box.y1), rules, body_h, depth + 1))
    for s in sorted(rules.v_segments, key=lambda s: -s.height):
        if box.x0 + margin < s.cx < box.x1 - margin and s.y_overlap(box) >= 0.8 * box.height:
            return (_rule_panels(BBox(box.x0, box.y0, s.x0, box.y1), rules, body_h, depth + 1)
                    + _rule_panels(BBox(s.x1, box.y0, box.x1, box.y1), rules, body_h, depth + 1))
    return [box]


def _drop_slivers(regions: list[Region], rules: im.RuleLines, content: BBox, dpi: int) -> None:
    """Remove / clip ad regions that come from the facing page.

    Anything lying wholly outside the page's outermost long vertical rules
    and too narrow to be a side ad is a sliver of the neighbouring leaf;
    ads that merely poke beyond the frame are clipped back to it."""
    long = [s for s in rules.v_segments if s.height >= 0.6 * content.height]
    if not long:
        return
    lx = min(s.x0 for s in long) - 2
    rx = max(s.x1 for s in long) + 2
    for r in list(regions):
        if not r.kind.is_ad:
            continue
        b = r.bbox
        if (b.x1 <= lx + 4 or b.x0 >= rx - 4) and b.width < dpi * 0.55:
            regions.remove(r)
            continue
        inside = b.intersect(BBox(lx, b.y0, rx, b.y1))
        if not inside.is_empty() and inside.area >= 0.7 * b.area and inside.area < b.area:
            r.bbox = inside


def _merge_overlapping(regions: list[Region]) -> None:
    """Merge ad regions that substantially overlap each other."""
    changed = True
    while changed:
        changed = False
        ads = [r for r in regions if r.kind.is_ad]
        for i, a in enumerate(ads):
            for b in ads[i + 1:]:
                inter = a.bbox.intersection_area(b.bbox)
                if inter <= 0:
                    continue
                if inter >= 0.2 * min(a.bbox.area, b.bbox.area):
                    a.bbox = a.bbox.union(b.bbox)
                    regions.remove(b)
                    changed = True
                    break
            if changed:
                break


def _zone(b: BBox, body: BBox, content: BBox, cfg: LayoutConfig) -> RegionKind:
    if (b.x1 <= body.x0 + 2 or b.x0 >= body.x1 - 2) and b.y0 < body.y1 and b.y1 > body.y0:
        return RegionKind.AD_SIDE
    if b.y1 <= body.y0 + (body.height * 0.05):
        return RegionKind.AD_TOP
    if b.y0 >= body.y1 - (body.height * 0.05):
        return RegionKind.AD_BOTTOM
    if b.x1 <= body.x0 + 2 or b.x0 >= body.x1 - 2:
        return RegionKind.AD_SIDE
    # partially overlapping; decide by centre
    if b.cy < body.y0:
        return RegionKind.AD_TOP
    if b.cy > body.y1:
        return RegionKind.AD_BOTTOM
    if b.cx < body.x0 or b.cx > body.x1:
        return RegionKind.AD_SIDE
    return RegionKind.AD_INLINE


def _attach_labels(regions: list[Region], body_h: float) -> None:
    """A thin ad box directly above a bigger ad box of the same width is the
    category caption of that ad (classic Polk advertising pages)."""
    ads = [r for r in regions if r.kind.is_ad]
    has_body = any(r.kind is RegionKind.TEXT and r.column >= 0 for r in regions)
    captions = ads + ([] if has_body else
                      [r for r in regions if r.kind is RegionKind.TEXT])
    for r in captions:
        if r.bbox.height > body_h * 4.5:
            continue
        below = [o for o in ads if o is not r and o.bbox.y0 >= r.bbox.y1 - body_h
                 and o.bbox.y0 - r.bbox.y1 < body_h * 4.5
                 and o.bbox.x_overlap(r.bbox) > 0.7 * min(r.bbox.width, o.bbox.width)
                 and o.bbox.height > r.bbox.height * 1.5]
        if below:
            target = min(below, key=lambda o: o.bbox.y0)
            r.kind = RegionKind.AD_LABEL
            r.meta["label_for"] = target.id
            target.meta["label_region"] = r.id


def _order(regions: list[Region]) -> None:
    rank = {RegionKind.HEADER: 0, RegionKind.AD_TOP: 1, RegionKind.AD_LABEL: 2,
            RegionKind.AD_BOX: 2, RegionKind.AD_SIDE: 3, RegionKind.TEXT: 4,
            RegionKind.AD_INLINE: 5, RegionKind.AD_BOTTOM: 6, RegionKind.FOOTER: 7}
    regions.sort(key=lambda r: (rank.get(r.kind, 9), r.column if r.column >= 0 else 0,
                                round(r.bbox.y0 / 20), r.bbox.x0))
