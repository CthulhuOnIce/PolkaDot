"""Low level raster operations: binarisation, page-area detection, ruling
lines, enclosed boxes and connected-component statistics.

Everything here works on numpy arrays.  ``ink`` masks are ``uint8`` arrays
with 255 for dark (printed) pixels and 0 for paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .models import BBox


# --------------------------------------------------------------------------
# Binarisation
# --------------------------------------------------------------------------


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def normalize_background(gray: np.ndarray, dpi: int) -> np.ndarray:
    """Flatten uneven illumination / tinted paper (e.g. yellow ad pages).

    The background is estimated with a grey-level closing (which erases dark
    print smaller than the kernel) and the page is divided by it, so paper
    becomes ~255 regardless of colour or shadow.
    """
    k = max(15, int(dpi * 0.12)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg = cv2.medianBlur(bg, 5)
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def binarize(gray: np.ndarray, dpi: int, level: float = 0.85) -> np.ndarray:
    """Return an ink mask (255 = ink).

    After background normalisation paper sits at ~255, so a fixed relative
    threshold is far more stable than Otsu, which is easily dragged around
    by dark book covers or big black display panels in the scan.
    """
    norm = normalize_background(gray, dpi)
    _, ink = cv2.threshold(norm, int(255 * level), 255, cv2.THRESH_BINARY_INV)
    # Very dark absolute pixels (large solid black areas that the closing
    # could not see past) are ink too.
    ink[gray < 60] = 255
    return ink


# --------------------------------------------------------------------------
# Skew
# --------------------------------------------------------------------------


def estimate_skew(ink: np.ndarray, max_angle: float = 3.0) -> float:
    """Estimate page skew in degrees (positive = content rotated CCW).

    Classic projection-profile search: the angle that makes horizontal ink
    profiles the "spikiest" (highest variance) wins.
    """
    h, w = ink.shape
    scale = 1000.0 / max(h, w)
    small = cv2.resize(ink, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    center = (sw / 2, sh / 2)

    def score(a: float) -> float:
        m = cv2.getRotationMatrix2D(center, a, 1.0)
        r = cv2.warpAffine(small, m, (sw, sh), flags=cv2.INTER_NEAREST)
        prof = r.sum(axis=1, dtype=np.float64)
        return float(np.var(prof))

    best = max(np.arange(-max_angle, max_angle + 1e-6, 0.25), key=score)
    fine = max(np.arange(best - 0.25, best + 0.25 + 1e-6, 0.05), key=score)
    return float(round(fine, 2))


def rotation_matrix(shape: tuple[int, ...], angle: float) -> np.ndarray:
    h, w = shape[:2]
    return cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``img`` by ``angle`` degrees about its centre (keeps size)."""
    if abs(angle) < 0.05:
        return img
    m = rotation_matrix(img.shape, angle)
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderValue=border)


def transform_bbox(b: BBox, m: np.ndarray) -> BBox:
    pts = np.array([[b.x0, b.y0, 1], [b.x1, b.y0, 1], [b.x0, b.y1, 1], [b.x1, b.y1, 1]])
    t = pts @ m.T
    return BBox(float(t[:, 0].min()), float(t[:, 1].min()),
                float(t[:, 0].max()), float(t[:, 1].max()))


# --------------------------------------------------------------------------
# Page / content area
# --------------------------------------------------------------------------


def paper_mask(gray: np.ndarray, dpi: int) -> np.ndarray:
    """Rough mask of the (light) paper as opposed to dark covers / table."""
    small = cv2.GaussianBlur(gray, (0, 0), dpi / 60)
    # Paper is "not as dark as a book cover".  (Otsu is no good here: on a
    # scan with a white scanner bed it splits *paper* from the bed, and
    # shaded paper near the binding is lost.)
    bright = float(np.percentile(small, 95))
    t, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark_class = small[small <= t]
    if dark_class.size and float(np.median(dark_class)) < bright * 0.5:
        # a genuinely dark surround (book cover, table): Otsu separates it;
        # bias a little toward paper so tinted (yellow) pages are kept
        thresh = t * 0.85
    else:
        thresh = bright * 0.55
    mask = (small > thresh).astype(np.uint8) * 255
    k = max(3, int(dpi * 0.05)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return mask


def _runs(profile: np.ndarray, thresh: float, max_gap: int) -> list[tuple[int, int, float]]:
    """Runs of indices where profile > thresh, merging gaps <= max_gap.

    Returns (start, end_exclusive, mass) sorted by mass desc.
    """
    on = profile > thresh
    runs: list[list[float]] = []
    i, n = 0, len(profile)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if runs and i - runs[-1][1] <= max_gap:
                runs[-1][1] = j
            else:
                runs.append([i, j, 0.0])
            i = j
        else:
            i += 1
    for r in runs:
        r[2] = float(profile[int(r[0]):int(r[1])].sum())
    return sorted([(int(a), int(b), m) for a, b, m in runs], key=lambda r: -r[2])


def remove_border_components(ink: np.ndarray, margin: int = 2) -> np.ndarray:
    """Drop ink blobs touching the image border (scanner bed, book covers)."""
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return ink
    x, y, bw, bh = (stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP],
                    stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    touch = (x <= margin) | (y <= margin) | (x + bw >= w - margin) | (y + bh >= h - margin)
    touch[0] = False
    keep = np.ones(n, dtype=bool)
    keep[touch] = False
    keep[0] = False
    out = np.where(keep[labels], 255, 0).astype(np.uint8)
    return out


def _long_vrules(ink: np.ndarray, y0: int, y1: int, dpi: int) -> list[tuple[int, int]]:
    """x-extents of vertical rules spanning most of rows y0..y1."""
    band = ink[y0:y1]
    if band.size == 0:
        return []
    k = max(10, int((y1 - y0) * 0.15))
    m = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    prof = (m > 0).sum(axis=0)
    on = prof >= 0.6 * (y1 - y0)
    out = []
    i = 0
    n = len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if j - i <= dpi * 0.08:          # thin: a rule, not a panel
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _boxed(ink: np.ndarray, a: int, b: int, y0: int, y1: int) -> bool:
    """Are two vertical rules at x=a and x=b joined by a horizontal rule
    (the top or bottom of a box)?  Two rules either side of the gutter -
    our page's frame and the facing page's - never are."""
    if b - a < 4:
        return False
    band = ink[max(0, y0 - 40):y1 + 40, a:b] > 0
    if band.size == 0:
        return False
    return bool((band.mean(axis=1) >= 0.85).any())


def _cut_border_sliver(profile: np.ndarray, lo: int, hi: int, size: int, dpi: int,
                       touch: float = 0.02, reach: float = 0.16) -> tuple[int, int]:
    """If the span [lo, hi) starts/ends at the image border, move that end
    inward to the emptiest position within ``reach`` of the border."""
    k = max(3, int(dpi * 0.03))
    sm = np.convolve(profile, np.ones(k) / k, mode="same")
    span = sm[lo:hi]
    if span.size == 0:
        return lo, hi
    level = float(np.median(span[span > 0])) if (span > 0).any() else 0.0
    if lo <= size * touch:
        a, b = lo + int(dpi * 0.05), min(hi, int(size * reach))
        if b > a:
            i = a + int(np.argmin(sm[a:b]))
            if sm[i] <= level * 0.12:
                lo = i
    if hi >= size * (1 - touch):
        a, b = max(lo, int(size * (1 - reach))), hi - int(dpi * 0.05)
        if b > a:
            i = a + int(np.argmin(sm[a:b]))
            if sm[i] <= level * 0.12:
                hi = i
    return lo, hi


def _absorb(colp: np.ndarray, start: int, direction: int, dpi: int, w: int, dark_between) -> int:
    """Extend an edge over nearby glyph columns (unboxed side matter)."""
    thresh = max(1.0, colp.max() * 0.01)
    gap_max = int(dpi * 0.32)
    edge = start
    run_ends = [start]          # edge reached at the end of each block of print
    x = start
    gap = 0
    while 0 < x < w - 1:
        x += direction
        if colp[x] > thresh:
            if gap > 0:
                if dark_between(min(edge, x), max(edge, x)):
                    break
                run_ends.append(edge)
            edge = x
            gap = 0
        else:
            gap += 1
            if gap > gap_max:
                break
    run_ends.append(edge)

    def at_border(e: int) -> bool:
        return (direction < 0 and e <= w * 0.06) or (direction > 0 and e >= w * 0.94)

    # if we ran into the image border, the last block was a sliver of the
    # facing page / book edge: stop at the block before it
    for e in reversed(run_ends):
        if not at_border(e):
            return e
    return start


def find_content_bbox(gray: np.ndarray, ink: np.ndarray, dpi: int) -> BBox:
    """Locate the printed area of the page being scanned.

    Scans of bound directories routinely include the book cover, the gutter,
    the stacked edges of other leaves, thumb-index tabs and slivers of the
    neighbouring page.  We look only at *glyph-like* connected components
    (letters of any reasonable size — not long lines or huge blotches) that
    sit on paper and do not touch the image border, and take the dominant
    run of glyph-bearing columns / rows, allowing for normal gutters.
    """
    h, w = ink.shape
    paper = paper_mask(gray, dpi)
    k = max(3, int(dpi * 0.04)) | 1
    paper = cv2.erode(paper, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    clean = remove_border_components(cv2.bitwise_and(ink, paper))

    n, _, st, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    st = st[1:]
    if len(st) == 0:
        return BBox(0, 0, w, h)
    cw_, ch_ = st[:, 2].astype(float), st[:, 3].astype(float)
    glyph = ((ch_ >= dpi * 2.5 / 72) & (ch_ <= dpi * 60 / 72) & (cw_ <= dpi * 60 / 72)
             & (ch_ / np.maximum(cw_, 1) < 6) & (cw_ / np.maximum(ch_, 1) < 40)   # fused words are wide
             & (st[:, 4] >= 0.12 * cw_ * ch_))
    st = st[glyph]
    if len(st) < 5:
        return BBox(0, 0, w, h)
    glyphs = np.zeros((h, w), np.uint8)
    for x, y, bw, bh, _a in st:
        glyphs[y:y + bh, x:x + bw] = 1

    # column brightness, used to refuse merging across the gutter shadow
    colmean = None
    if gray.ndim == 2:
        # median, not mean: print (even big display type) leaves most rows
        # of a column at paper level, the binding shadow darkens all of them
        colmean = np.median(gray[int(h * 0.2):int(h * 0.8):3], axis=0).astype(np.float32)

    def dark_between(a_: int, b_: int) -> bool:
        """Is there a gutter shadow (not just a printed rule) in [a_, b_)?

        A printed rule is a sharp, narrow dip; the shadow of the binding is
        dark *and* has a soft shoulder on at least one side."""
        if colmean is None or b_ <= a_:
            return False
        seg = colmean[a_:b_]
        i = int(np.argmin(seg)) + a_
        lo_, hi_ = max(0, i - int(dpi * 0.3)), min(len(colmean), i + int(dpi * 0.3))
        local = float(np.percentile(colmean[lo_:hi_], 80))
        if colmean[i] > local * 0.72:
            return False
        # a shadow is dark across a band; a (even double) printed rule is
        # only dark on a few columns
        half = max(4, int(dpi * 0.05))
        win = colmean[max(0, i - half):min(len(colmean), i + half + 1)]
        return float((win < local * 0.85).mean()) >= 0.6

    def main_span(profile: np.ndarray, thresh: float, size: int,
                  reach: float = 0.4, check_dark: bool = False,
                  edge_frac: float = 0.06) -> Optional[tuple[int, int]]:
        """Dominant run, then greedily absorb neighbouring runs that are close
        and do not hug the image border (slivers of the facing page)."""
        runs = _runs(profile, thresh, max_gap=int(dpi * 0.12))
        if not runs:
            return None
        lo, hi, _ = runs[0]
        edge = size * edge_frac
        others = sorted(runs[1:], key=lambda r: -r[2])
        grew = True
        while grew:
            grew = False
            for r in others:
                a_, b_, mass = r
                if a_ >= lo and b_ <= hi:
                    continue
                gap = (a_ - hi) if a_ >= hi else (lo - b_)
                if gap > dpi * reach or a_ <= edge or b_ >= size - edge:
                    continue
                if check_dark and (dark_between(hi, a_) if a_ >= hi else dark_between(b_, lo)):
                    continue
                lo, hi = min(lo, a_), max(hi, b_)
                grew = True
        return lo, hi

    # 1. core: dominant run of glyph columns (tight gaps only)
    colp = glyphs.sum(axis=0).astype(float)
    runs = _runs(colp, max(1.0, 0.01 * h), max_gap=int(dpi * 0.12))
    if not runs:
        return BBox(0, 0, w, h)
    x0, x1, _ = runs[0]
    rowp = glyphs[:, x0:x1].sum(axis=1).astype(float)
    # facing pages sit left/right, so vertically we can be far more generous
    span = main_span(rowp, max(1.0, 0.01 * (x1 - x0)), h, reach=1.5, edge_frac=0.01)
    if span is None:
        return BBox(x0, 0, x1, h)
    y0, y1 = span
    colp2 = glyphs[y0:y1].sum(axis=0).astype(float)
    runs2 = _runs(colp2, max(1.0, 0.01 * (y1 - y0)), max_gap=int(dpi * 0.12))
    if runs2:
        x0, x1, _ = runs2[0]

    # 1b. print running right up to the image border is the facing page
    #     (or the book edge) glued onto the core by a tight gutter: cut it
    #     off at the deepest whitespace valley near that border
    x0, x1 = _cut_border_sliver(glyphs[y0:y1].sum(axis=0).astype(float), x0, x1, w, dpi)
    y0, y1 = _cut_border_sliver(glyphs[:, x0:x1].sum(axis=1).astype(float), y0, y1, h, dpi)
    colp2 = glyphs[y0:y1].sum(axis=0).astype(float)

    # 2. walk outwards over the page frame: long vertical rules right next
    #    to the core (a column rule / the inner border of a side ad) and,
    #    one step further, the outer border of that side-ad box.
    # lines hugging the scan's edge are the edge of the leaf / the binding,
    # not printed rules
    vr = [r for r in _long_vrules(ink, y0, y1, dpi) if w * 0.06 < r[0] and r[1] < w * 0.94]
    near, far = dpi * 0.25, dpi * 1.25
    core_mass = float(colp2[x0:x1].sum()) or 1.0
    grun = _runs(colp2, max(1.0, 0.01 * (y1 - y0)), max_gap=int(dpi * 0.12))

    def column_beyond(edge: int, direction: int) -> Optional[tuple[int, int]]:
        """A full column of print just past a column rule (dense listings
        such as telephone books put rules between their columns)."""
        best = None
        for a_, b_, mass in grun:
            gap = (a_ - edge) if direction > 0 else (edge - b_)
            if not (0 <= gap <= dpi * 0.15) or mass < 0.12 * core_mass:
                continue
            if (direction < 0 and a_ <= w * 0.02) or (direction > 0 and b_ >= w * 0.98):
                continue           # runs off the image: the facing page
            if dark_between(min(edge, a_ if direction > 0 else b_), max(edge, a_ if direction > 0 else b_)):
                continue
            best = (a_, b_)
        return best

    gx0, gx1 = x0, x1
    for _ in range(8):                                   # walk left
        left_inner = [r for r in vr if gx0 - near <= r[1] <= gx0 + dpi * 0.1]
        if not left_inner:
            gx0 = _absorb(colp2, gx0, -1, dpi, w, dark_between)
            break
        edge = max(r[0] for r in left_inner)        # the rule nearest the print
        col = column_beyond(edge, -1)
        if col is not None:
            gx0 = col[0]
            continue
        gx0 = min(gx0, edge)
        outer = [r for r in vr if edge - far <= r[1] < edge - dpi * 0.3
                 and not dark_between(r[1], edge) and r[0] > w * 0.02
                 and colp2[r[1]:edge].sum() >= 0.015 * colp2.sum()    # a box with print in it
                 and _boxed(ink, r[0], edge, y0, y1)]
        if outer:
            gx0 = min(r[0] for r in outer)
        break
    for _ in range(8):                                   # walk right
        right_inner = [r for r in vr if gx1 - dpi * 0.1 <= r[0] <= gx1 + near]
        if not right_inner:
            gx1 = _absorb(colp2, gx1, +1, dpi, w, dark_between)
            break
        edge = min(r[1] for r in right_inner)       # the rule nearest the print
        col = column_beyond(edge, +1)
        if col is not None:
            gx1 = col[1]
            continue
        gx1 = max(gx1, edge)
        outer = [r for r in vr if edge + dpi * 0.3 < r[0] <= edge + far
                 and not dark_between(edge, r[0]) and r[1] < w * 0.98
                 and colp2[edge:r[0]].sum() >= 0.015 * colp2.sum()
                 and _boxed(ink, edge, r[1], y0, y1)]
        if outer:
            gx1 = max(r[1] for r in outer)
        break
    x0, x1 = gx0, gx1
    # grow a little so the rules framing ads at the edges are included
    pad = int(dpi * 0.07)
    return BBox(max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad))


# --------------------------------------------------------------------------
# Ruling lines and boxes
# --------------------------------------------------------------------------


@dataclass
class RuleLines:
    horizontal: np.ndarray      # mask of horizontal rules
    vertical: np.ndarray        # mask of vertical rules
    h_segments: list[BBox]
    v_segments: list[BBox]

    @property
    def combined(self) -> np.ndarray:
        return cv2.bitwise_or(self.horizontal, self.vertical)


def _segments(mask: np.ndarray, horizontal: Optional[bool] = None,
              max_thick: float = 1e9, min_ratio: float = 0.0) -> list[BBox]:
    """Bounding boxes of mask components, optionally filtered as rules.

    A rule's thickness is measured as pixel area / length, which is robust to
    a slight residual skew (where the bbox of a long thin line grows tall).
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if horizontal is not None:
            length = w if horizontal else h
            thick = a / max(1, length)
            if thick > max_thick or length < thick * min_ratio:
                continue
        out.append(BBox(float(x), float(y), float(x + w), float(y + h)))
    return out


def _continuity(ink: np.ndarray, seg: BBox, horizontal: bool) -> float:
    """Share of a candidate rule's length that is actually inked.

    Printed rules are continuous; dot/dash leaders ("Gary------TU 2-1341")
    and the aligned edges of text stacked line upon line only look like
    rules after gap bridging, and are inked for a much smaller share."""
    x0, y0, x1, y1 = seg.to_int()
    sub = ink[max(0, y0):y1 + 1, max(0, x0):x1 + 1] > 0
    if sub.size == 0:
        return 0.0
    along = sub.any(axis=0) if horizontal else sub.any(axis=1)
    return float(along.mean())


def find_rule_lines(ink: np.ndarray, content: BBox, dpi: int) -> RuleLines:
    """Detect long straight horizontal / vertical rules inside ``content``."""
    cw, ch = content.width, content.height
    hk = max(int(dpi * 0.6), int(cw * 0.10))
    vk = max(int(dpi * 0.6), int(ch * 0.06))
    # bridge 1px breaks in scanned rules first
    bridged_h = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    bridged_v = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    hmask = cv2.morphologyEx(bridged_h, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    vmask = cv2.morphologyEx(bridged_v, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    # rules must be thin: drop thick blobs (solid black panels)
    # Reverse-video panels (white type on black) produce lots of fake rules
    # between the letters; mask out solid areas before looking for rules.
    sk = max(5, int(dpi * 0.05)) | 1
    solid = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (sk, sk)))
    solid = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_RECT, (sk * 2, sk * 2)))
    hmask[solid > 0] = 0
    vmask[solid > 0] = 0
    max_thick = max(4.0, dpi * 0.06)
    hseg = [g for g in _segments(hmask, True, max_thick, 15) if _continuity(ink, g, True) >= 0.9]
    vseg = [g for g in _segments(vmask, False, max_thick, 15) if _continuity(ink, g, False) >= 0.9]
    hmask2 = np.zeros_like(ink)
    vmask2 = np.zeros_like(ink)
    for s in hseg:
        x0, y0, x1, y1 = s.to_int()
        hmask2[y0:y1, x0:x1] = hmask[y0:y1, x0:x1]
    for s in vseg:
        x0, y0, x1, y1 = s.to_int()
        vmask2[y0:y1, x0:x1] = vmask[y0:y1, x0:x1]
    # restrict to content area
    keep = np.zeros_like(ink)
    x0, y0, x1, y1 = content.to_int()
    keep[y0:y1, x0:x1] = 255
    hmask2 = cv2.bitwise_and(hmask2, keep)
    vmask2 = cv2.bitwise_and(vmask2, keep)
    return RuleLines(hmask2, vmask2,
                     merge_collinear(_segments(hmask2), True, dpi),
                     merge_collinear(_segments(vmask2), False, dpi))


def merge_collinear(segs: list[BBox], horizontal: bool, dpi: int,
                    max_gap: float = 0.35, tol: float = 0.02) -> list[BBox]:
    """Join broken pieces of the same rule (scans often interrupt rules)."""
    gap, t = dpi * max_gap, dpi * tol
    key = (lambda s: (s.cy, s.x0)) if horizontal else (lambda s: (s.cx, s.y0))
    segs = sorted(segs, key=key)
    out: list[BBox] = []
    for s in segs:
        for i, o in enumerate(out):
            if horizontal:
                ok = abs(o.cy - s.cy) <= t and s.x0 - o.x1 <= gap and o.x0 - s.x1 <= gap
            else:
                ok = abs(o.cx - s.cx) <= t and s.y0 - o.y1 <= gap and o.y0 - s.y1 <= gap
            if ok:
                out[i] = o.union(s)
                break
        else:
            out.append(s)
    return out


def find_solid_panels(ink: np.ndarray, content: BBox, dpi: int,
                      min_side: float = 0.25) -> list[BBox]:
    """Large dark panels (reverse-video display ads, cuts / engravings)."""
    k = max(5, int(dpi * 0.05)) | 1
    solid = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    # join the pieces of a panel broken up by its white lettering
    j = max(3, int(dpi * 0.03)) | 1
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (j, j)))
    out = []
    for b in _segments(solid):
        if not content.contains(b, tol=dpi * 0.05):
            continue
        if b.width >= dpi * min_side and b.height >= dpi * min_side * 0.6:
            out.append(b)
    return out


def expand_to_rules(content: BBox, rules: "RuleLines", dpi: int,
                    reach: float = 0.15, reach_x: float = 0.08) -> BBox:
    """Grow ``content`` so that ruling lines framing the printed area just
    outside it (ad box borders, page frames) are included."""
    r = dpi * reach
    rx = dpi * reach_x          # sideways the facing page is close: be strict
    out = content
    for s in rules.h_segments:
        if s.x_overlap(content) >= 0.5 * s.width and content.y0 - r <= s.cy <= content.y1 + r:
            out = out.union(BBox(out.x0, s.y0, out.x1, s.y1))
    for s in rules.v_segments:
        if s.y_overlap(content) >= 0.5 * s.height and content.x0 - rx <= s.cx <= content.x1 + rx:
            out = out.union(BBox(s.x0, out.y0, s.x1, out.y1))
    return out


def trim_outside_frame(content: BBox, rules: "RuleLines", ink: np.ndarray, dpi: int,
                       max_frac: float = 0.25, min_gap: float = 0.2) -> BBox:
    """Drop a band of print lying beyond the outermost long vertical rule
    when it is separated from that rule by a wide blank margin.

    Directory pages are framed by rules (ad boxes, column rules).  Print
    beyond the outermost one *and* well separated from it is the facing
    page (margins + gutter), whereas a genuine unboxed side ad hugs it."""
    long = [s for s in rules.v_segments if s.height >= 0.6 * content.height]
    if not long:
        return content
    left = min(long, key=lambda s: s.x0)
    right = max(long, key=lambda s: s.x1)
    y0, y1 = int(content.y0), int(content.y1)
    x0, x1 = content.x0, content.x1
    thresh = max(2, int(content.height * 0.003))

    def inked(a_: int, b_: int) -> np.ndarray:
        if b_ <= a_:
            return np.zeros(0, bool)
        return (ink[y0:y1, a_:b_] > 0).sum(axis=0) > thresh

    strip = int(left.x0) - int(content.x0)
    if dpi * 0.05 < strip < max_frac * content.width:
        cols = inked(int(content.x0), int(left.x0) - 3)
        idx = np.nonzero(cols)[0]
        if len(idx) == 0 or (len(cols) - 1 - idx.max()) >= dpi * min_gap:
            x0 = left.x0 - 2
    strip = int(content.x1) - int(right.x1)
    if dpi * 0.05 < strip < max_frac * content.width:
        cols = inked(int(right.x1) + 3, int(content.x1))
        idx = np.nonzero(cols)[0]
        if len(idx) == 0 or idx.min() >= dpi * min_gap:
            x1 = right.x1 + 2
    return BBox(x0, content.y0, x1, content.y1)


@dataclass
class Cell:
    """An area fully enclosed by ruling lines (an ad box, a framed column...)."""
    bbox: BBox
    mask: np.ndarray            # full-page boolean mask of the enclosed pixels
    fill: float                 # enclosed pixels / bbox area


def find_cells(rules: RuleLines, content: BBox, dpi: int) -> list[Cell]:
    """Find regions enclosed on all sides by ruling lines.

    We look at connected components of *non-rule* space.  The component that
    touches the content border is the open page; every other sufficiently
    large, mostly rectangular component is a box.
    """
    lines = rules.combined
    # close small gaps at rule junctions
    k = max(3, int(dpi * 0.012)) | 1
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    H, W = lines.shape
    m = 3
    x0, y0, x1, y1 = content.to_int()
    x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(W, x1 + m), min(H, y1 + m)
    sub = lines[y0:y1, x0:x1]
    space = cv2.bitwise_not(sub)
    # make the content border "open" (part of the outside)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(space, connectivity=4)
    sh, sw = space.shape
    min_side = dpi * 0.25
    cells: list[Cell] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x <= 1 or y <= 1 or x + w >= sw - 1 or y + h >= sh - 1:
            continue
        if w < min_side or h < min_side:
            continue
        fill = area / float(w * h)
        if fill < 0.6:
            continue
        m = np.zeros(lines.shape, dtype=bool)
        m[y0:y1, x0:x1] = labels == i
        cells.append(Cell(BBox(x + x0, y + y0, x + x0 + w, y + y0 + h), m, fill))
    return cells


# --------------------------------------------------------------------------
# Connected components & text statistics
# --------------------------------------------------------------------------


@dataclass
class Components:
    boxes: np.ndarray           # (n, 4) x, y, w, h
    areas: np.ndarray           # (n,)

    def __len__(self) -> int:
        return len(self.areas)

    def select(self, idx: np.ndarray) -> "Components":
        return Components(self.boxes[idx], self.areas[idx])

    def inside(self, box: BBox) -> "Components":
        if len(self) == 0:
            return self
        cx = self.boxes[:, 0] + self.boxes[:, 2] / 2
        cy = self.boxes[:, 1] + self.boxes[:, 3] / 2
        sel = (cx >= box.x0) & (cx <= box.x1) & (cy >= box.y0) & (cy <= box.y1)
        return self.select(sel)


def components(ink: np.ndarray, min_area: int = 4) -> Components:
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return Components(np.zeros((0, 4), int), np.zeros((0,), int))
    st = stats[1:]
    keep = st[:, cv2.CC_STAT_AREA] >= min_area
    st = st[keep]
    return Components(st[:, :4].astype(int), st[:, 4].astype(int))


def estimate_text_height(comps: Components, dpi: int) -> float:
    """Most common glyph height (≈ x-height of body type) in pixels."""
    lo, hi = dpi * 3 / 72.0, dpi * 14 / 72.0
    if len(comps) == 0:
        return dpi * 6 / 72.0
    h = comps.boxes[:, 3]
    w = comps.boxes[:, 2]
    sel = (h >= lo) & (h <= hi) & (w <= h * 3) & (w >= 2)
    hs = h[sel]
    if len(hs) < 20:
        return dpi * 6 / 72.0
    hist = np.bincount(hs)
    # smooth histogram a little, take the mode
    kernel = np.array([1, 2, 3, 2, 1], float)
    sm = np.convolve(hist, kernel, mode="same")
    return float(np.argmax(sm))


@dataclass
class BlockStats:
    n: int
    body_frac: float      # fraction of glyph-like comps sized like body text
    large_frac: float     # fraction of ink in comps much taller than body text
    tall_frac: float      # fraction of comps taller than wide by > 2.5 (rotated text hint)
    median_h: float
    ink_density: float

    def looks_like_body_text(self) -> bool:
        return self.n >= 8 and self.body_frac >= 0.6 and self.large_frac < 0.3


def block_stats(comps: Components, box: BBox, body_h: float, ink: np.ndarray) -> BlockStats:
    c = comps.inside(box)
    x0, y0, x1, y1 = box.to_int()
    area = max(1, (x1 - x0) * (y1 - y0))
    density = float((ink[y0:y1, x0:x1] > 0).sum()) / area
    if len(c) == 0:
        return BlockStats(0, 0.0, 0.0, 0.0, 0.0, density)
    h = c.boxes[:, 3].astype(float)
    w = c.boxes[:, 2].astype(float)
    glyph = h >= body_h * 0.45
    n = int(glyph.sum())
    if n == 0:
        return BlockStats(0, 0.0, 0.0, 0.0, 0.0, density)
    hg, wg, ag = h[glyph], w[glyph], c.areas[glyph].astype(float)
    body = (hg <= body_h * 2.3) & (wg <= body_h * 4.0)
    large = (hg > body_h * 2.8) | (wg > body_h * 8)
    tall = (hg > wg * 2.5) & (hg > body_h * 1.2)
    return BlockStats(
        n=n,
        body_frac=float(body.mean()),
        large_frac=float(ag[large].sum() / max(1.0, ag.sum())),
        tall_frac=float(tall.mean()),
        median_h=float(np.median(hg)),
        ink_density=density,
    )


def crop(img: np.ndarray, box: BBox) -> np.ndarray:
    x0, y0, x1, y1 = box.to_int()
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    return img[y0:y1, x0:x1]


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate by a multiple of 90° counter-clockwise."""
    angle %= 360
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError("only multiples of 90 supported")


def unrotate_bbox(b: BBox, angle: int, orig_w: int, orig_h: int) -> BBox:
    """Map a bbox found in an image rotated by ``angle`` (CCW) back."""
    angle %= 360
    if angle == 0:
        return b
    if angle == 90:
        # rotated (x', y') = (y, W - x)  =>  x = W - y', y = x'
        return BBox(orig_w - b.y1, b.x0, orig_w - b.y0, b.x1)
    if angle == 270:
        # rotated (x', y') = (H - y, x)  =>  x = y', y = H - x'
        return BBox(b.y0, orig_h - b.x1, b.y1, orig_h - b.x0)
    if angle == 180:
        return BBox(orig_w - b.x1, orig_h - b.y1, orig_w - b.x0, orig_h - b.y0)
    raise ValueError("only multiples of 90 supported")
