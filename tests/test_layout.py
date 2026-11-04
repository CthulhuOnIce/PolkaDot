from collections import Counter

import synth

from polkadot.layout import analyze_layout
from polkadot.models import RegionKind


def _layout(dpi=200):
    import cv2
    img = synth.page_array()
    if dpi != 200:
        f = dpi / 200
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return analyze_layout(img, dpi)


def test_synthetic_page_regions():
    lay = _layout()
    kinds = Counter(r.kind for r in lay.regions)
    assert kinds[RegionKind.TEXT] == 3
    assert kinds[RegionKind.AD_TOP] == 1
    assert kinds[RegionKind.AD_BOTTOM] == 1
    assert kinds[RegionKind.AD_SIDE] == 1
    assert kinds[RegionKind.HEADER] == 1
    side = next(r for r in lay.regions if r.kind is RegionKind.AD_SIDE)
    # the whole boxed column ad, not fragments of it
    assert side.bbox.height > 1200 and side.bbox.x1 < 240


def test_columns_in_reading_order():
    lay = _layout()
    cols = [r for r in lay.regions if r.kind is RegionKind.TEXT]
    assert [c.column for c in cols] == [0, 1, 2]
    assert cols[0].bbox.x1 <= cols[1].bbox.x0 <= cols[2].bbox.x0


def test_layout_is_resolution_independent():
    a = _layout(200)
    b = _layout(300)
    assert sorted(r.kind.value for r in a.regions) == sorted(r.kind.value for r in b.regions)
    for ra in a.regions:
        rb = min(b.regions, key=lambda r: abs(r.bbox.cx - ra.bbox.cx * 1.5) + abs(r.bbox.cy - ra.bbox.cy * 1.5))
        assert rb.kind == ra.kind
        assert rb.bbox.iou(ra.bbox.scale(1.5)) > 0.8
