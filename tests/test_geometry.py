import numpy as np
import pytest

from polkadot import imaging as im
from polkadot.models import BBox


def test_bbox_ops():
    a, b = BBox(0, 0, 10, 10), BBox(5, 5, 15, 15)
    assert a.intersection_area(b) == 25
    assert a.union(b) == BBox(0, 0, 15, 15)
    assert abs(a.iou(b) - 25 / 175) < 1e-9
    assert a.gap_to(BBox(20, 0, 30, 10)) == 10
    assert a.pad(2, limit=BBox(0, 0, 11, 11)) == BBox(0, 0, 11, 11)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_unrotate_bbox_roundtrip(angle):
    h, w = 40, 70
    img = np.zeros((h, w), np.uint8)
    img[10:20, 30:50] = 255                       # a marked box
    r = im.rotate_image(img, angle)
    ys, xs = np.nonzero(r)
    found = BBox(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    back = im.unrotate_bbox(found, angle, w, h)
    assert back.to_int() == (30, 10, 50, 20)


def test_skew_estimate_recovers_rotation():
    img = np.zeros((600, 500), np.uint8)
    for y in range(40, 560, 18):                 # fake text lines
        img[y:y + 6, 40:460] = 255
    rot = im.deskew(img, 1.5)
    est = im.estimate_skew(rot)
    assert abs(est + 1.5) < 0.3                  # undo = opposite angle


def test_merge_collinear_rules():
    segs = [BBox(100, 10, 103, 200), BBox(100, 230, 103, 400), BBox(300, 10, 303, 400)]
    out = im.merge_collinear(segs, horizontal=False, dpi=200)
    assert len(out) == 2
    assert any(s.y0 == 10 and s.y1 == 400 and s.x0 == 100 for s in out)


def test_trim_outside_frame_drops_facing_page_strip():
    """Print beyond a double frame rule, after a blank margin, is the facing page."""
    import cv2
    dpi = 200
    ink = np.zeros((1200, 1000), np.uint8)
    ink[100:1100, 100:104] = 255                 # left frame rule
    ink[100:1100, 800:804] = 255                 # right frame rule ...
    ink[100:1100, 807:810] = 255                 # ... doubled
    for y in range(150, 1050, 30):               # text inside the frame
        ink[y:y + 10, 130:770:14] = 255
    for y in range(150, 1050, 30):               # facing page text beyond a gap
        ink[y:y + 10, 880:960:14] = 255
    ink[100:1100, 975:978] = 255                 # facing leaf edge looks like a rule too
    content = BBox(95, 95, 965, 1105)
    rules = im.find_rule_lines(ink, BBox(0, 0, 1000, 1200), dpi)
    out = im.trim_outside_frame(content, rules, ink, dpi)
    assert 805 <= out.x1 <= 830
    assert out.x0 == content.x0                   # nothing to trim on the left
