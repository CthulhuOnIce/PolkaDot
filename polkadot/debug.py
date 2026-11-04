"""Visual debugging helpers: draw detected regions on a page image."""

from __future__ import annotations

import cv2
import numpy as np

from .models import RegionKind

COLORS = {
    RegionKind.TEXT: (40, 160, 40),
    RegionKind.HEADER: (120, 120, 120),
    RegionKind.FOOTER: (120, 120, 120),
    RegionKind.AD_TOP: (220, 40, 40),
    RegionKind.AD_BOTTOM: (220, 120, 0),
    RegionKind.AD_SIDE: (40, 60, 220),
    RegionKind.AD_INLINE: (200, 0, 200),
    RegionKind.AD_BOX: (0, 170, 200),
    RegionKind.AD_LABEL: (150, 100, 40),
}


def draw_regions(img: np.ndarray, regions, entries=None, content=None,
                 thickness: int = 3) -> np.ndarray:
    """Return an RGB copy of ``img`` with regions (and entries) outlined."""
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    overlay = vis.copy()
    for r in regions:
        c = COLORS.get(r.kind, (0, 0, 0))
        x0, y0, x1, y1 = r.bbox.to_int()
        if r.kind.is_ad:
            cv2.rectangle(overlay, (x0, y0), (x1, y1), c, -1)
    vis = cv2.addWeighted(overlay, 0.18, vis, 0.82, 0)
    if entries:
        for e in entries:
            x0, y0, x1, y1 = e.bbox.to_int()
            col = (230, 60, 60) if e.kind.value in ("business", "proprietor") else (160, 160, 255)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 1)
    for r in regions:
        c = COLORS.get(r.kind, (0, 0, 0))
        x0, y0, x1, y1 = r.bbox.to_int()
        cv2.rectangle(vis, (x0, y0), (x1, y1), c, thickness)
        cv2.putText(vis, r.kind.value, (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, c, 2, cv2.LINE_AA)
    if content is not None:
        x0, y0, x1, y1 = content.to_int()
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 0), 1)
    return vis
