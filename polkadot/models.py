"""Core data structures shared across the pipeline.

All geometry is expressed in *pixel* coordinates of the page raster that the
pipeline rendered (see ``PageImage.dpi``).  Anything exported also carries the
equivalent PDF-point coordinates so results can be mapped back onto the
original document.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable, Optional


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    # -- basic properties ---------------------------------------------------
    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    # -- set operations -----------------------------------------------------
    def union(self, other: "BBox") -> "BBox":
        return BBox(min(self.x0, other.x0), min(self.y0, other.y0),
                    max(self.x1, other.x1), max(self.y1, other.y1))

    def intersect(self, other: "BBox") -> "BBox":
        return BBox(max(self.x0, other.x0), max(self.y0, other.y0),
                    min(self.x1, other.x1), min(self.y1, other.y1))

    def intersection_area(self, other: "BBox") -> float:
        i = self.intersect(other)
        return 0.0 if i.is_empty() else i.area

    def iou(self, other: "BBox") -> float:
        inter = self.intersection_area(other)
        if inter <= 0:
            return 0.0
        return inter / (self.area + other.area - inter)

    def overlap_ratio(self, other: "BBox") -> float:
        """Fraction of *self* covered by *other*."""
        if self.area <= 0:
            return 0.0
        return self.intersection_area(other) / self.area

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def contains(self, other: "BBox", tol: float = 0.0) -> bool:
        return (other.x0 >= self.x0 - tol and other.y0 >= self.y0 - tol
                and other.x1 <= self.x1 + tol and other.y1 <= self.y1 + tol)

    def x_overlap(self, other: "BBox") -> float:
        return max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))

    def y_overlap(self, other: "BBox") -> float:
        return max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))

    def gap_to(self, other: "BBox") -> float:
        """Shortest axis-aligned distance between two boxes (0 if touching)."""
        dx = max(0.0, max(self.x0, other.x0) - min(self.x1, other.x1))
        dy = max(0.0, max(self.y0, other.y0) - min(self.y1, other.y1))
        return (dx * dx + dy * dy) ** 0.5

    # -- transforms ---------------------------------------------------------
    def pad(self, px: float, py: Optional[float] = None,
            limit: Optional["BBox"] = None) -> "BBox":
        py = px if py is None else py
        b = BBox(self.x0 - px, self.y0 - py, self.x1 + px, self.y1 + py)
        return b.intersect(limit) if limit is not None else b

    def scale(self, factor: float) -> "BBox":
        return BBox(self.x0 * factor, self.y0 * factor,
                    self.x1 * factor, self.y1 * factor)

    def translate(self, dx: float, dy: float) -> "BBox":
        return BBox(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def to_int(self) -> tuple[int, int, int, int]:
        return (int(round(self.x0)), int(round(self.y0)),
                int(round(self.x1)), int(round(self.y1)))

    def to_list(self, ndigits: int = 1) -> list[float]:
        return [round(float(self.x0), ndigits), round(float(self.y0), ndigits),
                round(float(self.x1), ndigits), round(float(self.y1), ndigits)]

    @staticmethod
    def from_xywh(x: float, y: float, w: float, h: float) -> "BBox":
        return BBox(x, y, x + w, y + h)

    @staticmethod
    def enclosing(boxes: Iterable["BBox"]) -> Optional["BBox"]:
        out: Optional[BBox] = None
        for b in boxes:
            out = b if out is None else out.union(b)
        return out


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


@dataclass
class Word:
    text: str
    bbox: BBox
    conf: float = -1.0          # 0..100 for OCR output, -1 when unknown
    source: str = "ocr"         # "ocr" or "pdf"


@dataclass
class Line:
    words: list[Word]

    @property
    def bbox(self) -> BBox:
        return BBox.enclosing(w.bbox for w in self.words)  # type: ignore[return-value]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def height(self) -> float:
        """Robust text height: median word height."""
        hs = sorted(w.bbox.height for w in self.words)
        return hs[len(hs) // 2] if hs else 0.0

    @property
    def conf(self) -> float:
        cs = [w.conf for w in self.words if w.conf >= 0]
        return sum(cs) / len(cs) if cs else -1.0


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


class RegionKind(str, Enum):
    TEXT = "text"               # a column / block of listing text
    HEADER = "header"           # running head ("GARY CITY DIRECTORY (1922) 17")
    FOOTER = "footer"
    AD_TOP = "ad_top"           # horizontal strip ad above the text
    AD_BOTTOM = "ad_bottom"     # horizontal strip ad below the text
    AD_SIDE = "ad_side"         # vertical column ad at the left/right edge
    AD_INLINE = "ad_inline"     # boxed / display ad embedded in a text column
    AD_BOX = "ad_box"           # ad on an advertising page (grid of boxes)
    AD_LABEL = "ad_label"       # category caption sitting on top of an ad box

    @property
    def is_ad(self) -> bool:
        return self.value.startswith("ad_") and self is not RegionKind.AD_LABEL


@dataclass
class Region:
    id: str
    kind: RegionKind
    bbox: BBox
    rotation: int = 0                    # degrees the crop was rotated for OCR
    lines: list[Line] = field(default_factory=list)
    column: int = -1                     # reading-order index for TEXT regions
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


# --------------------------------------------------------------------------
# Extracted records
# --------------------------------------------------------------------------


class EntryKind(str, Enum):
    PERSON = "person"           # private resident listing
    BUSINESS = "business"       # a firm / institution
    PROPRIETOR = "proprietor"   # individual running a business at an address
    HEADING = "heading"         # surname / street / category heading
    STREET = "street"           # street-guide line ("1529 Pasko Joseph")
    OTHER = "other"


@dataclass
class Entry:
    """One logical listing reconstructed from one or more printed lines."""
    id: str
    page: int                           # 0-based PDF page index
    region_id: str
    section: str
    kind: EntryKind
    text: str
    bbox: BBox
    lines: list[Line] = field(default_factory=list, repr=False)
    heading: Optional[str] = None       # surname / street / category in force
    fields: dict[str, Any] = field(default_factory=dict)
    advertiser: bool = False            # printed in caps/bold => paid listing
    crop: Optional[str] = None


@dataclass
class Ad:
    id: str
    page: int
    region_id: str
    kind: RegionKind
    bbox: BBox
    rotation: int
    text: str
    business_name: Optional[str] = None
    category: Optional[str] = None
    addresses: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    crop: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageResult:
    page: int
    label: Optional[str]
    width: int
    height: int
    dpi: int
    section: str
    text_source: str                    # "pdf", "ocr" or "mixed"
    regions: list[Region] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    ads: list[Ad] = field(default_factory=list)
    content_bbox: Optional[BBox] = None
    warnings: list[str] = field(default_factory=list)


def to_jsonable(obj: Any) -> Any:
    """Convert models (dataclasses / enums / BBoxes) into JSON-friendly data."""
    if isinstance(obj, BBox):
        return obj.to_list()
    if hasattr(obj, "item") and type(obj).__module__ == "numpy":
        return obj.item()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Line):
        return {"text": obj.text, "bbox": obj.bbox.to_list()}
    if isinstance(obj, Word):
        return {"text": obj.text, "bbox": obj.bbox.to_list(), "conf": obj.conf}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


__all__ = [
    "BBox", "Word", "Line", "Region", "RegionKind", "Entry", "EntryKind",
    "Ad", "PageResult", "to_jsonable", "asdict",
]
