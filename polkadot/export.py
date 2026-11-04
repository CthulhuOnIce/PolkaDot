"""Writing results: JSON / CSV data, cropped images and review overlays.

Output layout::

    OUT/
      document.json          run summary + configuration
      pages.jsonl            one JSON object per page: regions, entries, ads
      entries.csv            every parsed listing
      businesses.csv         businesses, proprietors and advertisers (listings + ads)
      businesses.jsonl       same, as JSON lines
      ads.csv                every advertisement occurrence
      business_index.csv     one row per distinct business: all its listings + ads merged
      crops/ads/…            one image per advertisement
      crops/entries/…        one image per business / proprietor listing
      overlays/…             page images with the detected layout drawn on
      index.html             a browsable contact sheet of everything above
"""

from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from . import debug
from .models import Ad, BBox, Entry, EntryKind, PageResult, to_jsonable
from .pipeline import PageWork


# --------------------------------------------------------------------------
# coordinates
# --------------------------------------------------------------------------


def to_pdf_bbox(b: BBox, work: PageWork) -> list[float]:
    """Deskewed raster pixels -> PDF points of the original page."""
    scale = work.result.dpi / 72.0
    if work.matrix is not None:
        m = np.array(work.matrix, dtype=float)
        inv = cv2.invertAffineTransform(m)
        pts = np.array([[b.x0, b.y0, 1], [b.x1, b.y0, 1], [b.x0, b.y1, 1], [b.x1, b.y1, 1]])
        t = pts @ inv.T
        b = BBox(float(t[:, 0].min()), float(t[:, 1].min()), float(t[:, 0].max()), float(t[:, 1].max()))
    return b.scale(1.0 / scale).to_list(2)


# --------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------


class PageImages:
    """Lazily loads the cached deskewed page rasters."""

    def __init__(self) -> None:
        self._cache: dict[int, np.ndarray] = {}

    def get(self, work: PageWork) -> Optional[np.ndarray]:
        p = work.result.page
        if p not in self._cache:
            if not work.image_path or not os.path.exists(work.image_path):
                return None
            img = cv2.imread(work.image_path)
            if img is None:
                return None
            self._cache = {p: img}           # keep one page in memory
        return self._cache[p]


def save_crop(img: np.ndarray, box: BBox, path: Path, pad: float = 6) -> bool:
    h, w = img.shape[:2]
    b = box.pad(pad, limit=BBox(0, 0, w, h))
    x0, y0, x1, y1 = b.to_int()
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), img[y0:y1, x0:x1]))


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

BUSINESS_KINDS = (EntryKind.BUSINESS, EntryKind.PROPRIETOR)


def entry_record(e: Entry, work: PageWork) -> dict:
    f = e.fields
    return {
        "id": e.id,
        "source": "listing",
        "page": e.page + 1,
        "printed_page": work.result.label,
        "section": e.section,
        "kind": e.kind.value,
        "advertiser": e.advertiser,
        "heading": e.heading,
        "category": f.get("category") or (e.heading if e.section in ("classified", "misc") else None),
        "name": f.get("name"),
        "address": f.get("address") or f.get("business_address") or f.get("residence_address"),
        "business_address": f.get("business_address") or f.get("address"),
        "residence_address": f.get("residence_address"),
        "occupation": f.get("occupation_expanded") or f.get("occupation") or f.get("description"),
        "employer": f.get("employer_expanded") or f.get("employer"),
        "phones": "; ".join(f.get("phones", []) or []),
        "ad_reference": f.get("ad_reference"),
        "linked_ads": "; ".join(f.get("ad_ids", []) or []),
        "text": e.text,
        "crop": e.crop,
        "bbox_px": e.bbox.to_list(),
        "bbox_pdf": to_pdf_bbox(e.bbox, work),
    }


def ad_record(a: Ad, work: PageWork) -> dict:
    return {
        "id": a.id,
        "source": "ad",
        "page": a.page + 1,
        "printed_page": work.result.label,
        "section": work.result.section,
        "kind": a.kind.value,
        "advertiser": True,
        "category": a.category,
        "name": a.business_name or a.meta.get("group_business_name"),
        "address": "; ".join(a.addresses),
        "phones": "; ".join(a.phones),
        "people": "; ".join(a.people),
        "rotation": a.rotation,
        "ad_group": a.meta.get("group"),
        "occurrences": a.meta.get("occurrences"),
        "linked_entries": "; ".join(a.meta.get("linked_entries", [])),
        "text": a.text,
        "text_quality": a.meta.get("text_quality"),
        "crop": a.crop,
        "bbox_px": a.bbox.to_list(),
        "bbox_pdf": to_pdf_bbox(a.bbox, work),
    }


def _write_csv(path: Path, rows: list[dict], fields: Optional[list[str]] = None) -> None:
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(to_jsonable(v)) if isinstance(v, (list, dict)) else to_jsonable(v))
                    for k, v in r.items()})


# --------------------------------------------------------------------------
# main export
# --------------------------------------------------------------------------


def export(works: list[PageWork], out: Path, *, crop_entries: str = "business",
           overlays: bool = True, html_index: bool = True, summary: Optional[dict] = None,
           config: Optional[dict] = None, crop_format: str = "png") -> dict:
    """Write everything for ``works`` into ``out`` and return the summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    images = PageImages()

    entry_rows, biz_rows, ad_rows = [], [], []
    for w in works:
        res = w.result
        img = images.get(w)
        tag = f"p{res.page + 1:04d}"
        if img is not None:
            for a in res.ads:
                path = out / "crops" / "ads" / f"{tag}_{a.id.split('_')[-1]}_{a.kind.value}.{crop_format}"
                if save_crop(img, a.bbox, path):
                    a.crop = str(path.relative_to(out))
            for e in res.entries:
                want = (crop_entries == "all" and e.kind is not EntryKind.HEADING) or \
                       (crop_entries == "business" and (e.kind in BUSINESS_KINDS or e.advertiser))
                if want:
                    path = out / "crops" / "entries" / f"{tag}_{e.id.split('_')[-1]}.{crop_format}"
                    if save_crop(img, e.bbox, path, pad=4):
                        e.crop = str(path.relative_to(out))
            if overlays:
                vis = debug.draw_regions(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), res.regions,
                                         res.entries, res.content_bbox)
                scale = 1400.0 / max(vis.shape[:2])
                if scale < 1:
                    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                op = out / "overlays" / f"page_{res.page + 1:04d}.jpg"
                op.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(op), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        for e in res.entries:
            if e.kind is EntryKind.HEADING:
                continue
            rec = entry_record(e, w)
            entry_rows.append(rec)
            if e.kind in BUSINESS_KINDS or e.advertiser:
                biz_rows.append(rec)
        for a in res.ads:
            rec = ad_record(a, w)
            ad_rows.append(rec)
            biz_rows.append(rec)

    with open(out / "pages.jsonl", "w", encoding="utf-8") as fh:
        for w in works:
            r = w.result
            obj = {
                "page": r.page + 1, "printed_page": r.label, "section": r.section,
                "text_source": r.text_source, "dpi": r.dpi, "width": r.width, "height": r.height,
                "skew_deg": w.skew, "content_bbox": r.content_bbox.to_list() if r.content_bbox else None,
                "warnings": r.warnings,
                "regions": [{"id": g.id, "kind": g.kind.value, "bbox_px": g.bbox.to_list(),
                             "bbox_pdf": to_pdf_bbox(g.bbox, w), "rotation": g.rotation,
                             "column": g.column, "meta": to_jsonable(g.meta), "text": g.text}
                            for g in r.regions],
                "entries": [entry_record(e, w) | {"fields": to_jsonable(e.fields)} for e in r.entries],
                "ads": [ad_record(a, w) for a in r.ads],
            }
            fh.write(json.dumps(to_jsonable(obj), ensure_ascii=False) + "\n")

    _write_csv(out / "entries.csv", entry_rows)
    _write_csv(out / "ads.csv", ad_rows)
    biz_fields = ["id", "source", "page", "printed_page", "section", "kind", "advertiser",
                  "category", "name", "address", "phones", "occupation", "employer",
                  "ad_reference", "linked_ads", "linked_entries", "ad_group", "occurrences",
                  "text", "crop", "bbox_px", "bbox_pdf"]
    _write_csv(out / "businesses.csv", biz_rows, biz_fields)
    with open(out / "businesses.jsonl", "w", encoding="utf-8") as fh:
        for r in biz_rows:
            fh.write(json.dumps(to_jsonable(r), ensure_ascii=False) + "\n")

    from .postprocess import consolidate
    index = consolidate([e for w in works for e in w.result.entries],
                        [a for w in works for a in w.result.ads])
    _write_csv(out / "business_index.csv", [
        {k: ("; ".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items()}
        for r in index])
    with open(out / "business_index.jsonl", "w", encoding="utf-8") as fh:
        for r in index:
            fh.write(json.dumps(to_jsonable(r), ensure_ascii=False) + "\n")

    doc = {"summary": summary or {}, "config": config or {},
           "files": {"pages": "pages.jsonl", "entries": "entries.csv", "ads": "ads.csv",
                     "businesses": "businesses.csv", "business_index": "business_index.csv"},
           "counts": {"entries": len(entry_rows), "businesses": len(biz_rows), "ads": len(ad_rows),
                      "distinct_businesses": len(index)}}
    with open(out / "document.json", "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(doc), fh, indent=2, ensure_ascii=False)
    if html_index:
        write_html_index(out, biz_rows, works)
    return doc


def write_html_index(out: Path, rows: list[dict], works: list[PageWork]) -> None:
    """A static, dependency-free page for eyeballing the results."""
    esc = html.escape
    cards = []
    for r in rows:
        img = f'<img loading="lazy" src="{esc(r["crop"])}">' if r.get("crop") else ""
        meta = " · ".join(x for x in [
            f'p.{r["page"]}' + (f' ({r["printed_page"]})' if r.get("printed_page") else ""),
            r.get("section") or "", r.get("kind") or "", r.get("category") or ""] if x)
        details = " · ".join(x for x in [r.get("address") or "", r.get("phones") or ""] if x)
        cards.append(
            f'<div class="card {esc(r["source"])}" data-text="{esc((r.get("name") or "") + " " + (r.get("text") or "")).lower()}">'
            f'<div class="img">{img}</div><div class="name">{esc(r.get("name") or "—")}</div>'
            f'<div class="meta">{esc(meta)}</div><div class="det">{esc(details)}</div>'
            f'<details><summary>text</summary><pre>{esc(r.get("text") or "")}</pre></details></div>')
    pages = "".join(
        f'<a href="overlays/page_{w.result.page + 1:04d}.jpg">{w.result.page + 1}'
        f'<small>{esc(w.result.section)}</small></a>' for w in works)
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PolkaDot results</title>
<style>
:root{{--bg:#fafaf7;--fg:#222;--card:#fff;--muted:#777;--line:#ddd;--ad:#b5651d;--listing:#2b6cb0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16161a;--fg:#eee;--card:#222228;--muted:#999;--line:#333}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}}
header{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:12px}}
input{{flex:1;min-width:200px;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--listing);border-radius:6px;padding:8px;overflow:hidden}}
.card.ad{{border-left-color:var(--ad)}}
.img img{{max-width:100%;max-height:220px;display:block;margin:auto}}
.name{{font-weight:600;margin-top:6px}} .meta,.det{{color:var(--muted);font-size:12px}}
pre{{white-space:pre-wrap;font-size:11px}} nav{{margin:12px 0;display:flex;flex-wrap:wrap;gap:4px}}
nav a{{font-size:11px;padding:2px 5px;border:1px solid var(--line);border-radius:4px;color:var(--fg);text-decoration:none}}
nav small{{color:var(--muted);margin-left:3px}}
</style></head><body>
<header><strong>PolkaDot</strong><span>{len(rows)} businesses &amp; ads from {len(works)} pages</span>
<input id="q" placeholder="filter by name / text…"></header>
<details><summary>Page overlays</summary><nav>{pages}</nav></details>
<div class="grid" id="grid">{''.join(cards)}</div>
<script>
const q=document.getElementById('q');q.addEventListener('input',()=>{{const v=q.value.toLowerCase();
document.querySelectorAll('.card').forEach(c=>c.style.display=c.dataset.text.includes(v)?'':'none')}});
</script></body></html>"""
    (out / "index.html").write_text(doc, encoding="utf-8")
