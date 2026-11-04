"""Command line interface.

    polkadot process  DIRECTORY.pdf -o out/       full extraction
    polkadot layout   DIRECTORY.pdf -p 42 -o x.png  draw the segmentation of one page
    polkadot info     DIRECTORY.pdf               page count / text-layer quality
    polkadot strip-text IN.pdf OUT.pdf            make an image-only copy (for testing OCR)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from . import __version__


def _progress(done: int, total: int, page: int) -> None:
    sys.stderr.write(f"\r  analysed {done}/{total} pages (last: {page + 1})   ")
    sys.stderr.flush()
    if done == total:
        sys.stderr.write("\n")


def cmd_process(a: argparse.Namespace) -> int:
    import pymupdf

    from . import export, postprocess
    from .ocr import OcrConfig, tesseract_available
    from .pipeline import (PipelineConfig, parse_page_spec, parse_section_map,
                           process_document, summarize)

    src = Path(a.input)
    out = Path(a.output)
    with pymupdf.open(src) as doc:
        n = doc.page_count
    pages = parse_page_spec(a.pages, n)
    if a.text != "pdf" and not tesseract_available():
        if a.text == "ocr":
            print("error: --text ocr needs the `tesseract` binary on PATH", file=sys.stderr)
            return 2
        print("warning: tesseract not found - using the PDF text layer only", file=sys.stderr)
    cache = Path(a.cache) if a.cache else out / "_pages"
    cfg = PipelineConfig(
        dpi=a.dpi, text_source=a.text, workers=a.workers, deskew=not a.no_deskew,
        min_text_quality=a.min_text_quality, ocr_ads=not a.no_ocr_ads,
        section_overrides=parse_section_map(a.sections, n),
        ocr=OcrConfig(lang=a.lang), cache_dir=str(cache))
    t0 = time.time()
    print(f"PolkaDot {__version__}: {src.name}, {len(pages)} of {n} pages at {a.dpi} dpi")
    works = process_document(str(src), pages, cfg, progress=None if a.quiet else _progress)
    ads = [ad for w in works for ad in w.result.ads]
    entries = [e for w in works for e in w.result.entries]
    postprocess.group_ads(ads)
    postprocess.link_ads_to_entries(ads, entries)
    summary = summarize(works)
    summary["seconds"] = round(time.time() - t0, 1)
    summary["source"] = str(src)
    conf = asdict(cfg)
    conf.pop("section_overrides", None)
    export.export(works, out, crop_entries=a.crop_entries, overlays=not a.no_overlays,
                  html_index=not a.no_html, summary=summary, config=conf, crop_format=a.crop_format)
    if not a.keep_pages and not a.cache:
        import shutil
        shutil.rmtree(cache, ignore_errors=True)
    print(json.dumps(summary, indent=2, default=str))
    print(f"results written to {out}/ (open {out}/index.html to browse)")
    return 0


def cmd_layout(a: argparse.Namespace) -> int:
    import cv2
    import pymupdf

    from . import debug
    from . import imaging as im
    from .layout import analyze_layout
    from .pipeline import render_page

    with pymupdf.open(a.input) as doc:
        page = doc[a.page - 1]
        img = render_page(page, a.dpi)
    skew = im.estimate_skew(im.binarize(im.to_gray(img), a.dpi))
    img = im.deskew(img, skew)
    lay = analyze_layout(img, a.dpi, a.page - 1)
    vis = debug.draw_regions(img, lay.regions, content=lay.content)
    cv2.imwrite(a.output, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    for r in lay.regions:
        print(f"{r.id:>10} {r.kind.value:<10} {r.bbox.to_int()}")
    print(f"skew {skew}°, body x-height {lay.body_h:.1f}px -> {a.output}")
    return 0


def cmd_info(a: argparse.Namespace) -> int:
    import pymupdf

    from .ocr import pdf_words, tesseract_available, text_quality

    with pymupdf.open(a.input) as doc:
        n = doc.page_count
        step = max(1, n // 12)
        print(f"{a.input}: {n} pages, tesseract available: {tesseract_available()}")
        for i in range(0, n, step):
            p = doc[i]
            ws = pdf_words(p, 1.0)
            imgs = len(p.get_images())
            print(f"  page {i + 1:>4}: {len(ws):>5} text-layer words, quality {text_quality(ws):.2f}, "
                  f"{imgs} image(s), label {p.get_label() or '-'}")
    return 0


def cmd_strip(a: argparse.Namespace) -> int:
    """Rasterise pages into a new PDF with no text layer."""
    import pymupdf

    from .pipeline import parse_page_spec

    with pymupdf.open(a.input) as src, pymupdf.open() as dst:
        for i in parse_page_spec(a.pages, src.page_count):
            p = src[i]
            pix = p.get_pixmap(dpi=a.dpi)
            np_ = dst.new_page(width=p.rect.width, height=p.rect.height)
            np_.insert_image(np_.rect, pixmap=pix)
        dst.save(a.output, deflate=True)
    print(f"wrote image-only PDF {a.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="polkadot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="extract structured data and crops from a directory PDF")
    p.add_argument("input")
    p.add_argument("-o", "--output", required=True, help="output directory")
    p.add_argument("-p", "--pages", help="1-based pages, e.g. '1-20,45,300-' (default: all)")
    p.add_argument("--dpi", type=int, default=300, help="render resolution (default 300)")
    p.add_argument("--text", choices=["auto", "pdf", "ocr"], default="auto",
                   help="text source: auto (PDF text layer when it is good, else OCR), "
                        "pdf (never OCR) or ocr (always OCR)")
    p.add_argument("--min-text-quality", type=float, default=0.88,
                   help="auto mode: minimum text-layer quality (0-1) to trust it over OCR")
    p.add_argument("--no-ocr-ads", action="store_true",
                   help="auto mode: use a good text layer for ads too instead of OCR")
    p.add_argument("--lang", default="eng", help="tesseract language(s), e.g. 'eng+deu'")
    p.add_argument("-j", "--workers", type=int, default=0, help="parallel pages (default: CPU count)")
    p.add_argument("--sections", help="force sections, e.g. '1-18:ads,41-131:street,132-502:alphabetical,503-:classified' "
                                      "(sections: front, ads, misc, street, alphabetical, classified)")
    p.add_argument("--crop-entries", choices=["business", "all", "none"], default="business",
                   help="which listings get an image crop (default: businesses / proprietors / advertisers)")
    p.add_argument("--crop-format", choices=["png", "jpg", "webp"], default="png")
    p.add_argument("--no-overlays", action="store_true", help="skip page overlay images")
    p.add_argument("--no-html", action="store_true", help="skip index.html")
    p.add_argument("--no-deskew", action="store_true")
    p.add_argument("--cache", help="keep rendered page images in this directory")
    p.add_argument("--keep-pages", action="store_true", help="keep OUT/_pages page images")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("layout", help="draw the detected layout of one page")
    p.add_argument("input")
    p.add_argument("-p", "--page", type=int, required=True, help="1-based page number")
    p.add_argument("-o", "--output", default="layout.png")
    p.add_argument("--dpi", type=int, default=200)
    p.set_defaults(func=cmd_layout)

    p = sub.add_parser("info", help="show page count and text-layer quality")
    p.add_argument("input")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("strip-text", help="write an image-only (un-OCR'd) copy of a PDF")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("-p", "--pages")
    p.add_argument("--dpi", type=int, default=200)
    p.set_defaults(func=cmd_strip)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return a.func(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
