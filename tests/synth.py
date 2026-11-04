"""Synthetic directory pages for tests (no copyrighted scans needed).

``make_page`` draws a Polk-style page: three listing columns separated by
rules, a boxed strip ad on top, a boxed column ad down the left side and a
boxed strip ad at the bottom, on slightly grey paper.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 200
W, H = 1100, 1460

LISTINGS = [
    ("SMITH", None),
    ("Smith John (Mary) lab h 123 Adams", None),
    ('" Mary (wid James) h 456 Broadway', None),
    ("Smith Peter clk Gary Land Co r 12 W", "7th av"),
    ("Smithson Carl (Anna) gro 901 Madison", "h same"),
    ("Snyder Fred mach I S Co h 77 Polk", None),
    ("Sobel Bros & Co Inc 400 Broadway", None),
    ("Sokol Anna Mrs h 1510 Jefferson", None),
    ("Solomon Isaac tailor 610 Washing-", "ton h 612 same"),
    ("Sommer Paul (Rose) carp h 20 E", "5th av"),
    ("South Side Trust Co 1100 Broadway", None),
    ("Spencer Wm A lab r 15 W 9th av", None),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def make_page(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (W, H), (226, 222, 210))
    d = ImageDraw.Draw(img)
    body = _font(17)
    # --- top strip ad -----------------------------------------------------
    d.rectangle((240, 70, 1010, 190), outline=(20, 20, 20), width=4)
    d.text((300, 82), "ACME COAL COMPANY", font=_font(52), fill=(15, 15, 15))
    d.text((300, 150), "Office 1032 Broadway   Phone 1170", font=_font(22), fill=(15, 15, 15))
    # --- left side ad (stacked letters) ------------------------------------
    d.rectangle((70, 70, 225, 1390), outline=(20, 20, 20), width=4)
    y = 110
    for ch in "DELUXE":
        d.text((120, y), ch, font=_font(60), fill=(15, 15, 15))
        y += 80
    y += 40
    for ch in "CAFE":
        d.text((120, y), ch, font=_font(60), fill=(15, 15, 15))
        y += 80
    d.text((95, 1300), "Phone 202", font=_font(20), fill=(15, 15, 15))
    # --- running head ------------------------------------------------------
    d.text((420, 205), "SAMPLE CITY DIRECTORY (1922)   17", font=_font(16), fill=(30, 30, 30))
    d.line((240, 230, 1010, 230), fill=(30, 30, 30), width=2)
    # --- three listing columns ---------------------------------------------
    xs = [245, 500, 755]
    for x in xs[1:]:
        d.line((x - 8, 240, x - 8, 1255), fill=(40, 40, 40), width=2)
    for ci, x in enumerate(xs):
        y = 245
        k = 0
        while y < 1200:
            first, cont = LISTINGS[(k + ci * 3) % len(LISTINGS)]
            if first.isupper():
                d.text((x + 80, y), first, font=body, fill=(20, 20, 20))
                y += 24
                k += 1
                continue
            # wrap to the column width
            words, line = first.split(), ""
            lines = []
            for wd in words:
                trial = (line + " " + wd).strip()
                if body.getlength(trial) > 235:
                    lines.append(line)
                    line = wd
                else:
                    line = trial
            lines.append(line)
            if cont:
                lines.append(cont)
            for li, t in enumerate(lines):
                d.text((x + (0 if li == 0 else 16), y), t, font=body, fill=(20, 20, 20))
                y += 22
            k += 1
    # --- bottom strip ad ---------------------------------------------------
    d.rectangle((240, 1270, 1010, 1390), outline=(20, 20, 20), width=4)
    d.text((290, 1285), "RADIGAN BROS.", font=_font(48), fill=(15, 15, 15))
    d.text((290, 1350), "637 Broadway  Telephone 1170", font=_font(20), fill=(15, 15, 15))
    # a little scanner noise
    arr = np.asarray(img).astype(np.int16)
    arr += rng.integers(-6, 7, size=arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def page_array(seed: int = 0) -> np.ndarray:
    return np.asarray(make_page(seed))


def make_pdf(path, pages: int = 2) -> None:
    """Image-only (no text layer) PDF of synthetic pages."""
    import pymupdf

    doc = pymupdf.open()
    for i in range(pages):
        im = make_page(i)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        w_pt, h_pt = W * 72 / DPI, H * 72 / DPI
        page = doc.new_page(width=w_pt, height=h_pt)
        page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(str(path))
    doc.close()
