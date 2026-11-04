# PolkaDot

Tools for ripping apart scanned **city directories** — R. L. Polk & Co. and
the many publishers that copied their format — into **structured data** plus
a **cropped image reference** for every business, listing and advertisement.

It works on PDFs with or without an OCR text layer: a usable text layer is
read directly, anything else goes through Tesseract.

```
polkadot process "Polk's Gary City Directory 1922.pdf" -o gary1922/
```

```
gary1922/
  businesses.csv      every firm, proprietor and advertiser (listings + ads), with crops
  entries.csv         every parsed listing: residents, firms, street-guide lines, …
  ads.csv             every advertisement occurrence (grouped when an ad repeats)
  business_index.csv  one row per distinct business: its listings in every section + its ads
  pages.jsonl         full per-page detail: regions, lines, entries, ads, coordinates
  crops/ads/          one image per advertisement
  crops/entries/      one image per business / proprietor / advertiser listing
  overlays/           every page with the detected layout drawn on it
  index.html          searchable contact sheet of all of the above
```

## What it understands

**Page layout.** Directory pages wrap their listing columns in advertising.
PolkaDot segments each page into

| region | what it is |
|---|---|
| `text` | listing columns, in reading order |
| `header` / `footer` | running heads (`GARY CITY DIRECTORY (1922) 17`) — the printed page number is read from them |
| `ad_top` / `ad_bottom` | horizontal strip ads above / below the columns |
| `ad_side` | vertical column ads down the outer edge (including stacked letters and rotated type) |
| `ad_inline` | boxed display ads set inside a column |
| `ad_box` | ads on whole advertising pages (the yellow pages of ad grids) |
| `ad_label` | the category caption above an ad box (`DYERS AND CLEANERS`) |

It deskews pages, finds the printed area while ignoring book covers, the
gutter shadow and slivers of the facing page, detects ruling lines, boxes and
reverse-video (white-on-black) panels, and splits the rest with an XY-cut
classified by type size. Analysis runs at a fixed working resolution so results
do not depend on the render DPI.

**Sections.** Each page is classified as `alphabetical` (the general
directory), `street` (street & avenue guide / householders' directory),
`classified` (buyers' guide / classified business directory), `misc`
(churches, lodges, schools, officials…), `ads` or `front`, from its content
and from section titles, smoothed over neighbouring pages. You can override
it (`--sections`).

**Listings.** Hanging-indent entries are reassembled across lines (hyphenated
words are re-joined), headings carry context across columns and pages, and
fields are extracted per section:

* *alphabetical* — `Karr Joseph (Martha) autowkr r 2760 W 10th pl` →
  surname, given name, spouse / widow-of, occupation (with Polk abbreviations
  expanded: *autowkr → auto worker*, *I S Co → Illinois Steel Co*), employer,
  business address, residence type (`h` householder, `r` resides, `bds`
  boards…) and address. Ditto marks (`"`) take the surname above. Entries are
  classed as `person`, `proprietor` (someone running their own business:
  `Kaserich Eli auto repr 2122 Adams h same`) or `business` (firms and
  institutions: `Katz Apts 120 W 8th av`). Capitals mark paid listings
  (`advertiser`).
* *street guide* — street headings (`ARTHUR, south from 15th av…`,
  `ADAMS—Contd`) and `1529 Pasko Joseph` lines become full addresses; extra
  residents at the same number inherit it; `Vacant` is flagged.
* *classified* — category headings (`*Floral Designs`), firm name, address and
  the ad cross-reference (`(See right bottom lines)`).
* *misc* — `Temple Beth-El—801 Connecticut, Rev Shuman D Hyman rabbi…` → name,
  address, people, under its heading (`Jewish`).

**Advertisements.** Each ad is OCR'd on its own: tall ads are tried at 0/90/270°
for rotated type, reverse-video panels are read inverted, and a reduced-scale
pass catches very large display type. The advertiser's name (the dominant type,
reassembling stacked letters such as `D/E/L/U/X/E C/A/F/E`), addresses, phone
numbers, people and category are extracted. Polk repeated strip ads across many
pages, so identical ads are grouped (`ad_group`, `occurrences`), and ads are
linked to the listings of the same business (`linked_entries` / `linked_ads`).

Every record carries its page, printed page, pixel box *and* PDF-point box on
the original page, and a crop image path.

## Install

```
pip install -e .            # Python ≥ 3.10
```

OCR needs the [Tesseract](https://github.com/tesseract-ocr/tesseract) binary
(`apt install tesseract-ocr`, `brew install tesseract`, or the UB Mannheim
Windows installer). Without it PolkaDot still runs on PDFs that already have a
text layer.

## Usage

```
polkadot info DIRECTORY.pdf                    # pages, text-layer quality, OCR available?
polkadot process DIRECTORY.pdf -o out/         # everything
polkadot process DIRECTORY.pdf -o out/ -p 130-160 --dpi 300 -j 8
polkadot process DIRECTORY.pdf -o out/ --keep-pages   # keep page images + work.pkl ...
polkadot reparse out/ --sections "503-:classified"    # ... to re-parse / re-export without OCR
polkadot layout DIRECTORY.pdf -p 42 -o p42.png # check the segmentation of one page
polkadot strip-text IN.pdf OUT.pdf -p 1-20     # image-only copy, to test the OCR path
```

Useful `process` options:

* `--text auto|pdf|ocr` — `auto` (default) uses the PDF's text layer where its
  quality is at least `--min-text-quality` (0.88) and OCRs everything else;
  ads are always OCR'd in `auto` unless `--no-ocr-ads`.
* `--sections "1-18:ads,41-131:street,132-502:alphabetical,503-:classified"` —
  force sections when automatic detection guesses wrong (1-based pages).
* `--crop-entries business|all|none` — which listings get image crops.
* `-j N` — pages processed in parallel (default: all cores).
* `--lang eng+deu` — Tesseract languages for foreign-language directories.

Use as a library:

```python
from polkadot.pipeline import PipelineConfig, process_document
from polkadot import postprocess, export

works = process_document("gary1922.pdf", pages=range(130, 140),
                         cfg=PipelineConfig(dpi=300, cache_dir="cache"))
ads = [a for w in works for a in w.result.ads]
postprocess.group_ads(ads)
for w in works:
    for e in w.result.entries:
        print(e.section, e.kind.value, e.fields.get("name"), e.fields.get("address"))
export.export(works, "out/")
```

## Code map

| module | role |
|---|---|
| `polkadot/imaging.py` | binarisation, deskew, printed-area detection, rules, boxes, reverse-video panels |
| `polkadot/layout.py` | page segmentation into columns / heads / ad regions (`analyze_layout`) |
| `polkadot/ocr.py` | Tesseract wrapper (rotations, inversion, multi-scale, rule erasing) and PDF text-layer reader |
| `polkadot/text.py` | words → lines → hanging-indent entries, heading detection |
| `polkadot/parse.py` | field grammars for each section, addresses, phones, ad fields, abbreviations |
| `polkadot/pipeline.py` | per-page analysis (parallel), section detection, entry / ad building, `reparse` |
| `polkadot/postprocess.py` | repeated-ad grouping, ad ↔ listing links, consolidated business index |
| `polkadot/export.py` | CSV / JSON / crops / overlays / `index.html` |
| `polkadot/cli.py` | the `polkadot` command |

## Where the sample data comes from

PolkaDot was developed against **Polk's Gary (Indiana) City Directory 1922**
(558 pages) from the Internet Archive
(<https://archive.org/details/gary-city-directory-1922-images>), which has
everything this tool targets: yellow advertising pages of boxed ads with
category captions, top/bottom strip ads and side column ads on every listing
page, stacked-letter and rotated display type, reverse-video ads, street guide,
general directory, classified business directory and miscellaneous lists.
The Internet Archive holds hundreds of other Polk directories (search
*"Polk directory"*); their `…_text.pdf` / original PDFs both work.

## Tests

```
pip install -e .[dev]
pytest
```

The tests build synthetic directory pages (no scans needed). OCR tests are
skipped when Tesseract is not installed.

## Limits and tuning

* OCR quality bounds everything downstream; scans below ~200 dpi or with heavy
  bleed-through give noisier fields. Render at 300 dpi (default) for scans.
* Field parsing follows Polk conventions; other publishers' abbreviations can
  be added to `polkadot/data/abbreviations.json`.
* Layout heuristics live in `polkadot/layout.py` (`LayoutConfig`) and
  `polkadot/imaging.py`. `polkadot layout` and the `overlays/` folder show
  what was detected; please open an issue with a page image when it is wrong.
