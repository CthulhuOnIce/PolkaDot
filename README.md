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
  index.html          searchable contact sheet of all of the above (wildcards: see below)
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
directory), `telephone` (telephone-book white pages), `street` (street & avenue guide / householders' directory),
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
* *telephone* — mid-century telephone books (white pages):
  `Hagen Glenn W 3111W45thAv Gary------TUrnr 4-8040` → name, occupation
  (`arch`, `DDS`…), address, town (abbreviations such as `Hamnd`, `EChgo`,
  `CalCty` expanded) and exchange-name phone numbers (`TUrnr 4-8040`,
  `Whiting 2585 W`). Firms set in capitals are `business`; `X also see Y`
  cross-references are kept as `other`. Their yellow pages are parsed as
  *classified*.
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

## Searching the viewer

`index.html` shows every business and ad with its crop. The search box takes
plain text (substring match) or wildcards, which match whole words:

| query | finds |
|---|---|
| `5?? Broadway` | any three-character house number starting with 5 on Broadway (the 500 block) |
| `##th av` | `10th av` … `99th av` (`#` = one digit) |
| `Gary*` | words starting with *Gary* (`Gary`, `Gary's`) |
| `Gary * Co` | *Gary*, then any words, then *Co* (`Gary Land Co`, `Gary Ice & Coal Co`) |

Searches cover the name, parsed address, phone numbers, category and the
OCR'd text, ignoring case. The query is kept in the URL (`index.html#q=...`)
so a search can be bookmarked or shared.

## Install

PolkaDot is a [uv](https://docs.astral.sh/uv/) project (any PEP 517 tool such
as pip works too). OCR also needs the
[Tesseract](https://github.com/tesseract-ocr/tesseract) binary
(`apt install tesseract-ocr`, `brew install tesseract`, or the UB Mannheim
Windows installer); without it PolkaDot still runs on PDFs that already have
a text layer.

Run it without installing anything permanently:

```
uvx --from git+https://github.com/CthulhuOnIce/PolkaDot polkadot process DIRECTORY.pdf -o out/
```

Install the `polkadot` command for your user:

```
uv tool install git+https://github.com/CthulhuOnIce/PolkaDot
```

Work on a checkout (creates `.venv` from the pinned `uv.lock`):

```
git clone https://github.com/CthulhuOnIce/PolkaDot && cd PolkaDot
uv sync
uv run polkadot process DIRECTORY.pdf -o out/
```

`uv sync` already installs PolkaDot into `.venv` in editable mode: changes
to the source take effect immediately. (Don't use `uv run pip install -e .`:
uv environments have no pip, so that runs your system pip instead.)

With pip instead: `pip install -e .` inside a virtualenv (Python ≥ 3.10).

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

### Profiles

Directory styles pull layout and parsing in different directions, so
settings are grouped into profiles:

| profile | for |
|---|---|
| `general` | all-round settings that adapt page by page (the fallback) |
| `polk-1920s` | R. L. Polk-style city directories, c. 1890s–1940s |
| `telephone-1950s` | mid-century telephone books (white + yellow pages) |

`--profile auto` (the default) reads a strip of listings from a few sample
pages and picks `polk-1920s` when lines end in residences (`h 123 Adams`),
`telephone-1950s` when they end in exchange numbers (`TUrnr 4-8040`), and
`general` when the evidence is unclear. `polkadot profiles DIRECTORY.pdf`
shows what it would choose. A profile fixes the section vocabulary (no
"telephone" pages inside a city directory and vice versa) and how classified
headings are recognised, and carries its own layout settings
(`polkadot/profiles.py`). On the 1958 Calumet telephone book the
`telephone-1950s` profile raises yellow-page listings with a parsed phone
number from 59% to 62% and with an address from 63% to 69% over `general`,
and halves the number of bogus categories. New directory styles should get a
profile of their own and be tuned there, with a full-book before/after comparison, so that
changes for one style cannot regress another.

Useful `process` options:

* `--profile auto|general|polk-1920s|telephone-1950s` — directory style (see above).
* `--text auto|pdf|ocr` — `auto` (default) uses the PDF's text layer where its
  quality is at least `--min-text-quality` (0.88) and OCRs everything else;
  ads are always OCR'd in `auto` unless `--no-ocr-ads`.
* `--sections "1-18:ads,41-131:street,132-502:alphabetical,503-:classified"` —
  force sections when automatic detection guesses wrong (1-based pages;
  sections: front, ads, misc, street, alphabetical, classified, telephone).
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
| `polkadot/profiles.py` | directory-style profiles and auto-detection |
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

Also tested on the **1958 Calumet Region telephone directory** (554 pages,
<https://archive.org/details/calumet-district-directory-1958-images_202301>):
four dense columns per page with rules between them, boxed notices and
yellow pages.

## Tests

```
uv run pytest
```

The tests build synthetic directory pages (no scans needed). OCR tests are
skipped when Tesseract is not installed, and the viewer-search test when
Node.js is not. CI runs them on Python 3.10, 3.12 and 3.13
(`uv run --locked`, so a stale `uv.lock` fails the build; refresh it with
`uv lock` after changing dependencies).

## Limits and tuning

* OCR quality bounds everything downstream; scans below ~200 dpi or with heavy
  bleed-through give noisier fields. Render at 300 dpi (default) for scans.
* Field parsing follows Polk conventions; other publishers' abbreviations can
  be added to `polkadot/data/abbreviations.json`.
* Telephone-book yellow pages are the hardest layout: listings, bold display
  listings and boxed ads share the columns. Ads are cropped reliably;
  roughly half of yellow-page listings get a parsed address, and a few pages
  still have a column read as advertising.
* Long runs: every analysed page is checkpointed in `OUT/_pages`, so an
  interrupted `process` resumes when re-run with the same arguments. A
  554-page telephone book takes about 50 minutes on 4 cores at 300 dpi,
  a 558-page 1922 Polk directory about 20.
* When changing heuristics, re-run a whole book with `--keep-pages` and
  compare `entries.csv` / `pages.jsonl` against a previous run: layout fixes
  for one directory style can easily break another.
* Layout heuristics live in `polkadot/layout.py` (`LayoutConfig`) and
  `polkadot/imaging.py`. `polkadot layout` and the `overlays/` folder show
  what was detected; please open an issue with a page image when it is wrong.

## License

PolkaDot is free software: you can redistribute it and/or modify it under the
terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. See [`LICENSE`](LICENSE).
