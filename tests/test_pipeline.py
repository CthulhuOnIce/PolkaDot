import csv
import json

import pytest
import synth

from polkadot.cli import main
from polkadot.models import EntryKind, RegionKind
from polkadot.ocr import tesseract_available
from polkadot.pipeline import (PipelineConfig, parse_page_spec, parse_section_map,
                               process_document)

needs_ocr = pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")


def test_page_specs():
    assert parse_page_spec("1-3,7,9-", 10) == [0, 1, 2, 6, 8, 9]
    assert parse_page_spec(None, 3) == [0, 1, 2]
    assert parse_section_map("1-2:ads,3-:alphabetical", 4) == {
        0: "ads", 1: "ads", 2: "alphabetical", 3: "alphabetical"}
    with pytest.raises(ValueError):
        parse_section_map("1:nonsense", 2)


@needs_ocr
def test_ocr_pipeline_on_image_only_pdf(tmp_path):
    pdf = tmp_path / "synthetic.pdf"
    synth.make_pdf(pdf, pages=1)
    cfg = PipelineConfig(dpi=200, workers=1, cache_dir=str(tmp_path / "cache"),
                         section_overrides={0: "alphabetical"})
    (work,) = process_document(str(pdf), None, cfg)
    res = work.result
    assert res.text_source == "ocr"
    kinds = {r.kind for r in res.regions}
    assert {RegionKind.AD_TOP, RegionKind.AD_SIDE, RegionKind.AD_BOTTOM, RegionKind.TEXT} <= kinds
    names = {a.business_name for a in res.ads}
    assert any("ACME COAL" in (n or "") for n in names)
    assert any("RADIGAN" in (n or "") for n in names)
    top = next(a for a in res.ads if a.kind is RegionKind.AD_TOP)
    assert "1170" in top.phones
    people = [e for e in res.entries if e.fields.get("name") == "Smith John"]
    assert people and people[0].fields.get("residence_address") == "123 Adams"
    biz = [e for e in res.entries if e.kind is EntryKind.BUSINESS]
    assert any("Trust Co" in (e.fields.get("name") or "") for e in biz)
    # the ditto line inherits the surname of the line before it
    assert any(e.fields.get("name") == "Smith Mary" and e.fields.get("widow_of") == "James"
               for e in res.entries)


def test_text_layer_pipeline_without_ocr(tmp_path):
    """A born-digital page: text comes straight from the PDF text layer."""
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=396, height=526)
    page.draw_rect(pymupdf.Rect(80, 25, 365, 68), width=1.5)
    page.insert_text((100, 52), "ACME COAL COMPANY", fontsize=18)
    page.insert_text((100, 63), "Phone 1170", fontsize=8)
    y = 90
    for i in range(40):
        page.insert_text((88, y), f"Smith John{i} (Mary) lab h {100 + i} Adams", fontsize=6.5)
        page.insert_text((235, y), f"Jones Paul{i} clk r {200 + i} Polk", fontsize=6.5)
        y += 9
    pdf = tmp_path / "digital.pdf"
    doc.save(str(pdf))
    cfg = PipelineConfig(dpi=200, workers=1, text_source="pdf", section_overrides={0: "alphabetical"})
    (work,) = process_document(str(pdf), None, cfg)
    res = work.result
    assert res.text_source == "pdf"
    names = {e.fields.get("name") for e in res.entries}
    assert "Smith John3" in names and "Jones Paul7" in names
    e = next(e for e in res.entries if e.fields.get("name") == "Smith John3")
    assert e.fields["residence_address"] == "103 Adams"


@needs_ocr
def test_cli_process_writes_outputs(tmp_path):
    pdf = tmp_path / "synthetic.pdf"
    synth.make_pdf(pdf, pages=2)
    out = tmp_path / "out"
    rc = main(["process", str(pdf), "-o", str(out), "--dpi", "200", "-j", "2", "-q",
               "--sections", "1-2:alphabetical"])
    assert rc == 0
    for f in ["document.json", "pages.jsonl", "entries.csv", "ads.csv", "businesses.csv",
              "businesses.jsonl", "index.html"]:
        assert (out / f).exists(), f
    ads = list(csv.DictReader(open(out / "ads.csv")))
    assert len(ads) >= 6
    for a in ads:
        assert (out / a["crop"]).exists()
    # the same ad on both pages is grouped together
    acme = [a for a in ads if "ACME" in a["name"]]
    assert len(acme) == 2 and acme[0]["ad_group"] == acme[1]["ad_group"]
    assert acme[0]["occurrences"] == "2"
    biz = list(csv.DictReader(open(out / "businesses.csv")))
    listing = [b for b in biz if b["source"] == "listing"]
    assert listing and all((out / b["crop"]).exists() for b in listing if b["crop"])
    pages = [json.loads(l) for l in open(out / "pages.jsonl")]
    assert len(pages) == 2 and pages[0]["regions"]
    bb = pages[0]["ads"][0]["bbox_pdf"]
    assert 0 <= bb[0] < bb[2] <= 396 + 1


def test_cli_strip_text(tmp_path):
    import pymupdf
    src = tmp_path / "a.pdf"
    d = pymupdf.open()
    p = d.new_page()
    p.insert_text((72, 72), "Hello directory")
    d.save(str(src))
    dst = tmp_path / "b.pdf"
    assert main(["strip-text", str(src), str(dst)]) == 0
    with pymupdf.open(dst) as d2:
        assert d2[0].get_text().strip() == ""
        assert d2[0].get_images()
