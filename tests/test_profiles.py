import pytest
import synth

from polkadot.layout import LayoutConfig
from polkadot.ocr import tesseract_available
from polkadot.pipeline import assign_sections
from polkadot.profiles import PROFILES, detect_profile, get_profile


def test_profiles_exist_and_general_is_default_layout():
    assert {"general", "polk-1920s", "telephone-1950s"} <= set(PROFILES)
    assert get_profile("general").layout == LayoutConfig()
    with pytest.raises(ValueError):
        get_profile("nonsense")


def test_profile_section_maps():
    assert get_profile("polk-1920s").map_section("telephone") == "alphabetical"
    assert get_profile("telephone-1950s").map_section("alphabetical") == "telephone"
    assert get_profile("general").map_section("telephone") == "telephone"


class _W:
    """Minimal stand-in for PageWork in section assignment."""
    def __init__(self, page, feats):
        from polkadot.models import PageResult
        self.result = PageResult(page, None, 1, 1, 200, "", "")
        self.features = feats


def _feats(**kw):
    f = {"n_lines": 120, "ad_frac": 0.2, "num_start": 0.0, "res": 0.0, "see": 0.0,
         "star": 0, "titled": 0.0, "dash": 0.0, "markers": [], "phone": 0.0, "yellow": 0}
    f.update(kw)
    return f


def test_profile_relabels_detected_sections():
    works = [_W(i, _feats(phone=0.6)) for i in range(5)]
    assign_sections(works, {}, get_profile("polk-1920s"))
    assert {w.result.section for w in works} == {"alphabetical"}
    works = [_W(i, _feats(res=0.5)) for i in range(5)]
    assign_sections(works, {}, get_profile("telephone-1950s"))
    assert {w.result.section for w in works} == {"telephone"}


@pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")
def test_detect_profile_on_synthetic_city_directory(tmp_path):
    pdf = tmp_path / "s.pdf"
    synth.make_pdf(pdf, pages=4)
    det = detect_profile(str(pdf), sample=4)
    # the synthetic page is set as a city directory ("lab h 123 Adams")
    assert det.profile in ("polk-1920s", "general")
