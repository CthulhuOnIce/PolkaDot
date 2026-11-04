import pytest

from polkadot import parse as P
from polkadot.models import BBox, Line, Word
from polkadot.text import assemble_entries, group_lines, join_lines


def test_alphabetical_person():
    out = P.parse_alphabetical("Karr Joseph (Martha) autowkr r 2760 W 10th pl")
    f = out["fields"]
    assert out["kind"] == "person"
    assert f["surname"] == "Karr" and f["given_name"] == "Joseph"
    assert f["spouse"] == "Martha"
    assert f["occupation"] == "autowkr" and f["occupation_expanded"] == "auto worker"
    assert f["residence_type"] == "resides" and f["residence_address"] == "2760 W 10th pl"


def test_alphabetical_ditto_and_widow():
    out = P.parse_alphabetical('" Mary (wid John) h 1512 Jefferson', surname_ctx="Kasarda")
    f = out["fields"]
    assert f["name"] == "Kasarda Mary" and f["widow_of"] == "John" and f["ditto"]
    # OCR glues the ditto mark onto the given name
    assert P.parse_alphabetical("‘Joseph lab h 1 Adams", "Karr")["fields"]["name"] == "Karr Joseph"


def test_alphabetical_employer_expansion():
    f = P.parse_alphabetical("Kasch Andrew (Susan) mach I S Co h 1632 Carolina")["fields"]
    assert f["employer"] == "I S Co"
    assert f["employer_expanded"] == "Illinois Steel Co"


@pytest.mark.parametrize("text,addr", [
    ("Karschuska John (Amanda) soft drinks 1704 Delaware h same", "1704 Delaware"),
    ("Blaga John osteopath 204 Ogden bldg h 720 Maryland", "204 Ogden bldg"),
])
def test_alphabetical_proprietor(text, addr):
    out = P.parse_alphabetical(text)
    assert out["kind"] == "proprietor"
    assert out["fields"]["business_address"] == addr


def test_alphabetical_business_and_surname_keywords():
    out = P.parse_alphabetical("GARY TRUST AND SAVINGS BANK, 5th av and Broadway (See left top lines)")
    assert out["kind"] == "business" and out["advertiser"]
    assert out["fields"]["address"] == "5th av and Broadway"
    assert out["fields"]["see_also"] == ["left top lines"]
    assert P.parse_alphabetical("Katz Apts 120 W 8th av")["kind"] == "business"
    # "Hall" / "Mills" as surnames are not businesses
    assert P.parse_alphabetical("Hall John (Mary) lab h 12 Adams")["kind"] == "person"


def test_street_guide():
    h = P.parse_street_heading("ARTHUR, south from 15th av to 23d av 20 west of Broadway")
    assert h["street"] == "Arthur" and h["description"].startswith("south")
    assert P.parse_street_heading("ADAMS—Contd") == {"street": "Adams", "continued": True}
    f = P.parse_street_line("1529 Pasko Joseph", "Arthur", None)["fields"]
    assert f["address"] == "1529 Arthur" and f["name"] == "Pasko Joseph"
    # second resident of the same house
    f = P.parse_street_line("Jones Laura Mrs", "Adams", "453")["fields"]
    assert f["address"] == "453 Adams"
    assert P.parse_street_line("4,32 Custom Frank", "Adams", None)["fields"]["house_number"] == "432"
    assert P.parse_street_line("1757 Vacant", "Arthur", None)["fields"]["vacant"]


def test_classified():
    out = P.parse_classified("BROADWAY FLORIST, 519 Broadway (See right bottom lines)", "Florists")
    f = out["fields"]
    assert f["name"] == "BROADWAY FLORIST" and f["address"] == "519 Broadway"
    assert f["category"] == "Florists" and f["ad_reference"] == "right bottom lines"
    f = P.parse_classified("Calumet Audit Co 201, 524 Broadway", "Accountants")["fields"]
    assert f["name"] == "Calumet Audit Co" and f["address"] == "201, 524 Broadway"


def test_misc_listing():
    f = P.parse_misc("Temple Beth-El—801 Connecticut, Rev Shuman D Hyman rabbi. Services Friday",
                     "Jewish")["fields"]
    assert f["name"] == "Temple Beth-El" and f["address"] == "801 Connecticut"
    assert f["people"] == ["Rev Shuman D Hyman"]


def test_phones_and_addresses():
    assert P.find_phones("Office 1032 Broadway, Tels. 96 and 1700") == ["96", "1700"]
    assert P.find_phones("Phones 272-4772") == ["272-4772"]
    assert "1032 Broadway" in P.find_addresses("Office 1032 Broadway, Tels. 96 and 1700")
    assert "ne cor 8th av and Massachusetts" in P.find_addresses("Our Savior's—ne cor 8th av and Massachusetts")


def test_ad_fields_stacked_letters():
    lines = [("A GOOD", 20, 0, 20), ("PLACE", 20, 25, 45), ("DE", 60, 100, 160), ("L", 60, 200, 260),
             ("U", 60, 270, 330), ("X", 60, 340, 400), ("E", 60, 410, 470), ("C", 60, 560, 620),
             ("A", 60, 630, 690), ("F", 60, 700, 760), ("E", 60, 770, 830), ("Phone 202", 15, 900, 915)]
    f = P.ad_fields(lines, "A GOOD PLACE DE LUXE CAFE Phone 202")
    assert f["business_name"] == "DE LUXE CAFE"
    assert f["phones"] == ["202"]


def test_ad_fields_firm_name():
    f = P.ad_fields([("Gary Land Company", 50), ("(SUBSIDIARY)", 10), ("Telephone 112", 14)],
                    "Gary Land Company (SUBSIDIARY) Telephone 112")
    assert f["business_name"] == "Gary Land Company" and f["phones"] == ["112"]


def _w(t, x0, y0, x1, y1):
    return Word(t, BBox(x0, y0, x1, y1))


def test_group_lines_mixed_sizes():
    words = [_w("Gary", 0, 10, 60, 50), _w("Emergency", 70, 12, 200, 52),
             _w("Club", 210, 25, 250, 45), _w("24", 0, 70, 20, 84), _w("Hours", 25, 70, 70, 84)]
    lines = group_lines(words)
    assert [l.text for l in lines] == ["Gary Emergency Club", "24 Hours"]


def test_join_lines_dehyphenates():
    assert join_lines(["Solomon Isaac tailor 610 Washing-", "ton h 612 same"]) == \
        "Solomon Isaac tailor 610 Washington h 612 same"
    assert join_lines(["21-", "23 W"]) == "21- 23 W"


def test_hanging_indent_entries():
    col = BBox(0, 0, 300, 400)
    rows = [("KARR", 120, 0), ("Karr Joseph lab r 2760 W", 0, 20), ("10th pl", 16, 40),
            ('" Michael lab h 2760 W 10th pl', 0, 60), ("Karralis John lab h 1237", 0, 80),
            ("Washington", 16, 100)]
    lines = []
    for text, x, y in rows:
        ws, cx = [], x
        for t in text.split():
            ws.append(_w(t, cx, y, cx + 8 * len(t), y + 14))
            cx += 8 * len(t) + 6
        lines.append(Line(ws))
    ents = assemble_entries(lines, col, heading_style="caps")
    assert ents[0].is_heading and ents[0].text == "KARR"
    assert [e.text for e in ents[1:]] == [
        "Karr Joseph lab r 2760 W 10th pl", '" Michael lab h 2760 W 10th pl',
        "Karralis John lab h 1237 Washington"]


def test_expand_abbrev_does_not_misread_initials():
    assert P.expand_abbrev("clk I S Co") == "clerk Illinois Steel Co"
    assert P.expand_abbrev("formn S M Jones") == "foreman S M Jones"


def test_street_guide_apartment_units():
    f = P.parse_street_line("A-1 McCullough James E", "Adams", "529-539")["fields"]
    assert f["unit"] == "A-1" and f["address"] == "529-539 Adams Apt A-1"
    out = P.parse_street_line("B-5 Hall Chester L", "Adams", "529-539")
    assert out["kind"] == "person" and out["fields"]["name"] == "Hall Chester L"
    assert P.parse_street_line("{ 616 Kaplan Max, tailor", "Adams", None)["fields"]["house_number"] == "616"


def test_telephone_listings():
    f = P.parse_telephone("Hagen Glenn W 3111W45thAv Gary------TUrnr 4-8040")["fields"]
    assert f["name"] == "Hagen Glenn W" and f["address"] == "3111 W45thAv"
    assert f["town"] == "Gary" and f["phones"] == ["TUrnr 4-8040"]
    out = P.parse_telephone("Hagerstrom Chas T arch 226Bridge Gary--TUrnr 5-3178")
    assert out["kind"] == "proprietor" and out["fields"]["occupation"] == "arch"
    out = P.parse_telephone("COCA-COLA BOTLG WKS 10thAv&Colfx Gary-TU 6-1573 10th&Colfx Gary ----WE 1-8020")
    assert out["kind"] == "business" and out["fields"]["phones"] == ["TU 6-1573", "WE 1-8020"]
    assert P.parse_telephone("HAGERTY also see HAGGERTY")["kind"] == "other"
    f = P.parse_telephone("5 Cofran IrvinG 7537ChestnutHamnd -----Tlidn 4-1616")["fields"]
    assert f["name"] == "Cofran IrvinG" and f["town"] == "Hammond" and f["address"] == "7537 Chestnut"
    assert P.parse_telephone("Hajduch A groc 1719, 119th....Whitng-1734")["fields"]["phones"] == ["Whiting 1734"]
    f = P.parse_classified("KAISER BROS SERVICE STA 2250 W 15th ------TU 2-7871", "Service Stations")["fields"]
    assert f["name"] == "KAISER BROS SERVICE STA" and f["phones"] == ["TU 2-7871"]


def test_hanging_indent_ignores_stray_fragments_at_left_edge():
    """Word-tails spilling in from the neighbouring column must not shift the
    column margin (which would glue every listing into one)."""
    col = BBox(0, 0, 300, 600)
    lines = []
    y = 0
    for i in range(12):
        if i % 4 == 0:                         # stray "fer-" tail left of the margin
            lines.append(Line([_w("son", -12, y, 8, y + 14)]))
            y += 20
        ws, cx = [], 10
        for t in f"Smith Paul{i} lab h {100 + i} Adams".split():
            ws.append(_w(t, cx, y, cx + 8 * len(t), y + 14))
            cx += 8 * len(t) + 6
        lines.append(Line(ws))
        y += 20
    ents = assemble_entries(lines, col, heading_style="caps")
    assert sum(1 for e in ents if e.text.startswith("Smith")) == 12
