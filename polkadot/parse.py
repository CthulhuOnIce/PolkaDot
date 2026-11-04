"""Field extraction for directory listings and advertisements.

The grammar of a Polk-style *alphabetical* listing is roughly::

    Surname Given [Initial] [(Spouse | wid Husband)] [occupation] [employer]
        [business-address] [h|r|bds|rms residence-address|same]

The *street guide* is ``[house-number] Name ...`` under a street heading,
and the *classified business directory* is ``NAME[,] address [(See ...)]``
under a category heading.  These parsers are deliberately forgiving: OCR
noise and inconsistent typesetting are normal.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Any, Optional

# --------------------------------------------------------------------------
# vocabularies
# --------------------------------------------------------------------------

RES_MARKERS = {"h", "r", "bds", "rms", "res", "b"}

BUSINESS_WORDS = {
    "co", "company", "corp", "corporation", "inc", "incorporated", "bros", "brothers",
    "sons", "son", "&", "assn", "association", "society", "soc", "union", "club",
    "lodge", "bank", "trust", "savings", "hotel", "apts", "apartments", "flats",
    "store", "stores", "shop", "market", "garage", "laundry", "bakery", "dairy",
    "works", "mfg", "factory", "mill", "mills", "theatre", "theater", "cafe",
    "restaurant", "lunch", "room", "rooms", "hall", "church", "school", "college",
    "academy", "hospital", "home", "agency", "exchange", "foundry", "press",
    "printing", "publishing", "ry", "railway", "railroad", "r", "telephone",
    "electric", "gas", "coal", "ice", "lumber", "supply", "furniture", "hardware",
    "drug", "drugs", "pharmacy", "studio", "parlor", "parlors", "florist",
    "realty", "estate", "insurance", "motor", "motors", "auto", "sales", "service",
    "transfer", "storage", "express", "garage", "plumbing", "heating", "tailors",
    "cleaners", "dyers", "undertaking", "construction", "contracting", "building",
    "loan", "investment", "title", "abstract", "the", "temple", "synagogue",
    "mission", "board", "dept", "department", "office", "bldg", "station",
    "cemetery", "post", "council", "court", "library", "y", "m", "c", "a",
}
# words that alone do not make something a business when the line looks
# like "Surname Given ..."
WEAK_BUSINESS_WORDS = {"the", "y", "m", "c", "a", "r", "home", "room", "rooms",
                       "son", "post", "court", "board", "auto", "estate", "&"}

STRONG_BUSINESS_RE = re.compile(
    r"\b(Co|Company|Corp|Corporation|Inc|Bros|Assn|Association|Society|Club|Lodge|"
    r"Bank|Hotel|Apts|Apartments|Garage|Laundry|Bakery|Dairy|Works|Mfg|Theatre|"
    r"Theater|Cafe|Restaurant|Church|School|College|Hospital|Agency|Exchange|Foundry|"
    r"Ry|Railway|Railroad|Store|Stores|Shop|Market|Studio|Parlor|Parlors|Florist|"
    r"Realty|Pharmacy|Temple|Mission|Union|Hall|Cemetery|Library|Transfer|Storage|"
    r"Lumber|Supply|Furniture|Hardware|Undertaking|Cleaners|Dyers|Sons|Mills?)\b\.?")

PROPRIETOR_OCCUPATIONS = {
    "gro", "grocer", "groceries", "confr", "restr", "saloon", "druggist", "phys",
    "physician", "dentist", "lawyer", "atty", "barber", "tailor", "shoemkr",
    "shoe repr", "butcher", "meats", "baker", "bakery", "plumber", "contr", "real est",
    "real estate", "ins", "insurance", "dry goods", "furn", "hardware", "jeweler",
    "jwlr", "florist", "undertaker", "photographer", "cigars", "tobacco", "soft drinks",
    "billiards", "pool", "notions", "milliner", "dressmkr", "drsmkr", "auto repr",
    "garage", "shoes", "clothing", "fruits", "fruit", "hotel", "rooming house",
    "boarding house", "lunch room", "junk", "coal", "feed", "laundry", "cleaner",
    "osteopath", "chiropractor", "architect", "veterinarian", "midwife", "notary",
}

STREET_SUFFIX = r"(?:av|ave|avenue|st|street|pl|place|rd|road|blvd|ct|court|ter|way|dr|drive|ln|lane|bldg|aly|alley|sq|pk|park|hwy|pike|row)\b\.?"

# 1529 Pasko  /  2760 W 10th pl  /  21-23 W 17th av  /  rear 1001 Broadway
ADDRESS_RE = re.compile(
    r"(?:\b(?:rear|r|bsmt|upstairs)\s+)?"
    r"\b\d{1,5}(?:\s?1/2|½)?(?:\s?[-–]\s?\d{1,5})?(?:,\s*\d{1,5})?"     # number (room, number)
    r"\s+(?:[NSEW]\s+)?"
    r"(?:\d{1,3}(?:st|nd|rd|th|d)\b\.?(?:\s+" + STREET_SUFFIX + r")?"  # 10th av
    r"|[A-Z][A-Za-z'’]+\.?(?:[ -][A-Z][A-Za-z'’]+\.?)?(?:\s+" + STREET_SUFFIX + r")?)",
    re.UNICODE)

CORNER_RE = re.compile(
    r"\b(?:ne|nw|se|sw|NE|NW|SE|SW|Ne|Nw|Se|Sw|n\s?e|s\s?w|n\s?w|s\s?e)\s+(?:cor|corner)\.?\s+"
    r"[A-Za-z0-9 ]+?(?:\s+(?:and|&)\s+[A-Z0-9][A-Za-z0-9]*(?:\s+" + STREET_SUFFIX + r")?)")

PHONE_RE = re.compile(
    r"\b(?:Phones?|Tels?|Telephones?|Tel\.?|Ph|Phone\s+Gary)\.?\s*:?\s*(?:No\.?\s*)?(?:Gary\s+)?"
    r"((?:\d[\d\-]{1,7}(?:-?[A-Z])?)(?:\s*(?:and|&|,|or)\s*\d[\d\-]{1,7}(?:-?[A-Z])?)*)",
    re.IGNORECASE)

# a 1950s exchange-name number ("TU 2-1341", "WEstmor 2-1975") somewhere in the text
EXCHANGE_HINT_RE = re.compile(r"\b[A-Z][A-Za-z]{0,7}\s?\d-\d{4}\b")

SEE_RE = re.compile(r"\(\s*See\s+([^)]*)\)?", re.IGNORECASE)
PAREN_RE = re.compile(r"\(([^)]*)\)")


@lru_cache(maxsize=1)
def abbreviations() -> dict[str, str]:
    try:
        txt = resources.files("polkadot").joinpath("data/abbreviations.json").read_text()
        data = json.loads(txt)
    except Exception:  # pragma: no cover
        return {}
    data.pop("_comment", None)
    return data


def expand_abbrev(phrase: str) -> str:
    """Expand known abbreviations ("mach I S Co" -> "machinist Illinois Steel Co").

    Multi-word keys are replaced first; single tokens are only expanded when
    the key matches exactly (so the capital ``S`` in ``I S Co`` is never
    read as "south")."""
    ab = abbreviations()
    if phrase in ab:
        return ab[phrase]
    multi = sorted((k for k in ab if " " in k), key=len, reverse=True)
    placeholders: dict[str, str] = {}
    for i, k in enumerate(multi):
        pat = r"(?<![\w])" + re.escape(k) + r"(?![\w])"
        if re.search(pat, phrase):
            ph = f"\x00{i}\x00"
            placeholders[ph] = ab[k]
            phrase = re.sub(pat, ph, phrase)
    out = []
    for tok in phrase.split():
        core = tok.strip(",.;:")
        if core in ab and len(core) <= 10:
            out.append(tok.replace(core, ab[core]))
        else:
            out.append(tok)
    res = " ".join(out)
    for ph, v in placeholders.items():
        res = res.replace(ph, v)
    return res


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def clean_text(t: str) -> str:
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def find_phones(text: str) -> list[str]:
    out = [p for p in find_exchange_phones(text)] if EXCHANGE_HINT_RE.search(text) else []
    for m in PHONE_RE.finditer(text):
        for p in re.split(r"\s*(?:and|&|,|or)\s*", m.group(1)):
            p = p.strip(" -")
            if p and p not in out and sum(c.isdigit() for c in p) >= 2:
                out.append(p)
    return out


INTERSECTION_RE = re.compile(
    r"\b\d{1,3}(?:st|nd|rd|th|d)\s+" + STREET_SUFFIX + r"\s+(?:and|&)\s+[A-Z][A-Za-z]+(?:\s+" + STREET_SUFFIX + r")?")


def find_addresses(text: str) -> list[str]:
    out: list[str] = []
    for m in INTERSECTION_RE.finditer(text):
        a = m.group(0).strip(" ,.")
        if a not in out:
            out.append(a)
    for m in CORNER_RE.finditer(text):
        a = m.group(0).strip(" ,.")
        if a not in out:
            out.append(a)
    t = PHONE_RE.sub(" ", text)
    for m in ADDRESS_RE.finditer(t):
        a = m.group(0).strip(" ,.")
        # discard bare "1922 Polk" style years and phone-like numbers
        num = re.match(r"\D*(\d+)", a)
        if not num:
            continue
        rest = a[num.end():].strip()
        if not rest or rest.lower() in {"and", "to", "the", "a", "m", "p"}:
            continue
        if re.match(r"^(?:19|18)\d\d$", num.group(1)) and rest[:1].isupper() and len(rest.split()) <= 1 and "Polk" in rest:
            continue
        if a not in out:
            out.append(a)
    return out


UNAMBIGUOUS_BUSINESS = {"co", "company", "corp", "corporation", "inc", "bros", "assn",
                        "apts", "mfg", "ry", "incorporated", "ltd"}


def _titled(name: str) -> str:
    """Title-case ALL-CAPS words so keyword tests work on advertiser caps."""
    return " ".join(w.capitalize() if w.isupper() and len(w) > 1 else w for w in name.split())


def is_business_name(name: str, person_order: bool = False) -> bool:
    """Does ``name`` look like a firm / institution rather than a person?

    ``person_order``: the text is from a listing printed "Surname Given",
    where a keyword in first position is most likely a surname (Hall,
    Mills, Church ...) and does not count unless unambiguous.
    """
    t = _titled(name)
    toks = [w.strip(".,").lower() for w in t.split()]
    if not toks:
        return False
    if any(w in UNAMBIGUOUS_BUSINESS for w in toks):
        return True
    for m in STRONG_BUSINESS_RE.finditer(t):
        if person_order and t[: m.start()].strip() == "":
            continue
        return True
    if "&" in name and len(toks) >= 2:
        return True
    return False


def is_caps_name(name: str) -> bool:
    letters = [c for c in name if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.85


def _strip_see(text: str) -> tuple[str, list[str]]:
    refs = [m.group(1).strip() for m in SEE_RE.finditer(text)]
    return SEE_RE.sub("", text).strip(), refs


def _norm_tok(t: str) -> str:
    return t.strip(",.;:")


# --------------------------------------------------------------------------
# Alphabetical (general) directory
# --------------------------------------------------------------------------


def parse_alphabetical(text: str, surname_ctx: Optional[str] = None) -> dict[str, Any]:
    """Parse one alphabetical-section listing.

    Returns a dict with ``kind`` ("person" | "business" | "proprietor") and
    whatever fields could be identified.
    """
    text = clean_text(text)
    body, refs = _strip_see(text)
    toks = body.split()
    f: dict[str, Any] = {}
    if refs:
        f["see_also"] = refs
    if not toks:
        return {"kind": "other", "fields": f}

    # ditto mark -> repeat the surname in force
    if re.match(r"""^(?:"|''|,,|“|”|‘|’|'|`|„)$""", toks[0]) and surname_ctx:
        toks[0] = surname_ctx
        f["ditto"] = True
    elif re.match(r"""^["'`‘’“”„]+[A-Z]""", toks[0]) and surname_ctx:
        toks = [surname_ctx, toks[0].lstrip("\"'`‘’“”„")] + toks[1:]
        f["ditto"] = True

    # ---- residence marker (last h/r/bds that is followed by an address) --
    res_idx = None
    for i in range(len(toks) - 1, 0, -1):
        if _norm_tok(toks[i]) in RES_MARKERS and i + 1 < len(toks):
            nxt = toks[i + 1]
            if nxt[:1].isdigit() or nxt.lower().startswith(("same", "do", "rear", "ne", "nw", "se", "sw", "es", "ws", "ns", "ss")) \
                    or (i + 2 < len(toks) and toks[i + 2].lower() in {"cor", "s", "n", "e", "w"}):
                res_idx = i
                break
    head = toks if res_idx is None else toks[:res_idx]
    if res_idx is not None:
        mk = _norm_tok(toks[res_idx])
        f["residence_type"] = {"h": "householder", "r": "resides", "b": "boards",
                               "bds": "boards", "rms": "rooms", "res": "resides"}.get(mk, mk)
        f["residence_address"] = " ".join(toks[res_idx + 1:]).strip(" ,.")

    # ---- business test on the leading capitalised run ---------------------
    lead = []
    for t in head:
        if t[:1].isdigit() or (t[:1].islower() and t not in {"and", "of", "&"}) or t.startswith("("):
            break
        lead.append(t)
    lead_str = " ".join(lead)
    if lead and len(lead) >= 2 and is_business_name(lead_str, person_order=True):
        name = lead_str.strip(" ,")
        rest = " ".join(head[len(lead):])
        f["name"] = name
        addrs = find_addresses(rest) or find_addresses(body)
        if addrs:
            f["address"] = addrs[0]
        m = PAREN_RE.search(rest)
        if m:
            f["note"] = m.group(1)
        desc = re.sub(r"\([^)]*\)", "", rest)
        if addrs:
            desc = desc.replace(addrs[0], "")
        desc = desc.strip(" ,.")
        if desc:
            f["description"] = desc
            f["description_expanded"] = expand_abbrev(desc)
        phones = find_phones(body)
        if phones:
            f["phones"] = phones
        return {"kind": "business", "fields": f, "advertiser": is_caps_name(name)}

    # ---- person ------------------------------------------------------------
    i = 0
    surname = _norm_tok(head[0]) if head else ""
    i = 1
    given = []
    while i < len(head):
        t = head[i]
        tt = _norm_tok(t)
        if t.startswith("(") or not tt or tt[:1].isdigit() or tt[:1].islower():
            break
        # "I S Co" style employer right after the name is unusual; stop on
        # a capitalised word that starts a known business tail
        if tt in {"Mrs", "Miss", "Dr", "Rev", "Jr", "Sr"}:
            given.append(tt)
            i += 1
            continue
        if len(given) >= 3:
            break
        given.append(tt)
        i += 1
    f["surname"] = surname
    if given:
        f["given_name"] = " ".join(given)
    f["name"] = " ".join([surname] + given)

    if i < len(head) and head[i].startswith("("):
        j = i
        buf = []
        while j < len(head):
            buf.append(head[j])
            if head[j].endswith(")"):
                break
            j += 1
        sp = " ".join(buf).strip("()")
        if sp.startswith("wid "):
            f["widow_of"] = sp[4:].strip()
        elif sp:
            f["spouse"] = sp
        i = j + 1

    rest = head[i:]
    occ = []
    k = 0
    while k < len(rest) and (rest[k][:1].islower() or rest[k] in {"&"}) and not rest[k][:1].isdigit():
        occ.append(rest[k])
        k += 1
    tail = rest[k:]
    if occ:
        o = " ".join(occ).strip(" ,")
        f["occupation"] = o
        f["occupation_expanded"] = expand_abbrev(o)
    tail_str = " ".join(tail).strip(" ,")
    kind = "person"
    if tail_str:
        addrs = find_addresses(tail_str)
        if addrs and tail_str.startswith(addrs[0].split()[0]):
            f["business_address"] = addrs[0]
            after = tail_str[len(addrs[0]):].strip(" ,")
            if after:
                f["employer"] = after
            kind = "proprietor"
        elif addrs:
            emp = tail_str[: tail_str.find(addrs[0])].strip(" ,")
            if emp:
                f["employer"] = emp
                f["employer_expanded"] = expand_abbrev(emp)
            f["business_address"] = addrs[0]
            if not emp:
                kind = "proprietor"
        else:
            f["employer"] = tail_str
            f["employer_expanded"] = expand_abbrev(tail_str)
    if kind == "person" and f.get("occupation", "").lower() in PROPRIETOR_OCCUPATIONS \
            and "employer" not in f:
        # a grocer / physician / tailor ... with no employer runs their own
        # business (at the residence when no business address is given)
        kind = "proprietor"
    phones = find_phones(body)
    if phones:
        f["phones"] = phones
    return {"kind": kind, "fields": f, "advertiser": is_caps_name(f.get("name", ""))}


# --------------------------------------------------------------------------
# Street guide
# --------------------------------------------------------------------------

STREET_HEAD_RE = re.compile(
    r"^(?P<street>[A-Z][A-Z0-9 .'’]+?)(?:\s*\((?P<alias>[A-Z])\))?\s*(?:,|—|-{1,2})\s*(?P<desc>.*)$")
CONTD_RE = re.compile(r"^(?P<street>[A-Z][A-Z0-9 .'’]+?)\s*[—\-]+\s*Cont(?:d|inued)\.?\s*$", re.IGNORECASE)


def parse_street_heading(text: str) -> Optional[dict[str, Any]]:
    t = clean_text(text)
    m = CONTD_RE.match(t)
    if m:
        return {"street": m.group("street").title().strip(), "continued": True}
    m = STREET_HEAD_RE.match(t)
    if m and len(m.group("street")) >= 3:
        letters = [c for c in m.group("street") if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
            return {"street": m.group("street").title().strip(), "description": m.group("desc").strip()}
    return None


def parse_street_line(text: str, street: Optional[str], last_number: Optional[str]) -> dict[str, Any]:
    t = clean_text(text)
    t = re.sub(r"^[^\w\"]+", "", t)                  # rule fragments, specks
    t = re.sub(r"^[a-z]\|\s*", "", t)
    # OCR splits house numbers with stray punctuation: "4,32" -> "432"
    t = re.sub(r"^(\d{1,3})[,.'](\d{1,3})\b", r"\1\2", t)
    f: dict[str, Any] = {}
    unit = re.match(r"^(?P<unit>[A-Z]-\d{1,3})\s+(?P<rest>.*)$", t)
    if unit:
        # apartment unit inside the building whose number is in force
        f["unit"] = unit.group("unit")
        t = unit.group("rest")
        if last_number:
            f["house_number"] = last_number
        if street and last_number:
            f["street"] = street
            f["address"] = f"{last_number} {street} Apt {f['unit']}"
        rest = t.strip()
        f["occupant"] = rest
        if rest.lower().startswith("vacant"):
            f["vacant"] = True
            return {"kind": "other", "fields": f}
        if is_business_name(rest.split(",")[0], person_order=True):
            f["name"] = rest.split(",")[0].strip()
            return {"kind": "business", "fields": f, "advertiser": is_caps_name(rest)}
        f["name"] = rest
        return {"kind": "person", "fields": f, "advertiser": False}
    m = re.match(r"^(?P<num>\d{1,5}(?:\s?1/2|½)?(?:-\d{1,5})?[A-Za-z]?)\s+(?P<rest>.*)$", t)
    if m:
        num, rest = m.group("num"), m.group("rest")
    else:
        m2 = re.match(r"^(?P<loc>(?:rear|[nsew][es]?\s+(?:cor|\d+\s+[nsew])|[nsew]s\s+\d?\s*[nsew]?)\b[^A-Z]*?)\s+(?P<rest>[A-Z].*)$", t)
        if m2:
            num, rest = m2.group("loc").strip(), m2.group("rest")
            f["location"] = num
        else:
            num, rest = None, t
            if last_number:
                f["same_number_as_previous"] = True
    number = num or last_number
    if number:
        f["house_number"] = number
    if street:
        f["street"] = street
        if number and not f.get("location"):
            f["address"] = f"{number} {street}"
        elif f.get("location"):
            f["address"] = f"{f['location']} {street}"
    rest = rest.strip()
    f["occupant"] = rest
    if rest.lower().startswith("vacant"):
        f["vacant"] = True
        return {"kind": "other", "fields": f}
    name_part = rest.split(",")[0]
    if is_business_name(name_part, person_order=True):
        f["name"] = name_part.strip()
        return {"kind": "business", "fields": f, "advertiser": is_caps_name(name_part)}
    # "Williams Charles J, real est" -> person with a trade
    if "," in rest:
        occ = rest.split(",", 1)[1].strip()
        if occ:
            f["occupation"] = occ
            f["occupation_expanded"] = expand_abbrev(occ)
            f["name"] = name_part.strip()
            return {"kind": "proprietor", "fields": f, "advertiser": is_caps_name(name_part)}
    f["name"] = rest
    return {"kind": "person", "fields": f, "advertiser": False}


# --------------------------------------------------------------------------
# Classified business directory / miscellaneous lists
# --------------------------------------------------------------------------


def clean_heading(text: str) -> str:
    t = clean_text(text).lstrip("*•\"'`▶►> ").strip()
    t = re.sub(r"\s*\(cont'?d\)\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*[—\-]+\s*Cont(?:d|inued)\.?$", "", t, flags=re.IGNORECASE)
    return t


def parse_classified(text: str, category: Optional[str]) -> dict[str, Any]:
    t = clean_text(text)
    phones_pb = find_exchange_phones(t)
    if phones_pb:
        # telephone yellow pages: "NAME address------TU 3-9425"
        t = EXCHANGE_PHONE_RE.sub(" ", t)
        t = LEADER_RE.sub(" ", t).strip(" -~")
    body, refs = _strip_see(t)
    f: dict[str, Any] = {}
    if category:
        f["category"] = category
    if refs:
        f["see_also"] = refs
        f["ad_reference"] = "; ".join(refs)
    # "NAME, address" (advertisers) or "Name address"
    name, addr = None, None
    m = ADDRESS_RE.search(body) or CORNER_RE.search(body)
    if m and m.start() > 0:
        name, addr = body[: m.start()].strip(" ,"), body[m.start():].strip()
    elif "," in body:
        a, b = body.split(",", 1)
        name, addr = a.strip(), b.strip()
    else:
        name = body
    paren = PAREN_RE.search(name or "")
    if paren:
        f["note"] = paren.group(1)
        name = PAREN_RE.sub("", name).strip()
    f["name"] = (name or "").strip(" ,.")
    if addr:
        f["address"] = addr.strip(" ,.")
    phones = phones_pb or find_phones(t)
    if phones:
        f["phones"] = phones
    return {"kind": "business", "fields": f, "advertiser": is_caps_name(f["name"]) or bool(refs)}


def parse_misc(text: str, category: Optional[str]) -> dict[str, Any]:
    """Miscellaneous-information lists: churches, lodges, schools, ...

    Typical form: ``Name—address, Rev X pastor. Services ...``"""
    t = clean_text(text)
    f: dict[str, Any] = {}
    if category:
        f["category"] = category
    m = re.match(r"^(?P<name>[^—]{2,80}?)\s*—\s*(?P<rest>.*)$", t)
    if not m:
        m = re.match(r"^(?P<name>[^—]{2,80}?)\s+-{1,2}\s*(?P<rest>.*)$", t)
    if m:
        f["name"] = m.group("name").strip(" ,.")
        rest = m.group("rest")
        addrs = find_addresses(rest)
        if addrs:
            f["address"] = addrs[0]
        else:
            first = rest.split(",")[0].strip()
            if first and len(first) < 60:
                f["address"] = first
        people = re.findall(r"\b(?:Rev|Dr|Mrs|Miss)\.?\s+[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){0,3}", rest)
        if people:
            f["people"] = people
        phones = find_phones(t)
        if phones:
            f["phones"] = phones
        return {"kind": "business", "fields": f, "advertiser": is_caps_name(f["name"])}
    return {"kind": "other", "fields": f, "advertiser": False}


# --------------------------------------------------------------------------
# Advertisements
# --------------------------------------------------------------------------

SLOGAN_WORDS = {"the", "a", "good", "place", "to", "eat", "phone", "telephone", "tel",
                "and", "of", "for", "with", "service", "courtesy", "we", "do", "it",
                "right", "our", "your", "best", "all", "work", "guaranteed"}


def ad_fields(lines: list[tuple], full_text: str) -> dict[str, Any]:
    """Pick the advertiser's name and contact details out of an ad.

    ``lines`` are (text, type-height) pairs in reading order.  The business
    name is normally the largest type in the ad; runs of single huge letters
    stacked vertically (``D / E / L / U / X / E``) are re-assembled.
    """
    f: dict[str, Any] = {}
    t = clean_text(full_text)
    f["phones"] = find_phones(t)
    f["addresses"] = find_addresses(t)
    people = re.findall(r"\b(?:[A-Z]\.\s?){1,2}[A-Z][a-zA-Z]+\b|\b(?:Dr|Rev|Mrs|Miss|Mr)\.?\s+[A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+){0,2}", t)
    if people:
        f["people"] = list(dict.fromkeys(people))

    # lines: (text, height[, y0, y1[, x0[, conf]]]) in reading order
    cands = []
    for ln in lines:
        s, h = clean_text(ln[0]), float(ln[1])
        y0, y1 = (float(ln[2]), float(ln[3])) if len(ln) >= 4 else (None, None)
        conf = float(ln[5]) if len(ln) >= 6 else -1.0
        # drop OCR garbage lines - but keep short ones, which may be
        # poorly-read letters of a stacked display word
        if s and (conf < 0 or conf >= 45 or len(s) <= 2):
            cands.append((s, h, y0, y1))
    if not cands:
        return f

    def flush(buf: list) -> list[tuple[str, float]]:
        if len(buf) < 3:
            return [(c, h) for c, h, _, _ in buf]
        # gaps between stacked letters; a clearly larger gap is a word break
        gaps = [b[2] - a[3] for a, b in zip(buf, buf[1:])
                if a[3] is not None and b[2] is not None]
        word = buf[0][0]
        med = sorted(gaps)[len(gaps) // 2] if gaps else None
        for i, (c, _, _, _) in enumerate(buf[1:]):
            g = gaps[i] if gaps and i < len(gaps) else None
            word += (" " if med is not None and g is not None and g > med * 1.8 + 2 else "") + c
        # stacked display letters are capitals; OCR confuses O/0 and L/i
        if sum(ch.isupper() for ch in word) >= sum(ch.islower() for ch in word):
            word = word.upper().replace("0", "O")
        return [(word, max(h for _, h, _, _ in buf))]

    # re-assemble stacked letters: consecutive one/two-character lines
    merged: list[tuple[str, float]] = []
    buf: list = []
    for s, h, y0, y1 in cands:
        core = re.sub(r"[^A-Za-z0-9&]", "", s)
        if 0 < len(core) <= 2 and (core.isalpha() or core in {"0", "0s", "5"}) and len(s) <= 4:
            buf.append((core, h, y0, y1))
            continue
        if buf and len(s) <= 2 and not core and h >= 0.7 * buf[-1][1]:
            continue      # an unreadable letter inside a stacked word: skip it
        merged.extend(flush(buf))
        buf = []
        merged.append((s, h))
    merged.extend(flush(buf))

    def informative(s: str) -> bool:
        letters = sum(c.isalpha() for c in s)
        if letters < 3:
            return False
        if find_phones(s) and letters < 12:
            return False
        if re.fullmatch(r"[\d\s,.\-]+[A-Za-z .]*", s) and find_addresses(s):
            return False
        words = [w.lower().strip(".,!\"'") for w in s.split()]
        return not all(w in SLOGAN_WORDS for w in words)

    good = [(s, h) for s, h in merged if informative(s)]
    if not good:
        good = merged
    hmax = max(h for _, h in good)
    top = [(s, h) for s, h in good if h >= hmax * 0.8]
    name = " ".join(s for s, _ in top[:3])
    # prefer a line that looks like a firm name if the biggest type is a slogan
    firm = [s for s, h in good if is_business_name(s) and h >= hmax * 0.5]
    if firm and not is_business_name(name):
        name = firm[0]
    f["business_name"] = name.strip(" ,.-—")
    return f


# --------------------------------------------------------------------------
# Telephone directories (white pages)
# --------------------------------------------------------------------------

# "TUrnr 4-8040", "WEstmor 2-1975", "TU 2-1341", "GRnit 4-2440", "Whiting 2585 W"
EXCHANGE_PHONE_RE = re.compile(
    r"(?P<exch>\b[A-Z][A-Za-z]{1,8})\.?\s*(?P<num>\d\s?[-–.]\s?\d{4})(?:\s*(?P<party>[A-Z])\b)?")
LEADER_RE = re.compile(r"\s*[-–—~_.·]{3,}\s*")

TOWNS = {
    "gary": "Gary", "hamnd": "Hammond", "hammond": "Hammond", "echgo": "East Chicago",
    "e chgo": "East Chicago", "calcty": "Calumet City", "lansng": "Lansing",
    "highld": "Highland", "munstr": "Munster", "grifth": "Griffith", "whting": "Whiting",
    "whiting": "Whiting", "burnhm": "Burnham", "dyer": "Dyer", "ross": "Ross",
    "schrvl": "Schererville", "sherer": "Schererville", "newelt": "New Elliott",
    "mrlvl": "Merrillville", "chgo": "Chicago", "hobart": "Hobart", "crwnpt": "Crown Point",
}


def find_exchange_phones(text: str) -> list[str]:
    out = []
    for m in EXCHANGE_PHONE_RE.finditer(text):
        num = re.sub(r"\s", "", m.group("num")).replace("–", "-").replace(".", "-")
        p = f"{m.group('exch')} {num}" + (f" {m.group('party')}" if m.group("party") else "")
        if p not in out:
            out.append(p)
    return out


def _split_house_number(addr: str) -> str:
    """'3111W45thAv' -> '3111 W45thAv' (phone books glue number and street)."""
    return re.sub(r"^(\d+(?:\s?1/2)?)(?!st|nd|rd|th|d[A-Z]|d\b)(?=[A-Za-z])", r"\1 ", addr.strip())


def parse_telephone(text: str, surname_ctx: Optional[str] = None) -> dict[str, Any]:
    """``Hagen John B 3440Pierce Gary--------TUrnr 4-3265``.

    Fields: name, occupation (when given, e.g. ``arch``, ``DDS``), address,
    town, phone(s).  Lines in capitals are usually firms; ``X also see Y``
    cross-references are returned as kind "other"."""
    t = clean_text(text)
    t = re.sub(r"(?<=[A-Za-z])[!|]", "l", t)            # OCR: "TEmp!" -> "TEmpl"
    t = re.sub(r"\b(Whi?tn?g)\s*-\s*(\d{3,4})(?:-([A-Z]))?", lambda m: "Whiting " + m.group(2) + (" " + m.group(3) if m.group(3) else ""), t)
    # stray characters bleeding in from the page edge / gutter
    t = re.sub(r"^(?:[^A-Za-z\"“”']+\s*|[a-z§}{|]\s+)+(?=[A-Z])", "", t)
    f: dict[str, Any] = {}
    if re.search(r"\balso\s+see\b", t, re.IGNORECASE):
        f["see_also"] = re.split(r"\balso\s+see\b", t, flags=re.IGNORECASE)[1].strip()
        f["name"] = re.split(r"\balso\s+see\b", t, flags=re.IGNORECASE)[0].strip()
        return {"kind": "other", "fields": f, "advertiser": False}
    phones = find_exchange_phones(t)
    m = EXCHANGE_PHONE_RE.search(t)
    if not phones:
        # small-town exchanges: "------Whiting 2585 W"
        m = re.search(r"(?P<exch>\b[A-Z][a-z]+)\s+(?P<num>\d{3,4})(?:\s*-?\s*(?P<party>[A-Z]))?\s*$", t)
        if m and LEADER_RE.search(t[: m.start()] + "---"):
            phones = [" ".join(x for x in (m.group("exch"), m.group("num"), m.group("party")) if x)]
        else:
            m = None
    if phones:
        f["phones"] = phones
    # everything before the first phone number (and its leader dots)
    head = t[: m.start()] if m else t
    head = LEADER_RE.split(head)[0].strip(" -–~")
    # town: the last word, when it is a known town abbreviation
    words = head.split()
    if words:
        key = words[-1].lower().strip(".,")
        two = " ".join(words[-2:]).lower() if len(words) >= 2 else ""
        if two in TOWNS:
            f["town"] = TOWNS[two]
            words = words[:-2]
        elif key in TOWNS:
            f["town"] = TOWNS[key]
            words = words[:-1]
    # a town glued to the street ("7537ChestnutHamnd")
    if words and "town" not in f:
        for key, town in TOWNS.items():
            k = key.replace(" ", "")
            last = words[-1]
            if len(last) > len(k) + 2 and last.lower().endswith(k) and last[-len(k)].isupper():
                words[-1] = last[: -len(k)]
                f["town"] = town
                break
    # address: from the first token that starts with a digit (or "rear")
    ai = next((i for i, w in enumerate(words) if i > 0 and re.match(r"^\d", w)), None)
    if ai is not None:
        f["address"] = _split_house_number(" ".join(words[ai:]))
        name_words = words[:ai]
    else:
        name_words = words
    if surname_ctx and name_words and re.match(r"""^["“”']$""", name_words[0]):
        name_words = [surname_ctx] + name_words[1:]
    name = " ".join(name_words).strip(" ,")
    # trailing lower-case words before the address are an occupation ("arch", "phys")
    occ = []
    while name_words and (name_words[-1].islower() or name_words[-1] in {"DDS", "MD", "DC", "DO"}) \
            and len(name_words) > 2:
        occ.insert(0, name_words.pop())
    if occ:
        f["occupation"] = " ".join(occ)
        f["occupation_expanded"] = expand_abbrev(f["occupation"])
        name = " ".join(name_words)
    f["name"] = name
    toks = name.split()
    if toks and not is_caps_name(name):
        f["surname"] = toks[0]
        if len(toks) > 1:
            f["given_name"] = " ".join(toks[1:])
    if is_business_name(name, person_order=not is_caps_name(name)) or is_caps_name(name):
        kind = "business"
    elif occ:
        kind = "proprietor"
    else:
        kind = "person"
    return {"kind": kind, "fields": f, "advertiser": is_caps_name(name)}
