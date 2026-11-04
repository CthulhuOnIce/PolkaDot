"""Document-level clean-up after all pages are parsed.

* Repeated advertisements: Polk rotated the same strip ads through the
  book, so one advertiser's ad may appear on dozens of pages.  Ads are
  grouped so each advertiser is reported once with all its occurrences.
* Ads are linked to the listings of the same business (classified
  entries such as ``BROADWAY FLORIST, 519 Broadway (See right bottom
  lines)`` and bold alphabetical entries), by name and phone/address.
"""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Iterable

from .models import Ad, Entry, EntryKind

STOP = {"the", "and", "co", "company", "inc", "&", "of", "a", "an", "bros", "corp",
        "gary", "ind", "indiana", "phone", "tel", "telephone"}


def norm_name(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_tokens(s: str) -> set[str]:
    return {t for t in norm_name(s).split() if t not in STOP and len(t) > 1}


def similar(a: str, b: str) -> float:
    a, b = norm_name(a), norm_name(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def group_ads(ads: list[Ad], threshold: float = 0.82) -> None:
    """Assign ``meta['group']`` / ``meta['occurrences']`` to repeated ads."""
    groups: list[list[Ad]] = []
    keys: list[str] = []
    for ad in ads:
        key = norm_name(ad.text)[:240]
        if len(key) < 6:
            groups.append([ad])
            keys.append(key)
            continue
        hit = None
        for i, k in enumerate(keys):
            if len(k) < 6 or abs(len(k) - len(key)) > max(len(k), len(key)) * 0.4:
                continue
            r = SequenceMatcher(None, k, key).quick_ratio()
            if r < threshold * 0.8:
                continue
            r = SequenceMatcher(None, k, key).ratio()
            # a shared phone number only helps when the names agree too
            # (one number can serve several firms in the same building)
            same_phone = bool(set(ad.phones) & set(groups[i][0].phones))
            names_agree = similar(ad.business_name or "", groups[i][0].business_name or "") >= 0.75
            if r >= threshold or (same_phone and names_agree and r >= threshold * 0.8):
                hit = i
                break
        if hit is None:
            groups.append([ad])
            keys.append(key)
        else:
            groups[hit].append(ad)
    for gi, g in enumerate(groups):
        # the best-read instance names the group
        best = max(g, key=lambda a: (a.meta.get("text_quality") or 0) * len(a.text))
        for ad in g:
            ad.meta["group"] = f"adgroup{gi}"
            ad.meta["occurrences"] = len(g)
            ad.meta["group_pages"] = sorted({a.page + 1 for a in g})
            if not ad.business_name and best.business_name:
                ad.meta["group_business_name"] = best.business_name


def link_ads_to_entries(ads: list[Ad], entries: Iterable[Entry]) -> None:
    """Cross-reference ads and listings that name the same business."""
    index: dict[str, list[Entry]] = defaultdict(list)
    candidates = [e for e in entries if e.kind in (EntryKind.BUSINESS, EntryKind.PROPRIETOR)
                  or e.advertiser]
    for e in candidates:
        for t in name_tokens(e.fields.get("name", "")):
            index[t].append(e)
    phones: dict[str, list[Entry]] = defaultdict(list)
    for e in candidates:
        for p in e.fields.get("phones", []) or []:
            phones[p].append(e)
    for ad in ads:
        name = ad.business_name or ad.meta.get("group_business_name") or ""
        toks = name_tokens(name)
        hits: dict[str, tuple[float, Entry]] = {}
        pool: dict[str, Entry] = {}
        for t in toks:
            for e in index.get(t, []):
                pool[e.id] = e
        for e in pool.values():
            en = e.fields.get("name", "")
            et = name_tokens(en)
            if not et:
                continue
            overlap = len(toks & et) / max(1, min(len(toks), len(et)))
            score = max(overlap if len(toks & et) >= min(2, len(et)) else 0.0, similar(name, en))
            if score >= 0.8:
                hits[e.id] = (score, e)
        for p in ad.phones:
            for e in phones.get(p, []):
                hits.setdefault(e.id, (0.75, e))
        linked = sorted(hits.values(), key=lambda x: -x[0])[:25]
        if linked:
            ad.meta["linked_entries"] = [e.id for _, e in linked]
            for _, e in linked:
                e.fields.setdefault("ad_ids", [])
                if ad.id not in e.fields["ad_ids"]:
                    e.fields["ad_ids"].append(ad.id)


# --------------------------------------------------------------------------
# Consolidated business index
# --------------------------------------------------------------------------


def _addr_key(a: str) -> str:
    a = norm_name(a or "")
    m = re.match(r"(\d+)\s+(\w+)", a)
    return f"{m.group(1)} {m.group(2)}" if m else ""


def business_key(name: str, kind: str, address: str = "") -> str:
    toks = sorted(name_tokens(name))
    if not toks:
        return ""
    key = " ".join(toks)
    if kind == "proprietor":
        # people share names; a proprietor is identified by name + address
        key += " @ " + _addr_key(address)
    return key


def consolidate(entries: Iterable[Entry], ads: list[Ad]) -> list[dict]:
    """One record per distinct business across sections, pages and ads."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    items: dict[str, list] = defaultdict(list)
    ent_key: dict[str, str] = {}
    for e in entries:
        if e.kind not in (EntryKind.BUSINESS, EntryKind.PROPRIETOR) and not e.advertiser:
            continue
        name = e.fields.get("name") or ""
        addr = e.fields.get("address") or e.fields.get("business_address") or ""
        k = business_key(name, e.kind.value, addr)
        if not k:
            continue
        find(k)
        ent_key[e.id] = k
        items[k].append(("entry", e))
    for ad in ads:
        name = ad.business_name or ad.meta.get("group_business_name") or ""
        k = business_key(name, "business")
        linked = [ent_key[i] for i in ad.meta.get("linked_entries", []) if i in ent_key]
        if not k and not linked:
            k = "ad " + (ad.meta.get("group") or ad.id)
        k = k or linked[0]
        find(k)
        for lk in linked[:3]:
            union(lk, k)
        if ad.meta.get("group"):
            union("group " + ad.meta["group"], k)
        items[k].append(("ad", ad))

    groups: dict[str, list] = defaultdict(list)
    for k, its in items.items():
        groups[find(k)].extend(its)

    out = []
    for gi, (k, its) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        names, cats, addrs, phones, pages, secs, kinds, crops = (
            [], [], [], [], set(), set(), set(), [])
        entry_ids, ad_ids = [], []
        for typ, obj in its:
            if typ == "entry":
                f = obj.fields
                names.append(f.get("name") or "")
                if f.get("category"):
                    cats.append(f["category"])
                for a in (f.get("address"), f.get("business_address")):
                    if a:
                        addrs.append(a)
                phones += f.get("phones", []) or []
                pages.add(obj.page + 1)
                secs.add(obj.section)
                kinds.add(obj.kind.value)
                entry_ids.append(obj.id)
                if obj.crop:
                    crops.append(obj.crop)
            else:
                if obj.business_name:
                    names.append(obj.business_name)
                if obj.category:
                    cats.append(obj.category)
                addrs += obj.addresses
                phones += obj.phones
                pages.add(obj.page + 1)
                kinds.add("ad")
                ad_ids.append(obj.id)
                if obj.crop:
                    crops.append(obj.crop)

        def ranked(xs: list[str]) -> list[str]:
            c: dict[str, int] = defaultdict(int)
            for x in xs:
                c[x.strip(" ,.")] += 1
            return [x for x, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])) if x]

        nm = ranked(names)
        # prefer a mixed-case reading for display when one exists
        display = next((n for n in nm if not n.isupper()), nm[0] if nm else "")
        out.append({
            "business_id": f"B{gi:05d}",
            "name": display,
            "name_variants": nm[:6],
            "kinds": sorted(kinds),
            "categories": ranked(cats)[:8],
            "addresses": ranked(addrs)[:6],
            "phones": ranked(phones)[:6],
            "sections": sorted(secs),
            "pages": sorted(pages),
            "n_listings": len(entry_ids),
            "n_ads": len(ad_ids),
            "entry_ids": entry_ids,
            "ad_ids": ad_ids,
            "crops": crops[:12],
        })
    return out
