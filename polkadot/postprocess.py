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
            same_phone = set(ad.phones) & set(groups[i][0].phones)
            r = SequenceMatcher(None, k, key).quick_ratio()
            if r < threshold * 0.9 and not same_phone:
                continue
            r = SequenceMatcher(None, k, key).ratio()
            if r >= threshold or (same_phone and r >= threshold * 0.7):
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
