"""Free keyword prefilter over title + description. Tuned for recall; ranks candidates per topic."""

from __future__ import annotations

import re

from lawtracker.topics import KEYWORDS

VERSION = 2  # v2: Florida H/S numbering
# Bills and joint resolutions (constitutional amendments). Simple/concurrent resolutions are mostly ceremonial.
BILL_TYPES = re.compile(r"^(?:HB|SB|HJR|SJR|AB|ACA|SCA)\d+$")
FEDERAL_TYPES = re.compile(r"^(?:HR|S|HJRES|SJRES)\d+$")  # federal HR is a bill; state HR is a resolution
# States whose LegiScan numbering differs from the default (FL numbers bills and joint resolutions H1234/S1234).
# NY: Assembly/Senate bills A1234/S1234 (resolutions are J/K/R/E). VA: HB/SB plus HJ/SJ joint resolutions, which
# carry constitutional amendments (HJR/SJR spelling is not used there).
STATE_TYPES = {
    "US": FEDERAL_TYPES,
    "FL": re.compile(r"^[HS]\d+$"),
    "NY": re.compile(r"^[AS]\d+$"),
    "VA": re.compile(r"^(?:HB|SB|HJ|SJ)\d+$"),
}
_PATTERNS = {t: [re.compile(p, re.I) for p in ps] for t, ps in KEYWORDS.items()}


def score(text: str) -> dict[str, int]:
    """Number of distinct patterns that match, per topic (topics with 0 are omitted)."""
    out = {}
    for topic, pats in _PATTERNS.items():
        n = sum(1 for p in pats if p.search(text))
        if n:
            out[topic] = n
    return out


def hits(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of every keyword match in `text`, any topic."""
    spans = []
    for pats in _PATTERNS.values():
        for p in pats:
            spans.extend(m.span() for m in p.finditer(text))
    return sorted(spans)


_REPEAL = re.compile(r"(?i)\b(?:is|are) (?:hereby )?repealed\b")


def windows(text: str, radius: int = 1500, max_chars: int = 24000) -> str:
    """Merged ±radius windows around keyword hits; the head of the text if there are none."""
    spans = hits(text)
    if not spans:
        return text[:max_chars]
    # Repeal clauses ("... is repealed", "are repealed:") can change a topic without naming it: keep them too.
    spans = sorted(spans + [(m.start(), m.end()) for m in _REPEAL.finditer(text)])
    merged: list[list[int]] = []
    for s, e in spans:
        s, e = max(0, s - radius), min(len(text), e + radius)
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, total = [], 0
    head = text[:1200]  # caption and enacting clause give context
    if merged[0][0] > 0:
        out.append(head)
        total += len(head)
    for s, e in merged:
        if total >= max_chars:
            break
        chunk = text[s:e][: max_chars - total]
        out.append(chunk)
        total += len(chunk)
    return "\n[...]\n".join(out)


def run(db) -> dict:
    """(Re)score bills whose prefilter_version is stale. Candidate = bill type matches and any topic scores."""
    rows = list(
        db.paged(
            "select id, state, number, title, description from bill "
            "where (prefilter_version is distinct from :v) and id > :_after order by id limit :_limit",
            {"v": VERSION},
            page_size=1000,
        )
    )
    updates = []
    for r in rows:
        sc = score(f"{r['title']} {r.get('description') or ''}")
        types = STATE_TYPES.get(r["state"], BILL_TYPES)
        cand = bool(sc) and bool(types.match(r["number"]))
        updates.append({"id": r["id"], "c": cand, "p": sc or None, "v": VERSION})
    db.executemany("update bill set is_candidate = :c, prefilter = :p, prefilter_version = :v where id = :id", updates)
    return {"scored": len(rows), "candidates": sum(u["c"] for u in updates)}
