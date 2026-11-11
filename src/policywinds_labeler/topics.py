"""Seed topic taxonomy (docs/mvp.md section 1). In M1+ the DB `topic` table is the source of truth;
this module seeds it and is used by the M0 verification command."""

TOPICS: dict[str, list[str]] = {
    "firearms": [
        "purchase-background-checks",
        "carry",
        "weapon-accessory-restrictions",
        "red-flag-erpo",
        "storage-minors",
        "preemption-sanctuary",
        "industry-liability",
        "other",
    ],
    "transgender": ["healthcare", "sports", "id-documents", "bathrooms", "education", "other"],
    "abortion": [
        "bans-gestational-limits",
        "exceptions",
        "medication-abortion",
        "travel-interstate-shield",
        "provider-clinic-regulation",
        "funding-insurance",
        "personhood-ivf",
        "ballot-constitutional",
        "other",
    ],
    "immigration-enforcement": [
        "state-federal-cooperation",
        "sanctuary-policies",
        "state-immigration-crimes-border",
        "employment-e-verify",
        "detention-transport",
        "other",
    ],
}

# Direction axis per topic (docs/mvp.md, AI pipeline). Neutral wording: "expands" and "restricts" refer
# to the axis below, not to either side's framing. Users pick which direction they want alerts on.
DIRECTION_AXES: dict[str, str] = {
    "firearms": "the ability to buy, own, carry or make firearms and accessories",
    "transgender": "legal recognition of, and access to services and facilities for, transgender people",
    "abortion": "legal access to abortion",
    "immigration-enforcement": "the scope of immigration enforcement by state or local government",
}
assert DIRECTION_AXES.keys() == TOPICS.keys()


def all_tags() -> list[str]:
    """Flat `topic/subtag` identifiers; a single enum guarantees valid topic-subtag pairs."""
    return [f"{t}/{s}" for t, subs in TOPICS.items() for s in subs]
