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


def all_tags() -> list[str]:
    """Flat `topic/subtag` identifiers; a single enum guarantees valid topic-subtag pairs."""
    return [f"{t}/{s}" for t, subs in TOPICS.items() for s in subs]
