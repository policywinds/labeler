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


# One-paragraph definitions for the classifier prompt (neutral wording).
DEFINITIONS: dict[str, str] = {
    "firearms": "Laws on buying, selling, owning, carrying, storing, making or using firearms, ammunition, "
    "magazines and accessories; background checks; age limits; extreme-risk (red-flag) orders; firearm "
    "liability of manufacturers or dealers; state preemption of local gun rules; 'Second Amendment sanctuary' "
    "measures. Not: general violent-crime penalties that merely mention a deadly weapon, hunting seasons or "
    "wildlife rules, unless they change firearm access.",
    "transgender": "Laws specific to transgender or gender-nonconforming people or to gender identity: "
    "gender-transition medical care, sex-designated sports teams, bathrooms and facilities, sex markers on IDs "
    "and records, school policies on pronouns or gender identity, definitions of 'sex' in statute, drag "
    "performance restrictions. Not: general anti-discrimination or health bills with no gender-identity provision.",
    "abortion": "Laws on abortion and closely linked issues: gestational limits and bans, exceptions, "
    "medication abortion, interstate travel and shield protections, clinic and provider regulation, public "
    "funding or insurance coverage, fetal personhood and its effect on IVF, ballot measures and constitutional "
    "amendments on abortion. Not: general maternal or prenatal health, contraception-only bills (unless they "
    "define abortion), or IVF bills with no personhood or abortion provision.",
    "immigration-enforcement": "Laws on enforcing immigration law by state or local government: cooperation "
    "with federal immigration authorities (detainers, 287(g)), sanctuary policies and their bans, state crimes "
    "for illegal entry or presence, border security operations, employer verification (E-Verify), detention "
    "and transport of migrants, human smuggling tied to immigration. Not: benefits or eligibility for "
    "immigrants, worker visas, foreign land-ownership rules, or economic development with no enforcement provision.",
}
assert DEFINITIONS.keys() == TOPICS.keys()

# Prefilter patterns (case-insensitive regex), tuned for recall; the classifier handles precision.
KEYWORDS: dict[str, list[str]] = {
    "firearms": [
        r"firearm",
        r"\bhandgun",
        r"\bguns?\b",
        r"\brifle",
        r"shotgun",
        r"ammunition",
        r"\bmagazines?\b",
        r"assault weapon",
        r"\bweapons?\b",
        r"concealed carry",
        r"license to carry",
        r"suppressor",
        r"silencer",
        r"extreme risk",
        r"red flag",
        r"ghost gun",
        r"bump stock",
        r"second amendment",
        r"shooting range",
        r"machine gun",
        r"\bpistol",
        r"\bfirearms? dealer",
    ],
    "transgender": [
        r"transgender",
        r"gender identity",
        r"gender transition",
        r"gender[- ]affirming",
        r"gender dysphoria",
        r"biological sex",
        r"puberty suppress",
        r"cross[- ]sex",
        r"\bpronoun",
        r"sex[- ]designated",
        r"multiple[- ]occupancy",
        r"\bdrag\b",
        r"sex marker",
        r"based on (?:biological )?sex",
        r"definitions? of (?:male|female|sex)",
        r"\bwomen'?s sports",
        r"interscholastic.*\bsex\b",
    ],
    "abortion": [
        r"abortion",
        r"unborn child",
        r"pregnan",
        r"\bfetal\b",
        r"\bfetus",
        r"\bembryo",
        r"in vitro",
        r"\bivf\b",
        r"reproductive",
        r"contracepti",
        r"mifepristone",
        r"misoprostol",
        r"planned parenthood",
        r"heartbeat",
        r"personhood",
        r"\bpregnancy center",
        r"unborn",
    ],
    "immigration-enforcement": [
        r"immigra",
        r"illegal entry",
        r"\baliens?\b",
        r"noncitizen",
        r"non-citizen",
        r"undocumented",
        r"\bborder",
        r"sanctuary",
        r"e-verify",
        r"\bdetainer",
        r"homeland security",
        r"deport",
        r"human smuggling",
        r"smuggling of persons",
        r"\b287\(g\)",
        r"citizenship status",
        r"lawful presence",
        r"federal immigration",
    ],
}
assert KEYWORDS.keys() == TOPICS.keys()
