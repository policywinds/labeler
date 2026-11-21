# Copyright 2026 Kernelcraft LLC (PolicyWinds)
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Topics and subtags, their definitions, keywords and scope rules, and the fixed measure (direction axis) each
topic's labels are judged against."""

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
    "custody-support": [
        "custody-parenting-time",
        "child-support",
        "paternity",
        "support-enforcement",
        "alimony",
        "domestic-violence-custody",
        "other",
    ],
}

# Direction axis per topic. Neutral wording: "expands" and "restricts" refer
# to the axis below, not to either side's framing. Users pick which direction they want alerts on.
# Short public URL segment per topic (/guns, /tx/guns). Topics without an entry use their tag. Must not be a
# two-letter state code or a reserved site path (checked in tests).
TOPIC_SLUG: dict[str, str] = {
    "firearms": "guns",
    "immigration-enforcement": "immigration",
    "custody-support": "custody",
}

DIRECTION_AXES: dict[str, str] = {
    "firearms": "the ability to buy, own, carry or make firearms and accessories",
    "transgender": "legal recognition of, and access to services and facilities for, transgender people",
    "abortion": "legal access to abortion",
    "immigration-enforcement": "the scope of immigration enforcement by state or local government",
    # Gender-neutral on purpose: the measure is the parent who pays support or has less custody, whoever it is.
    "custody-support": "the rights of the parent who pays support or has less custody, against the obligations "
    "placed on that parent",
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
    "custody-support": "Laws on child custody and parenting time, child support, paternity and spousal support "
    "(alimony): custody standards and presumptions (including joint or equal shared parenting), parenting-time "
    "enforcement and relocation, how child support is calculated, when it starts and ends (including retroactive "
    "support and support from conception), enforcement and penalties for unpaid support, establishing or "
    "disestablishing paternity (including genetic testing and paternity fraud), alimony amount, duration and "
    "termination, domestic-violence or abuse findings that change custody or parenting time, and custody or "
    "visitation claims by grandparents, other relatives, nonparents or additional legal parents. Not: "
    "child-welfare cases brought by the state (child-protective investigations, foster care, removals, "
    "reunification, state-filed termination of parental rights, and procedures, reports or lawyers in those "
    "cases); adoption; visitation of prison inmates; police, jail or interrogation rules; divorce grounds, "
    "waiting periods, property division and records that do not change custody, child support or alimony; tax "
    "credits; protective orders or criminal domestic-violence penalties that do not change custody or parenting "
    "time; marriage rules; guardianship of adults; inheritance; or administration of the child-support agency "
    "with no change for parents.",
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
        r"\bconception\b",  # e.g. child support from conception (TX SB942, US HR1104)
        r"\bconceived\b",
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
    # "custody" alone would match criminal custody ("in custody", "custodial interrogation").
    "custody-support": [
        r"child support",
        r"child custody",
        r"custody of (?:a |the |minor )?child",
        r"joint (?:legal |physical |managing )?(?:custody|conservator)",
        r"\bnon-?custodial",
        r"custodial parent",
        r"(?:managing|possessory) conservator",  # Texas terms for custody
        r"conservatorship of (?:a |the )?child",
        r"possession of (?:and access to )?(?:a |the )?child",
        r"parent-child relationship",  # Texas SAPCR
        r"parenting (?:time|plan)",
        r"shared parenting",
        r"(?:child|grandparent|parental|parenting) visitation",
        r"visitation (?:rights|orders?|schedule)",
        r"\bparentage",
        r"paternity",
        r"spousal (?:support|maintenance)",
        r"\balimony",
        r"\bdivorce",
        r"dissolution of (?:a |the )?marriage",
        r"support orders?",
    ],
}
assert KEYWORDS.keys() == TOPICS.keys()

# A topic's keyword match is dropped when the title or description matches one of these: the AI kept filing
# prison visitation and child-protective cases under custody despite the definition (2026-11-13 check).
KEYWORD_EXCLUDE: dict[str, list[str]] = {
    "custody-support": [
        r"correctional|\binmates?\b|\bprisons?\b|\bjails?\b|incarcerated",
        r"interrogation|law[- ]enforcement",
        r"foster care|protective services|child protective|\bDFPS\b|child welfare",
        r"tax credit",
        r"home ?school",
        r"vital records?|death records?",
        r"divorce; (?:grounds|cruelty)",  # Virginia divorce-grounds bills (no custody or support change)
        # State-filed cases (parents against the state) and other out-of-scope terms
        r"governmental entit",
        r"attorneys? ad litem",
        r"removal and placement",
        r"termination of the parent-child relationship",
        r"relief of care (?:&|and) custody",  # Virginia: a parent asks social services to take the child
        r"companion animal",
        r"\bretire|\bpension",
        r"religious barrier",
        r"gender[- ]affirming",  # covered by the transgender topic
    ],
}
# Topics whose labels count only for bills that matched the topic's keywords (precision guard; label.keyword_guard).
REQUIRES_KEYWORD = {"custody-support"}


# Plain-language meaning of each direction, with examples. The axis alone was misread in v1
# (a background-check mandate was labelled "expands" because it expands checks).
DIRECTION_GUIDE: dict[str, dict[str, str]] = {
    "firearms": {
        "expands": "fewer restrictions on firearms or more ability to buy, own or carry them "
        "(e.g. permitless or wider carry, lower age limits, preempting local gun rules, repealing a ban, "
        "shielding gun owners or dealers from liability, tax breaks on firearm sales)",
        "restricts": "more restrictions on firearms (e.g. new or wider background-check requirements, "
        "higher age limits, waiting periods, bans on weapons, magazines or 3D-printed guns, red-flag orders, "
        "storage mandates, carry bans in places, dealer reporting, removing guns from prohibited persons)",
    },
    "transgender": {
        "expands": "more legal recognition of, or access for, transgender people (e.g. protecting gender-affirming "
        "care, allowing ID sex-marker changes, anti-discrimination protections, access to facilities or teams "
        "matching gender identity, banning 'gender identity change efforts' / conversion therapy for minors)",
        "restricts": "less recognition or access (e.g. banning or limiting gender-affirming care, defining sex as "
        "assigned at birth, sports or bathroom restrictions by birth sex, barring ID marker changes, pronoun limits, "
        "allowing or protecting therapy aimed at changing a minor's gender identity, parental-notification of "
        "a student's gender identity)",
    },
    "abortion": {
        "expands": "more legal access to abortion (e.g. protecting the right, shield laws, coverage mandates, "
        "medication-abortion access, repealing restrictions)",
        "restricts": "less legal access (e.g. bans or earlier limits, narrower exceptions, clinic regulations, "
        "limits on medication abortion or travel, funding bans, fetal personhood)",
    },
    "immigration-enforcement": {
        "expands": "more immigration enforcement by state or local government (e.g. requiring cooperation with ICE "
        "or honoring detainers, banning sanctuary policies, state illegal-entry crimes, E-Verify mandates, "
        "border operations)",
        "restricts": "less immigration enforcement by state or local government (e.g. sanctuary or "
        "limited-cooperation policies, limits on sharing data with ICE, barring enforcement officers from "
        "schools, courts or health facilities without a judicial warrant, limits on detention contracts, "
        "repealing required ICE agreements)",
    },
    "custody-support": {
        "expands": "more rights or fewer obligations for the parent who pays support or has less custody (e.g. a "
        "presumption of joint or equal shared parenting, enforcing parenting time, ending support when genetic "
        "testing excludes the man named as father, limits on retroactive support, easier modification or an "
        "earlier end of child support or alimony, alimony caps or end dates, requiring evidence or a hearing before "
        "abuse allegations change custody, making it harder for grandparents or other nonparents to win custody "
        "or visitation over a parent's objection)",
        "restricts": "fewer rights or more obligations for that parent (e.g. support starting at conception or "
        "reaching further back, support extended past 18 or through college, higher support amounts, new penalties "
        "for unpaid support such as license suspension, jail or new crimes, limits on contesting paternity, "
        "presumptions against custody or parenting time for a parent found to have committed domestic violence, "
        "longer or permanent alimony, more rights for grandparents, other nonparents or additional legal parents to "
        "seek custody or visitation over a parent's objection)",
    },
}
assert DIRECTION_GUIDE.keys() == TOPICS.keys()
