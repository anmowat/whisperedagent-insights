"""
Data collection: field schema, gap analysis, and extraction prompts.

Fields we collect from community members
──────────────────────────────────────────
Role.Find         Who's leading the search — internal sponsor at the company,
                  external recruiter (name + firm), and any community member
                  who can help get someone in.

Role.Notes        Open-text field covering: scope / responsibilities / reports,
                  remote/hybrid/in-office, hiring manager, team size, key skills
                  sought, why they're hiring, interview panel details.

Role.Location     Where the role is based (use Region picklist values from Airtable).

Role.Compensation OTE = cash base + bonus, in USD.

Company.Confidential Notes  Non-public intel across four topics:
  Status    — ARR, employee count, growth rate, competitive dynamics,
               last funding round, runway.
  GTM Motion — inbound vs outbound channels, ACV, buyer personas, GRR/NRR.
  GTM Org   — org structure, function leaders, how GTM decisions are made
               (CRO-led vs CEO-involved).
  CEO/Culture — CEO reputation (trusted, founder-mode, etc.), work culture
                (pace, style, values).
"""

# ── Role gap descriptions (in conversational language for Claude) ─────────────

ROLE_GAP_DESCRIPTIONS = {
    "Find": (
        "who's leading the search — the internal hiring sponsor at the company, "
        "the external recruiter or search firm, and any community member who can "
        "help someone get an introduction"
    ),
    "Notes": (
        "role details — scope, responsibilities, who it reports to, team size, "
        "who the hiring manager is, what skills they're prioritising, why they're "
        "making this hire, and what the interview process / panel looks like"
    ),
    "Location": "where the role is based and whether it's remote, hybrid, or in-office",
    "Compensation": "total compensation — OTE cash + bonus in USD",
}

# ── Company confidential gap descriptions ─────────────────────────────────────

COMPANY_GAP_DESCRIPTIONS = {
    "Status": (
        "company health — ARR, employee count, growth rate, competitive dynamics, "
        "when they last raised and what their runway looks like"
    ),
    "GTM Motion": (
        "how they go to market — inbound vs outbound mix, ACV, who they sell to "
        "(buyer personas), and whether GRR/NRR are healthy"
    ),
    "GTM Org": (
        "how the GTM org is structured — who leads each function, and whether "
        "GTM decisions are made by a CRO or are CEO-driven"
    ),
    "CEO/Culture": (
        "the CEO and culture — how the CEO is perceived (trusted, founder-mode, "
        "hands-off), and what the work culture is like (pace, intensity, values)"
    ),
}


# ── Gap analysis ──────────────────────────────────────────────────────────────

def get_role_gaps(role_fields: dict) -> list[tuple[str, str]]:
    """Return (field_name, description) for role fields that are empty."""
    return [
        (field, desc)
        for field, desc in ROLE_GAP_DESCRIPTIONS.items()
        if not (role_fields.get(field) or "").strip()
    ]


def get_company_gaps(company_fields: dict) -> list[tuple[str, str]]:
    """
    Return (topic, description) for company confidential topics not yet covered.
    We check Confidential Notes for keywords; if the field is empty we list all topics.
    """
    notes = (company_fields.get("Confidential Notes") or "").lower()
    keywords = {
        "Status": ["arr", "revenue", "growth", "runway", "raise", "funding", "employees"],
        "GTM Motion": ["inbound", "outbound", "acv", "grr", "nrr", "gtm", "channel"],
        "GTM Org": ["cro", "vp sales", "vp marketing", "gtm org", "reporting"],
        "CEO/Culture": ["ceo", "founder", "culture", "work style", "pace"],
    }
    gaps = []
    for topic, desc in COMPANY_GAP_DESCRIPTIONS.items():
        covered = any(kw in notes for kw in keywords.get(topic, []))
        if not covered:
            gaps.append((topic, desc))
    return gaps


# ── Extraction prompt ─────────────────────────────────────────────────────────

def build_data_extraction_prompt(user_text: str, role_name: str, company_name: str) -> str:
    """
    Prompt Claude to pull structured field values out of a natural-language message.
    Returns a prompt whose response should be a compact JSON object.
    """
    return f"""Extract any database-worthy information from this message about role "{role_name}" at "{company_name}".

Role fields (only include if explicitly mentioned):
- Find: who leads the hiring search (internal sponsor, recruiter name/firm, helpful community contacts)
- Notes: role scope, responsibilities, reporting line, team size, hiring manager name, key skills, reason for hire, interview process / panel
- Location: where based, remote/hybrid/in-office
- Compensation: base + bonus / OTE in USD

Company field (only include if mentioned):
- Confidential Notes: any non-public intel — ARR, growth, GTM strategy, org structure, leadership, CEO, culture

Message: "{user_text}"

Rules:
- Only populate fields where the user actually shared relevant info
- Combine related details into the right field (e.g. "HM is Sarah, team of 12" both go in Notes)
- For Confidential Notes collect all company intel in one string
- Return ONLY valid JSON, nothing else

{{"role":{{"Find":null,"Notes":null,"Location":null,"Compensation":null}},"company":{{"Confidential Notes":null}}}}"""


# ── Follow-up question prompt ─────────────────────────────────────────────────

def build_gap_question_prompt(role_name: str, company_name: str,
                               role_gaps: list, company_gaps: list) -> str:
    """
    Prompt Claude to ask ONE natural follow-up question targeting the highest-priority gap.
    """
    all_gaps = [desc for _, desc in role_gaps] + [desc for _, desc in company_gaps]
    if not all_gaps:
        return (
            f"You already have good coverage on {role_name or company_name}. "
            "Ask the user if they have any recent updates or additional observations to share. "
            "One sentence, warm and casual."
        )

    top_two = all_gaps[:2]
    gaps_text = "\n".join(f"- {g}" for g in top_two)

    return f"""You are chatting with a community member about "{role_name or company_name}".

Things we'd love to learn (pick the most natural one to ask about):
{gaps_text}

Ask ONE open, conversational question that might uncover this information.
- Don't ask for specific fields by name ("what is the team size?" is too robotic)
- Ask naturally ("What's the team setup like there?" is better)
- One question only, 1-2 sentences, warm and curious tone."""
