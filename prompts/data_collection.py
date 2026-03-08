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

# Ordered by collection priority: Find → Location → Notes → Company Confidential Notes → Compensation
ROLE_GAP_DESCRIPTIONS = {
    "Find": (
        "how to find or reach the hiring manager — the internal sponsor, "
        "the external recruiter or search firm, and any community member who can "
        "help make an introduction"
    ),
    "Location": "where the role is based and whether it's remote, hybrid, or in-office",
    "Notes": (
        "role details across three areas — Scope (responsibilities and team size); "
        "Criteria (key skills, interview panel, reason for hire); "
        "Details (hiring manager, who the role reports to)"
    ),
    # Compensation is lowest priority — only surfaced once everything else is known
    "Compensation": (
        "total compensation — only worth asking if the user is deep in the process "
        "or volunteers it; frame naturally, e.g. asking about the overall package"
    ),
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
        if not str(role_fields.get(field) or "").strip()
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
- Notes: structured as three sections — Scope (responsibilities, team size); Criteria (key skills, interview panel, reason for hire); Details (location/remote/hybrid/in-office, hiring manager, reports to). Also capture any qualitative comp context in Criteria (e.g. "comp reportedly low" or "strong equity component").
- Location: where based, remote/hybrid/in-office
- Compensation: INTEGER only — the annual USD cash total (base + bonus / OTE) as a plain number with no symbols, words, or punctuation (e.g. 180000). If the user gives a range use the midpoint. If the figure is vague or only qualitative (e.g. "low", "competitive") set this to null and capture the context in Notes instead.

Company field (only include if mentioned):
- Confidential Notes: any non-public intel — ARR, growth, GTM strategy, org structure, leadership, CEO, culture

Message: "{user_text}"

Rules:
- Only populate fields where the user actually shared relevant info
- Combine related details into the right field (e.g. "HM is Sarah, team of 12" both go in Notes)
- For Confidential Notes collect all company intel in one string
- Compensation must be a plain integer or null — never a string
- Return ONLY valid JSON, nothing else

{{"role":{{"Find":null,"Notes":null,"Location":null,"Compensation":null}},"company":{{"Confidential Notes":null}}}}"""


# ── Structured field schemas ──────────────────────────────────────────────────

ROLE_NOTES_SCHEMA = """\
Scope: <responsibilities and team size>
Criteria: <key skills, interview panel, reason for hire>
Details: <location/remote/hybrid/in-office, hiring manager, reports to>\
"""

COMPANY_NOTES_SCHEMA = """\
Status: ARR ~$Xm, ~Y employees, growth rate Z%, <competitive dynamics>, last raise <series / date>, runway <X months>
GTM Motion: <inbound/outbound mix>, ACV $X, buyers <personas>, GRR X% / NRR X%
GTM Org: <structure>, function leads: <names/titles>, GTM decisions: <CRO-led / CEO-driven>
CEO/Culture: CEO <name> — <reputation/style>, culture: <description>\
"""


def build_structured_merge_prompt(schema_type: str, existing: str, new_info: str) -> str:
    """
    Return a prompt that asks Claude to merge *new_info* into *existing*
    structured notes, preserving the correct schema and eliminating duplicates.

    schema_type: "role_notes" | "company_notes"
    existing:    current field value (may be empty or unstructured)
    new_info:    freshly extracted text to merge in
    """
    if schema_type == "role_notes":
        schema = ROLE_NOTES_SCHEMA
        label = "Role.Notes"
    else:
        schema = COMPANY_NOTES_SCHEMA
        label = "Company.Confidential Notes"

    existing_block = existing.strip() if existing else "(empty)"

    return f"""You are updating the {label} field for a tracked company/role.

TARGET SCHEMA (output must follow this exact structure):
---
{schema}
---

EXISTING CONTENT:
---
{existing_block}
---

NEW INFORMATION TO MERGE IN:
---
{new_info.strip()}
---

Instructions:
- Produce a clean merged version that follows the schema above exactly.
- Each line starts with the label in bold (e.g. "Scope & Responsibilities:") followed by the value.
- NEVER remove or summarise away information from existing content — every fact already there must be
  preserved unless the new information explicitly contradicts it (e.g. a correction).
- Incorporate all new information from "NEW INFORMATION TO MERGE IN".
- Remove exact duplicates only; keep the most specific / recent value when there is a direct conflict.
- If a section has no information at all (existing or new), omit that line entirely — do NOT write "Unknown".
- Be concise: one line per section unless the content genuinely needs more.
- Return ONLY the merged field content — no preamble, no JSON, no markdown fences."""


def build_simple_field_merge_prompt(field_name: str, existing: str, new_info: str) -> str:
    """
    Return a prompt that asks Claude to synthesize *new_info* into *existing*
    for a plain text field, preserving all information without duplication.
    """
    existing_block = existing.strip() if existing else "(empty)"
    return f"""You are updating the "{field_name}" field in a recruiting database.

EXISTING VALUE:
---
{existing_block}
---

NEW INFORMATION TO INCORPORATE:
---
{new_info.strip()}
---

Instructions:
- Write a single concise sentence or phrase that captures ALL the information from both.
- NEVER remove any unique facts from the existing value.
- Eliminate redundancy: if the new information is already expressed in the existing value
  (even partially or in different words), do not repeat it — just keep the best phrasing.
- If the new information contradicts the existing value, keep both with a note (e.g. "X or Y").
- Return ONLY the merged value — no preamble, no labels, no quotes, no markdown."""


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
