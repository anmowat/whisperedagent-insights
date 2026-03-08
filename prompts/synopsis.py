"""
Prompt builders for the Insights agent synopsis generation.

mode values:
  "free"    – public info only; no confidential notes, no insights
  "pro"     – full info minus contributor attribution; insights anonymised
  "premium" – full info including who contributed each insight
"""


def build_company_synopsis_prompt(company: dict, roles: list, insights: list, mode: str = "premium") -> str:
    """
    Build a prompt that instructs Claude to produce a concise, useful synopsis
    of a company for a job-seeker in the community.
    """
    fields = company.get("fields", {})

    investors = fields.get('Investors', '')
    if isinstance(investors, list):
        investors = ', '.join(investors)

    company_section = f"""
COMPANY: {fields.get('Company Name', 'Unknown')}

Description:
{fields.get('Description', 'No description on file.')}

HG6M (High-Growth 6-Month Outlook):
{fields.get('HG6M', 'Not available.')}

Number of Employees: {fields.get('Employees', 'Unknown')}
Investors: {investors or 'Unknown'}
""".strip()

    # Confidential notes: pro and premium only
    if mode in ("pro", "premium"):
        conf = fields.get('Confidential Notes', '').strip()
        if conf:
            company_section += f"\n\nConfidential Notes (internal only – do NOT share verbatim, summarise tastefully):\n{conf}"

    roles_section = ""
    if roles:
        role_lines = []
        for r in roles:
            rf = r.get("fields", {})
            if mode == "free":
                # Free tier: titles only — no confidential role details
                role_lines.append(f"- {rf.get('Title', 'Untitled')}")
            else:
                role_lines.append(
                    f"- {rf.get('Title', 'Untitled')} | "
                    f"Hiring Manager: {rf.get('HM Name', 'Unknown')} | "
                    f"Location: {', '.join(rf.get('HQ Location') or []) or fields.get('HQ', 'Unknown')} | "
                    f"Find: {rf.get('Find', '')} | "
                    f"Notes: {rf.get('Notes', '')}"
                )
        roles_section = "OPEN ROLES:\n" + "\n".join(role_lines)
    else:
        roles_section = "OPEN ROLES: No open roles currently tracked."

    # Insights: excluded for free; anonymised for pro; attributed for premium
    insights_section = ""
    if mode == "free":
        insights_section = ""
    elif mode == "pro":
        if insights:
            lines = [
                f"- [{fi.get('Timestamp', '')}] {fi.get('Content', '')}"
                for i in insights
                for fi in [i.get("fields", {})]
            ]
            insights_section = (
                "COMMUNITY INSIGHTS (do NOT attribute to specific contributors):\n"
                + "\n".join(lines)
            )
        else:
            insights_section = "COMMUNITY INSIGHTS: No community insights yet."
    else:  # premium
        if insights:
            lines = [
                f"- [{fi.get('Timestamp', '')}] {fi.get('Content', '')} — shared by {fi.get('Contributor', 'anonymous')}"
                for i in insights
                for fi in [i.get("fields", {})]
            ]
            insights_section = "COMMUNITY INSIGHTS:\n" + "\n".join(lines)
        else:
            insights_section = "COMMUNITY INSIGHTS: No community insights yet."

    # Mode-specific instruction for Claude
    if mode == "free":
        mode_instruction = (
            "IMPORTANT: This is a Free-tier response. Share only public information "
            "(description, headcount, investors, open role titles). Do NOT mention "
            "confidential notes or community insights."
        )
    elif mode == "pro":
        mode_instruction = (
            "IMPORTANT: This is a Pro-tier response. You may reference community insights "
            "but do NOT name or attribute them to specific contributors."
        )
    else:
        mode_instruction = ""

    sections = [company_section, roles_section]
    if insights_section:
        sections.append(insights_section)

    body = "\n\n".join(sections)

    return f"""You are the Insights agent for a professional community. A member just asked about this company.

{mode_instruction + chr(10) + chr(10) if mode_instruction else ""}DATA ON FILE:
---
{body}
---

Write a SHORT response (3-4 sentences max) that:
1. Gives the most useful snapshot of the company — what makes it interesting right now.
2. Notes any open roles briefly (just titles, no details dump).
3. Ends with ONE natural, open question asking what they've learned or experienced with this company.

Do NOT try to share everything — leave room for dialogue. No markdown."""


def build_role_synopsis_prompt(role: dict, company: dict, insights: list, mode: str = "premium", top_gap: str = None) -> str:
    """
    Build a prompt for a role-specific synopsis when the user is focused on a particular position.
    """
    rf = role.get("fields", {})
    cf = company.get("fields", {}) if company else {}

    role_section = f"""
ROLE: {rf.get('Title', 'Unknown')}
Company: {cf.get('Company Name', 'Unknown')}
Hiring Manager: {rf.get('HM Name', 'Unknown')}
Location: {rf.get('HQ Location') and ', '.join(rf.get('HQ Location')) or cf.get('HQ', 'Unknown')}
How to Find / Apply: {rf.get('Find', 'Unknown')}
Notes: {rf.get('Notes', 'None.')}
""".strip()

    # Insights: excluded for free; anonymised for pro; attributed for premium
    if mode == "free":
        insights_section = ""
    elif mode == "pro":
        if insights:
            lines = [
                f"- [{i.get('fields', {}).get('Timestamp', '')}] {i.get('fields', {}).get('Content', '')}"
                for i in insights
            ]
            insights_section = (
                "COMMUNITY INSIGHTS ON THIS ROLE (do NOT attribute to specific contributors):\n"
                + "\n".join(lines)
            )
        else:
            insights_section = "COMMUNITY INSIGHTS ON THIS ROLE: No insights yet – you could be the first!"
    else:  # premium
        if insights:
            lines = [
                f"- [{i.get('fields', {}).get('Timestamp', '')}] {i.get('fields', {}).get('Content', '')} "
                f"— shared by {i.get('fields', {}).get('Contributor', 'anonymous')}"
                for i in insights
            ]
            insights_section = "COMMUNITY INSIGHTS ON THIS ROLE:\n" + "\n".join(lines)
        else:
            insights_section = "COMMUNITY INSIGHTS ON THIS ROLE: No insights yet – you could be the first!"

    company_context = f"""
Company Description: {cf.get('Description', 'N/A')}
HG6M Outlook: {cf.get('HG6M', 'N/A')}
""".strip()

    if mode == "free":
        mode_instruction = (
            "IMPORTANT: This is a Free-tier response. Do NOT share insights or confidential details."
        )
    elif mode == "pro":
        mode_instruction = (
            "IMPORTANT: This is a Pro-tier response. Do NOT name or attribute insights to specific contributors."
        )
    else:
        mode_instruction = ""

    sections = [role_section, company_context]
    if insights_section:
        sections.append(insights_section)

    body = "\n\n".join(sections)

    if mode == "premium":
        can_ask_more = "Let them know they can ask you more questions for additional details."
    else:
        can_ask_more = ""

    if top_gap and mode in ("pro", "premium"):
        ending_instruction = f"End with ONE natural conversational question that could surface: {top_gap}. Do NOT make it sound like a form field."
    else:
        ending_instruction = "End with ONE natural question asking what they've learned from their own conversations."

    return f"""You are the Insights agent for a professional community. A member just asked about this role.

{mode_instruction + chr(10) + chr(10) if mode_instruction else ""}DATA ON FILE:
---
{body}
---

Write a SHORT response (3-4 sentences max) that:
1. Summarises what we know about the role — the 1-2 most useful things.
2. Mentions anything notable about the hiring process if we have it.
{"3. " + can_ask_more + chr(10) if can_ask_more else ""}{"4" if can_ask_more else "3"}. {ending_instruction}

Do NOT try to share everything at once — the goal is to start a dialogue. No markdown."""


def build_info_collection_prompt(entity_type: str, entity_name: str, existing_fields: dict) -> str:
    """
    Build a prompt that guides Claude to ask the user the right follow-up questions
    to collect additional information about a company or role.
    """
    missing = []

    if entity_type == "company":
        desired = {
            "Description": "a general description of what the company does",
            "HG6M": "their growth outlook / momentum over the next 6 months",
            "Employees": "approximate headcount",
            "Confidential Notes": "any insider intel or confidential context",
        }
    else:  # role
        desired = {
            "HM Name": "who the hiring manager is",
            "HQ Location": "where the role is based (remote/hybrid/on-site, city)",
            "Find": "how to find or apply for the role",
            "Notes": "any other notes about the role or interview process",
        }

    for field, description in desired.items():
        if not existing_fields.get(field):
            missing.append(description)

    if not missing:
        return (
            f"We actually have pretty good coverage on {entity_name}. "
            "Ask the user if they have any additional insights or recent updates to share."
        )

    missing_str = "\n".join(f"- {m}" for m in missing)
    return f"""You are collecting insights from a community member about {entity_type} "{entity_name}".

We are missing the following information:
{missing_str}

Ask the user for this information in a natural, conversational way. Ask one or two questions at a time – don't overwhelm them. Be warm and appreciative of their contribution."""
