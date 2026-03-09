"""
Prompt builders for the Insights agent synopsis generation.

mode values:
  "free"    – public info only; no confidential notes, no insights
  "pro"     – full info minus contributor attribution; insights anonymised
  "premium" – full info including who contributed each insight
"""


def build_company_synopsis_prompt(company: dict, roles: list, insights: list, mode: str = "premium",
                                   company_url: str = "") -> str:
    """
    Build a prompt that instructs Claude to produce a concise, useful synopsis
    of a company for a job-seeker in the community.
    """
    fields = company.get("fields", {})
    company_name = fields.get('Company Name', 'Unknown')
    company_ref = f"[{company_name}]({company_url})" if company_url else company_name

    investors = fields.get('Investors', '')
    if isinstance(investors, list):
        investors = ', '.join(investors)

    company_section = f"""
COMPANY: {company_name}

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
                    + (f"Function: {rf['Function']} | " if rf.get('Function') else "")
                    + f"Hiring Manager: {rf.get('HM Name', 'Unknown')} | "
                    f"Region: {', '.join(rf.get('Region') or []) or fields.get('HQ', 'Unknown')} | "
                    f"Remote: {'Yes' if rf.get('Remote') else 'No'} | "
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

    link_instruction = (
        f"Formatting: whenever you mention the company name in your response, "
        f"write it as the markdown hyperlink {company_ref} — do not write the plain name.\n\n"
        if company_url else ""
    )

    return f"""You are the Insights agent for a professional community. A member just asked about this company.

{mode_instruction + chr(10) + chr(10) if mode_instruction else ""}{link_instruction}DATA ON FILE:
---
{body}
---

Write a SHORT response (3-4 sentences max) that:
1. Gives the most useful snapshot of the company — what makes it interesting right now.
2. Notes any open roles briefly (just titles, no details dump).
3. Ends with ONE natural, open question asking what they've learned or experienced with this company.

IMPORTANT: Only reference facts that are explicitly present in the DATA ON FILE above. Do NOT invent, estimate, or compute any metric, score, or figure (such as a "growth outlook score") that does not appear verbatim in the data. If a field says "Not available" or "N/A", omit it entirely.

Bold only the question sentence using **double asterisks**. Do not use any other markdown. Do NOT try to share everything — leave room for dialogue."""


def build_role_synopsis_prompt(role: dict, company: dict, insights: list, mode: str = "premium",
                               top_gap: str = None, company_url: str = "", role_url: str = "") -> str:
    """
    Build a prompt for a role-specific synopsis when the user is focused on a particular position.
    """
    rf = role.get("fields", {})
    cf = company.get("fields", {}) if company else {}

    role_title = rf.get('Title', 'Unknown')
    role_ref = f"[{role_title}]({role_url})" if role_url else role_title
    company_name = cf.get('Company Name', 'Unknown')
    company_ref = f"[{company_name}]({company_url})" if company_url else company_name

    function_line = f"\nFunction: {rf['Function']}" if rf.get('Function') else ""
    role_section = f"""
ROLE: {role_title}{function_line}
Company: {company_name}
Hiring Manager: {rf.get('HM Name', 'Unknown')}
Region: {', '.join(rf.get('Region') or []) or cf.get('HQ', 'Unknown')}
Remote: {'Yes' if rf.get('Remote') else 'No'}
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
        ending_instruction = (
            f"You MUST end with ONE question specifically aimed at surfacing: {top_gap}. "
            f"Frame it as a natural, conversational question — not a form field. "
            f"Do NOT ask why the user wants the role or anything about their personal motivations."
        )
    else:
        ending_instruction = (
            "End with ONE question asking what they know about the hiring process, timeline, "
            "or how the search is being run. Do NOT ask why the user wants the role or "
            "anything about their personal background or motivations."
        )

    link_parts = []
    if role_url:
        link_parts.append(f"the role name as {role_ref}")
    if company_url:
        link_parts.append(f"the company name as {company_ref}")
    link_instruction = (
        f"Formatting: whenever you mention {' and '.join(link_parts)} in your response, "
        "use those exact markdown hyperlinks — do not write the plain names.\n\n"
        if link_parts else ""
    )

    return f"""You are the Insights agent for a professional community. A member just asked about this role.

{mode_instruction + chr(10) + chr(10) if mode_instruction else ""}{link_instruction}DATA ON FILE:
---
{body}
---

Write a SHORT response (3-4 sentences max) that:
1. Summarises what we know about the role — the 1-2 most useful things.
2. Mentions anything notable about the hiring process if we have it.
{"3. " + can_ask_more + chr(10) if can_ask_more else ""}{"4" if can_ask_more else "3"}. {ending_instruction}

IMPORTANT: Only reference facts that are explicitly present in the DATA ON FILE above. Do NOT invent, estimate, or compute any metric, score, or figure (such as a "growth outlook score") that does not appear verbatim in the data. If a field says "Not available" or "N/A", omit it entirely.

Bold only the question sentence using **double asterisks**. Do not use any other markdown. Do NOT try to share everything at once — the goal is to start a dialogue."""


def build_roles_listing_prompt(company: dict, open_roles: list, closed_roles: list, company_url: str = "") -> str:
    """
    Build a prompt for a premium user asking what roles we have tracked at a company.
    Lists open roles and recently closed ones.
    """
    cf = company.get("fields", {})
    company_name = cf.get("Company Name", "this company")
    company_ref = f"[{company_name}]({company_url})" if company_url else company_name

    def format_role(r):
        rf = r.get("fields", {})
        region = ", ".join(rf.get("Region") or []) or cf.get("HQ", "Unknown")
        remote_flag = "Remote" if rf.get("Remote") else ""
        location = " | ".join(filter(None, [region, remote_flag]))
        title = rf.get("Title", "Untitled")
        app_page = (rf.get("App Page") or "").strip()
        title_ref = f"[{title}]({app_page})" if app_page else title
        # Always include the company name so attribution is unambiguous
        parts = [f"{title_ref} at {company_name}"]
        if rf.get("Function"):
            parts.append(f"Function: {rf['Function']}")
        if rf.get("HM Name"):
            parts.append(f"HM: {rf['HM Name']}")
        if location and location != "Unknown":
            parts.append(f"Location: {location}")
        if rf.get("Find"):
            parts.append(f"Find: {rf['Find']}")
        if rf.get("Notes"):
            parts.append(f"Notes: {rf['Notes']}")
        return "- " + " | ".join(parts)

    open_section = "\n".join(format_role(r) for r in open_roles) if open_roles else "None tracked."
    closed_section = "\n".join(format_role(r) for r in closed_roles) if closed_roles else "None tracked."

    multiple_open = len(open_roles) > 1
    ending_instruction = (
        "End by asking the user which of these roles they'd like to explore further — "
        "do NOT ask a specific question about one role before knowing which one they care about."
        if multiple_open else
        "End with ONE question about what the user has heard regarding the hiring process or timeline."
    )

    link_instruction = (
        f"Formatting: when mentioning the company name use {company_ref}, "
        "and when listing role titles use the markdown links provided in OPEN ROLES above — "
        "do not write plain names.\n\n"
        if company_url else ""
    )

    return f"""You are the Insights agent for a professional community. A premium member asked about the roles we have tracked for {company_ref}.

{link_instruction}OPEN ROLES:
{open_section}

RECENTLY CLOSED ROLES:
{closed_section}

Write a SHORT response (3-5 sentences) that:
1. Lists the open roles — each with its company name, hiring manager, and location if available. Make the company name explicit for every role, even when they are all the same company.
2. Mentions recently closed roles if any exist.
3. {ending_instruction}

IMPORTANT: Never attribute a role to a company other than the one stated in the data above.
Bold only the question/prompt sentence using **double asterisks**. Do not use any other markdown."""


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
