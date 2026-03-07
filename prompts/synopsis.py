"""
Prompt builders for the Insights agent synopsis generation.
"""


def build_company_synopsis_prompt(company: dict, roles: list, employees: list, insights: list) -> str:
    """
    Build a prompt that instructs Claude to produce a concise, useful synopsis
    of a company for a job-seeker in the community.

    Args:
        company: Airtable company record (fields dict)
        roles:   List of Airtable role records linked to the company
        employees: List of Airtable employee records at the company
        insights: List of community-contributed insight records
    """
    fields = company.get("fields", {})

    company_section = f"""
COMPANY: {fields.get('Name', 'Unknown')}

Description:
{fields.get('Description', 'No description on file.')}

HG6M (High-Growth 6-Month Outlook):
{fields.get('HG6M', 'Not available.')}

Confidential Notes (internal only – do NOT share verbatim, summarize tastefully):
{fields.get('ConfidentialNotes', 'None.')}

Number of Employees: {fields.get('Employees', 'Unknown')}
""".strip()

    roles_section = ""
    if roles:
        role_lines = []
        for r in roles:
            rf = r.get("fields", {})
            role_lines.append(
                f"- {rf.get('Title', 'Untitled')} | "
                f"Hiring Manager: {rf.get('HiringManager', 'Unknown')} | "
                f"Location: {rf.get('Location', 'Unknown')} | "
                f"Find: {rf.get('Find', '')} | "
                f"Notes: {rf.get('Notes', '')}"
            )
        roles_section = "OPEN ROLES:\n" + "\n".join(role_lines)
    else:
        roles_section = "OPEN ROLES: No open roles currently tracked."

    employees_section = ""
    if employees:
        emp_lines = [f"- {e.get('fields', {}).get('Name', 'Unknown')}" for e in employees[:10]]
        employees_section = "KNOWN EMPLOYEES / CONTACTS:\n" + "\n".join(emp_lines)
    else:
        employees_section = "KNOWN EMPLOYEES / CONTACTS: None on file."

    insights_section = ""
    if insights:
        insight_lines = []
        for i in insights:
            fi = i.get("fields", {})
            insight_lines.append(
                f"- [{fi.get('Timestamp', '')}] {fi.get('Content', '')}"
            )
        insights_section = "COMMUNITY INSIGHTS:\n" + "\n".join(insight_lines)
    else:
        insights_section = "COMMUNITY INSIGHTS: No community insights yet."

    return f"""You are an Insights agent helping members of a professional community learn about companies they are interested in or interviewing with.

Below is the data we have on file for this company. Your job is to generate a friendly, helpful synopsis that:
1. Highlights what makes this company interesting or noteworthy.
2. Mentions the open roles and what we know about them.
3. Summarizes community insights without exposing raw confidential notes – use your judgment to share relevant context diplomatically.
4. Keeps it concise (under 300 words) and conversational – like a knowledgeable friend briefing you before a coffee chat.

---
{company_section}

{roles_section}

{employees_section}

{insights_section}
---

Please write the synopsis now."""


def build_role_synopsis_prompt(role: dict, company: dict, insights: list) -> str:
    """
    Build a prompt for a role-specific synopsis when the user is focused on a particular position.
    """
    rf = role.get("fields", {})
    cf = company.get("fields", {}) if company else {}

    role_section = f"""
ROLE: {rf.get('Title', 'Unknown')}
Company: {cf.get('Name', 'Unknown')}
Hiring Manager: {rf.get('HiringManager', 'Unknown')}
Location: {rf.get('Location', 'Unknown')}
How to Find / Apply: {rf.get('Find', 'Unknown')}
Notes: {rf.get('Notes', 'None.')}
""".strip()

    insights_section = ""
    if insights:
        lines = [f"- [{i.get('fields', {}).get('Timestamp', '')}] {i.get('fields', {}).get('Content', '')}" for i in insights]
        insights_section = "COMMUNITY INSIGHTS ON THIS ROLE:\n" + "\n".join(lines)
    else:
        insights_section = "COMMUNITY INSIGHTS ON THIS ROLE: No insights yet – you could be the first!"

    company_context = f"""
Company Description: {cf.get('Description', 'N/A')}
HG6M Outlook: {cf.get('HG6M', 'N/A')}
""".strip()

    return f"""You are an Insights agent helping a community member learn about a specific job role.

Below is everything we know. Write a friendly, concise (under 200 words) briefing covering:
1. What we know about the role and the hiring process.
2. Community insights that might help them prepare or decide.
3. Any gaps in our knowledge they could help fill in.

---
{role_section}

{company_context}

{insights_section}
---

Please write the role briefing now."""


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
            "ConfidentialNotes": "any insider intel or confidential context",
        }
    else:  # role
        desired = {
            "HiringManager": "who the hiring manager is",
            "Location": "where the role is based (remote/hybrid/on-site, city)",
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
