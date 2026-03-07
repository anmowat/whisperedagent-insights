"""
Insights Agent – core conversation logic.

Orchestrates:
1. Identifying which company/role the user cares about
2. Fetching data from Airtable and generating a synopsis via Claude
3. Suggesting community contacts who have engaged with the company/role
4. Collecting additional insights from the user and writing them back to Airtable
5. Creating new records when a company/role isn't yet in the database
"""

import os
import json
from datetime import datetime, timezone
from typing import Optional

import anthropic

from database.airtable_client import AirtableClient
from agents.state import ConversationState, Phase, StateManager
from prompts.synopsis import (
    build_company_synopsis_prompt,
    build_role_synopsis_prompt,
    build_info_collection_prompt,
)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are the Insights agent for a professional community platform that tracks companies and open roles.
Your goal is to help community members:
1. Quickly learn what we know about a company or role they are interested in.
2. Share new information they have gathered through conversations or interviews.

Be warm, concise, and professional. Always acknowledge what the user has contributed.
When you need to identify a company or role, extract the name clearly from natural language.
Respond conversationally – you are a knowledgeable, helpful colleague, not a form.
"""


class InsightsAgent:
    def __init__(self, db: AirtableClient, state_manager: StateManager):
        self.db = db
        self.state_manager = state_manager
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def handle_message(self, slack_user_id: str, slack_user_name: str, user_text: str) -> str:
        """
        Process one turn of conversation and return the agent's reply.
        This is the single entry point called by the Slack bot.
        """
        state = self.state_manager.get_or_create(slack_user_id, slack_user_name)
        state.add_user_message(user_text)

        if state.phase == Phase.IDENTIFY:
            reply = self._handle_identify(state, user_text)
        elif state.phase == Phase.COMPANY_FOUND:
            reply = self._handle_company_found(state, user_text)
        elif state.phase == Phase.ROLE_FOUND:
            reply = self._handle_role_found(state, user_text)
        elif state.phase == Phase.CREATING_NEW:
            reply = self._handle_creating_new(state, user_text)
        elif state.phase == Phase.COLLECTING_INSIGHTS:
            reply = self._handle_collecting_insights(state, user_text)
        else:
            reply = "It looks like our conversation has wrapped up. Type **/insights** to start a new one!"

        state.add_assistant_message(reply)
        return reply

    def start_conversation(self, slack_user_id: str, slack_user_name: str) -> str:
        """Called when user first invokes the Insights agent."""
        state = self.state_manager.reset(slack_user_id, slack_user_name)
        greeting = (
            f"Hey {slack_user_name}! :wave: I'm the *Insights agent*.\n\n"
            "I can help you:\n"
            "• *Learn* what we know about a company or role you're interested in\n"
            "• *Share* what you've learned from conversations or interviews\n\n"
            "Which company or role would you like to explore? Just tell me the company name, "
            "role title, or both (e.g. _\"Acme Corp\"_ or _\"Software Engineer at Acme Corp\"_)."
        )
        state.add_assistant_message(greeting)
        return greeting

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_identify(self, state: ConversationState, user_text: str) -> str:
        """Extract company/role from the user's message and look them up."""
        # Ask Claude to parse the company and role from natural language
        parsed = self._parse_company_and_role(user_text)
        company_name = parsed.get("company")
        role_title = parsed.get("role")

        if not company_name and not role_title:
            return (
                "I didn't quite catch the company or role name. Could you share it again? "
                "For example: _\"Acme Corp\"_ or _\"Product Manager at Acme Corp\"_."
            )

        company_record = None
        if company_name:
            company_record = self.db.find_company(company_name)

        role_record = None
        if role_title and company_record:
            role_record = self.db.find_role(role_title, company_record["id"])
        elif role_title:
            role_record = self.db.find_role(role_title)

        # ---- Both company and role found ----
        if company_record and role_record:
            state.company_record_id = company_record["id"]
            state.company_name = company_record["fields"].get("Name", company_name)
            state.role_record_id = role_record["id"]
            state.role_title = role_record["fields"].get("Title", role_title)
            state.phase = Phase.ROLE_FOUND
            return self._generate_role_synopsis(state, company_record, role_record)

        # ---- Company found, no specific role ----
        if company_record:
            state.company_record_id = company_record["id"]
            state.company_name = company_record["fields"].get("Name", company_name)
            state.phase = Phase.COMPANY_FOUND
            return self._generate_company_synopsis(state, company_record, role_title)

        # ---- Role found at unknown company ----
        if role_record and not company_record:
            linked_company_ids = role_record["fields"].get("Company", [])
            if linked_company_ids:
                company_record = self.db.companies.get(linked_company_ids[0])
                state.company_record_id = company_record["id"]
                state.company_name = company_record["fields"].get("Name", "")
            state.role_record_id = role_record["id"]
            state.role_title = role_record["fields"].get("Title", role_title)
            state.phase = Phase.ROLE_FOUND
            return self._generate_role_synopsis(state, company_record, role_record)

        # ---- Nothing found ----
        state.company_name = company_name
        state.role_title = role_title
        state.phase = Phase.CREATING_NEW
        entity = company_name or role_title
        return (
            f"I don't have *{entity}* in our database yet. "
            "Would you like to add it? If so, tell me a bit about the company – "
            "what they do, roughly how big they are, and any roles you know about."
        )

    def _handle_company_found(self, state: ConversationState, user_text: str) -> str:
        """User is in a company-level discussion. Collect insights or answer follow-ups."""
        lower = user_text.lower()

        # Check if user wants to share new information
        if any(kw in lower for kw in ["share", "add", "update", "tell you", "learned", "found out", "know"]):
            state.phase = Phase.COLLECTING_INSIGHTS
            return self._ask_for_insights(state)

        # Otherwise use Claude to handle the free-form response (Q&A about the company)
        return self._ask_claude_freeform(state, user_text)

    def _handle_role_found(self, state: ConversationState, user_text: str) -> str:
        """User is in a role-level discussion. Collect insights or answer follow-ups."""
        lower = user_text.lower()

        if any(kw in lower for kw in ["share", "add", "update", "tell you", "learned", "found out", "know", "interviewed"]):
            state.phase = Phase.COLLECTING_INSIGHTS
            return self._ask_for_insights(state)

        return self._ask_claude_freeform(state, user_text)

    def _handle_creating_new(self, state: ConversationState, user_text: str) -> str:
        """User is providing info about a company/role that doesn't exist yet."""
        lower = user_text.lower()
        if any(kw in lower for kw in ["no", "nope", "not now", "skip", "cancel", "never mind"]):
            state.phase = Phase.DONE
            return "No problem! Feel free to come back anytime. Type **/insights** to start a new conversation."

        # Collect what they've shared and create records
        result = self._create_records_from_description(state, user_text)
        state.phase = Phase.COLLECTING_INSIGHTS
        return result

    def _handle_collecting_insights(self, state: ConversationState, user_text: str) -> str:
        """User is sharing specific insights. Extract, save, and ask if there's more."""
        lower = user_text.lower()
        if any(kw in lower for kw in ["no", "nope", "nothing", "that's all", "done", "thanks", "thank you"]):
            state.phase = Phase.DONE
            return (
                "Awesome – thanks for contributing! :tada: Your insights help the whole community.\n"
                "Type **/insights** anytime to look up another company or role."
            )

        # Save the insight
        self._save_insight(state, user_text)

        # Ask if there's anything else
        follow_up_prompt = self._build_follow_up_prompt(state, user_text)
        return self._call_claude([{"role": "user", "content": follow_up_prompt}])

    # ------------------------------------------------------------------
    # Synopsis generators
    # ------------------------------------------------------------------

    def _generate_company_synopsis(
        self, state: ConversationState, company_record: dict, role_hint: Optional[str]
    ) -> str:
        roles = self.db.get_company_roles(company_record["id"])
        insights = self.db.get_company_insights(company_record["id"])
        contributors = self._get_contributor_list(company_record["id"])

        prompt = build_company_synopsis_prompt(company_record, roles, insights)
        synopsis = self._call_claude([{"role": "user", "content": prompt}])

        # Append contributor suggestions
        contributor_block = self._format_contributor_block(contributors)

        suffix = (
            "\n\n*What would you like to do next?*\n"
            "• Share insights you've learned from conversations with them\n"
            "• Ask me a specific question about the company"
        )
        if role_hint:
            suffix += f"\n• Tell me the specific role you're looking at (_{role_hint}_?)"

        return synopsis + contributor_block + suffix

    def _generate_role_synopsis(
        self, state: ConversationState, company_record: Optional[dict], role_record: dict
    ) -> str:
        role_insights = self.db.get_insights_contributors(
            state.company_record_id or "",
            state.role_record_id,
        )
        prompt = build_role_synopsis_prompt(role_record, company_record or {}, role_insights)
        synopsis = self._call_claude([{"role": "user", "content": prompt}])

        contributors = self._get_contributor_list(state.company_record_id or "", state.role_record_id)
        contributor_block = self._format_contributor_block(contributors)

        suffix = (
            "\n\n*What would you like to do next?*\n"
            "• Share what you've learned about this role or the interview process\n"
            "• Ask me something specific about the role or company"
        )
        return synopsis + contributor_block + suffix

    # ------------------------------------------------------------------
    # Info collection helpers
    # ------------------------------------------------------------------

    def _ask_for_insights(self, state: ConversationState) -> str:
        entity_type = "role" if state.role_record_id else "company"
        entity_name = state.role_title or state.company_name or "this"

        if entity_type == "role":
            existing = {}
            if state.role_record_id:
                role_rec = self.db.find_role_by_id(state.role_record_id)
                existing = role_rec.get("fields", {}) if role_rec else {}
        else:
            existing = {}
            if state.company_record_id:
                company_rec = self.db.companies.get(state.company_record_id)
                existing = company_rec.get("fields", {}) if company_rec else {}

        prompt = build_info_collection_prompt(entity_type, entity_name, existing)
        return self._call_claude([{"role": "user", "content": prompt}])

    def _create_records_from_description(self, state: ConversationState, description: str) -> str:
        """Use Claude to extract structured fields, then create Airtable records."""
        extraction_prompt = f"""Extract structured information from this description to create a company/role record.
Return a JSON object with these keys (omit any you can't determine):
- company_name
- company_description
- company_employees (number or text)
- role_title
- role_location
- role_notes
- insights (any other info worth capturing)

Description: {description}

Return ONLY valid JSON, no other text."""

        raw = self._call_claude([{"role": "user", "content": extraction_prompt}])
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}

        created_parts = []

        # Create company if we have a name
        company_record_id = state.company_record_id
        company_name = data.get("company_name") or state.company_name
        if company_name and not company_record_id:
            company_fields = {"Name": company_name}
            if data.get("company_description"):
                company_fields["Description"] = data["company_description"]
            if data.get("company_employees"):
                company_fields["Employees"] = str(data["company_employees"])
            new_company = self.db.create_company(company_fields)
            company_record_id = new_company["id"]
            state.company_record_id = company_record_id
            state.company_name = company_name
            created_parts.append(f"company *{company_name}*")

        # Create role if we have a title
        role_title = data.get("role_title") or state.role_title
        if role_title:
            role_fields = {"Title": role_title}
            if company_record_id:
                role_fields["Company"] = [company_record_id]
            if data.get("role_location"):
                role_fields["Location"] = data["role_location"]
            if data.get("role_notes"):
                role_fields["Notes"] = data["role_notes"]
            new_role = self.db.create_role(role_fields)
            state.role_record_id = new_role["id"]
            state.role_title = role_title
            created_parts.append(f"role *{role_title}*")

        # Save any initial insights
        if data.get("insights") and company_record_id:
            self._save_insight(state, data["insights"])

        if created_parts:
            created_str = " and ".join(created_parts)
            return (
                f"I've added {created_str} to our database. :white_check_mark:\n\n"
                "Is there anything else you'd like to share about them – "
                "like interview process details, culture notes, or hiring manager info?"
            )
        return (
            "Thanks for the info! I wasn't able to extract a clear company or role name – "
            "could you tell me more specifically? (e.g. _\"The company is Acme Corp and the role is Software Engineer\"_)"
        )

    def _save_insight(self, state: ConversationState, content: str) -> None:
        """Write a user-contributed insight to Airtable."""
        fields = {
            "ContributedBy": state.slack_user_id,
            "Content": content,
            "Timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if state.company_record_id:
            fields["Company"] = [state.company_record_id]
        if state.role_record_id:
            fields["Role"] = [state.role_record_id]
        self.db.create_insight(fields)

    def _build_follow_up_prompt(self, state: ConversationState, last_insight: str) -> str:
        entity = state.role_title or state.company_name or "this company/role"
        return (
            f"The user just shared this insight about {entity}: \"{last_insight}\"\n\n"
            "Thank them briefly and ask if they have anything else to add – "
            "for example: interview process details, culture observations, hiring timeline, "
            "or any other info that would help a community member preparing to engage with them. "
            "Keep it short and conversational."
        )

    # ------------------------------------------------------------------
    # Contributor / contact suggestions
    # ------------------------------------------------------------------

    def _get_contributor_list(self, company_id: str, role_id: Optional[str] = None) -> list[str]:
        """Return a deduplicated list of Slack user IDs who have contributed insights."""
        records = self.db.get_insights_contributors(company_id, role_id)
        seen = set()
        contributors = []
        for r in records:
            uid = r.get("fields", {}).get("ContributedBy", "")
            if uid and uid not in seen:
                seen.add(uid)
                contributors.append(uid)
        return contributors

    def _format_contributor_block(self, contributors: list[str]) -> str:
        if not contributors:
            return ""
        mentions = " ".join(f"<@{uid}>" for uid in contributors[:5])
        return (
            f"\n\n:busts_in_silhouette: *Community members who have engaged with this company/role:* "
            f"{mentions}\n_Feel free to ping them for more context!_"
        )

    # ------------------------------------------------------------------
    # Claude helpers
    # ------------------------------------------------------------------

    def _parse_company_and_role(self, user_text: str) -> dict:
        """Ask Claude to extract company name and role title from free-form text."""
        prompt = (
            f"Extract the company name and job role title from this message. "
            f"Return a JSON object with keys 'company' and 'role'. "
            f"Use null for any field you can't determine.\n\nMessage: {user_text}\n\nJSON:"
        )
        raw = self._call_claude([{"role": "user", "content": prompt}])
        try:
            # Claude may include markdown fences
            cleaned = raw.strip().strip("```json").strip("```").strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, AttributeError):
            return {"company": None, "role": None}

    def _ask_claude_freeform(self, state: ConversationState, user_text: str) -> str:
        """Let Claude answer a free-form question using full conversation history."""
        return self._call_claude(state.messages)

    def _call_claude(self, messages: list[dict], max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text
