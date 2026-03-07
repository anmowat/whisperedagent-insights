"""
Insights Agent – core conversation logic (read-only mode).

Premium mode: immediately share synopsis when company/role is identified.
Basic mode: ask user what they know first; only share synopsis after they
            provide information that matches a company/role in the database.
"""

import json
import logging
import os
from typing import Optional

import anthropic

from database.airtable_client import AirtableClient
from agents.state import ConversationState, Phase, StateManager
from prompts.synopsis import build_company_synopsis_prompt, build_role_synopsis_prompt

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are the Insights agent for a professional community platform that tracks companies and open roles.
Your goal is to help community members quickly learn what we know about a company or role they are interested in.

Be warm, concise, and professional. Respond conversationally – you are a knowledgeable, helpful colleague, not a form.
Do not use markdown formatting symbols like **, ##, or _. Use plain text with clear paragraph breaks instead.
"""


class InsightsAgent:
    def __init__(self, db: AirtableClient, state_manager: StateManager):
        self.db = db
        self.state_manager = state_manager
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def start_conversation(self, user_id: str, user_name: str, mode: str = "premium") -> str:
        state = self.state_manager.reset(user_id, user_name, mode)

        if mode == "basic":
            greeting = (
                f"Hey {user_name}! I'm the Insights agent.\n\n"
                "Tell me about a company or role you've been in conversations with or interviewing at, "
                "and share what you've learned so far. "
                "If it matches something in our database, I'll pull up everything else we know."
            )
        else:
            greeting = (
                f"Hey {user_name}! I'm the Insights agent.\n\n"
                "Tell me which company or role you want to know about and I'll pull up everything we have on it.\n\n"
                "For example: \"Airtable\" or \"RevOps Manager at Salesforce\"."
            )

        state.add_assistant_message(greeting)
        return greeting

    def handle_message(self, user_id: str, user_name: str, user_text: str, mode: str = "premium") -> str:
        state = self.state_manager.get_or_create(user_id, user_name, mode)

        if state.mode != mode:
            state.mode = mode

        state.add_user_message(user_text)

        if state.phase == Phase.IDENTIFY:
            reply = self._handle_identify(state, user_text)
        elif state.phase == Phase.CONFIRMING:
            reply = self._handle_confirming(state, user_text)
        elif state.phase == Phase.AWAITING_SHARE:
            reply = self._handle_awaiting_share(state, user_text)
        elif state.phase in (Phase.COMPANY_FOUND, Phase.ROLE_FOUND):
            reply = self._handle_followup(state, user_text)
        else:
            state.phase = Phase.IDENTIFY
            reply = self._handle_identify(state, user_text)

        state.add_assistant_message(reply)
        return reply

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_identify(self, state: ConversationState, user_text: str) -> str:
        """Parse company/role, find the best match, ask user to confirm before showing synopsis."""
        parsed = self._parse_company_and_role(user_text)
        company_name = parsed.get("company")
        role_title = parsed.get("role")

        if not company_name and not role_title:
            return (
                "I didn't catch a company or role name there. "
                "Could you try again? For example: \"Acme Corp\" or \"Product Manager at Acme Corp\"."
            )

        company_record = None
        role_record = None

        if role_title and company_name:
            role_record, company_record = self.db.find_role_for_company(role_title, company_name)

        elif role_title:
            role_record = self.db.find_role(role_title)

        elif company_name:
            company_record = self.db.find_company(company_name)

        # If we have a role but no company, resolve company from the role's own linked field.
        # Only do this when company_record is still None — never overwrite a company that came
        # from the same search as the role (no mixing records from different lookups).
        if role_record and not company_record:
            linked = role_record["fields"].get("Company", [])
            if linked:
                company_record = self.db.get_company(linked[0])

        if not role_record and not company_record:
            entity = company_name or role_title
            return (
                f"I don't have \"{entity}\" in our database yet. "
                "Try a slightly different name, or ask about a different company or role."
            )

        if company_record:
            state.company_record_id = company_record["id"]
            state.company_name = company_record["fields"].get("Name", company_name)
        elif company_name:
            # No company record resolved but store user-supplied name for display
            state.company_name = company_name

        if role_record:
            state.role_record_id = role_record["id"]
            state.role_title = role_record["fields"].get("Title", role_title)

        state.phase = Phase.CONFIRMING

        role_label = state.role_title or "(role)"
        company_label = state.company_name or "(unknown company)"

        if state.role_title and state.company_name:
            return f"I found \"{role_label}\" at {company_label} — is that the one you're asking about?"
        elif state.role_title:
            return f"I found \"{role_label}\" — is that the role you're asking about?"
        else:
            return f"I found {company_label} in our database — is that the company you're asking about?"

    def _handle_confirming(self, state: ConversationState, user_text: str) -> str:
        """User responded to the confirmation prompt — proceed or re-ask."""
        low = user_text.lower()
        affirmative = any(w in low for w in (
            "yes", "yeah", "yep", "yup", "correct", "right", "that's it",
            "exactly", "sure", "ok", "okay", "confirm", "go ahead", "yea",
        ))
        negative = any(w in low for w in (
            "no", "nope", "wrong", "not that", "different", "other", "another",
        ))

        if negative:
            state.phase = Phase.IDENTIFY
            state.company_record_id = None
            state.company_name = None
            state.role_record_id = None
            state.role_title = None
            return (
                "Got it — could you give me a bit more detail? "
                "For example the full role title or the exact company name."
            )

        if not affirmative:
            # Looks like a new query — restart identify
            state.phase = Phase.IDENTIFY
            state.company_record_id = None
            state.company_name = None
            state.role_record_id = None
            state.role_title = None
            return self._handle_identify(state, user_text)

        # Confirmed — branch on mode
        if state.mode == "basic":
            entity = state.role_title or state.company_name
            state.phase = Phase.AWAITING_SHARE
            return (
                f"Great! Before I share what we know about {entity}, "
                "what have you already learned from your conversations or research?"
            )

        # Re-run the same lookup that produced the confirmation to get a fresh,
        # consistent role+company pair — never rely on stored IDs that could be stale
        # or mismatched between separate lookups.
        if state.role_title and state.company_name:
            role_rec, co_rec = self.db.find_role_for_company(state.role_title, state.company_name)
            if role_rec:
                state.role_record_id = role_rec["id"]
                state.company_record_id = co_rec["id"] if co_rec else state.company_record_id
                state.phase = Phase.ROLE_FOUND
                return self._generate_role_synopsis(co_rec, role_rec)

        if state.role_record_id:
            role_rec = self.db.find_role_by_id(state.role_record_id)
            co_rec = self.db.get_company(state.company_record_id) if state.company_record_id else None
            state.phase = Phase.ROLE_FOUND
            return self._generate_role_synopsis(co_rec, role_rec)

        co_rec = self.db.get_company(state.company_record_id)
        state.phase = Phase.COMPANY_FOUND
        return self._generate_company_synopsis(co_rec, None)

    def _handle_awaiting_share(self, state: ConversationState, user_text: str) -> str:
        """Basic mode: user has shared what they know – now reveal the synopsis."""
        # Generate synopsis and combine with acknowledgement of what they shared
        if state.role_record_id:
            role_rec = self.db.find_role_by_id(state.role_record_id)
            company_rec = self.db.get_company(state.company_record_id) if state.company_record_id else None
            synopsis = self._generate_role_synopsis(company_rec, role_rec) if role_rec else ""
            state.phase = Phase.ROLE_FOUND
        else:
            company_rec = self.db.get_company(state.company_record_id)
            synopsis = self._generate_company_synopsis(company_rec, None) if company_rec else ""
            state.phase = Phase.COMPANY_FOUND

        ack_prompt = (
            f"The user just shared this about {state.role_title or state.company_name}: \"{user_text}\"\n\n"
            "Acknowledge what they shared in 1-2 sentences (be genuinely appreciative and specific), "
            "then transition naturally into the synopsis below. "
            "Do not use markdown formatting.\n\n"
            f"Synopsis:\n{synopsis}"
        )
        return self._call_claude([{"role": "user", "content": ack_prompt}])

    def _handle_followup(self, state: ConversationState, user_text: str) -> str:
        """Answer follow-up questions using full conversation history."""
        parsed = self._parse_company_and_role(user_text)
        new_company = parsed.get("company")
        # If they're clearly asking about a different company, start over
        if new_company and new_company.lower() != (state.company_name or "").lower():
            state.phase = Phase.IDENTIFY
            state.company_record_id = None
            state.company_name = None
            state.role_record_id = None
            state.role_title = None
            return self._handle_identify(state, user_text)

        return self._call_claude(state.messages)

    # ------------------------------------------------------------------
    # Synopsis generators
    # ------------------------------------------------------------------

    def _generate_company_synopsis(self, company_record: dict, _role_hint=None) -> str:
        roles = self.db.get_company_roles(company_record["id"])
        prompt = build_company_synopsis_prompt(company_record, roles, [])
        synopsis = self._call_claude([{"role": "user", "content": prompt}])
        return synopsis + "\n\nWhat would you like to know more about? You can ask me anything or look up a specific role."

    def _generate_role_synopsis(self, company_record: Optional[dict], role_record: dict) -> str:
        prompt = build_role_synopsis_prompt(role_record, company_record or {}, [])
        synopsis = self._call_claude([{"role": "user", "content": prompt}])
        return synopsis + "\n\nAnything else you'd like to know about this role or company?"

    # ------------------------------------------------------------------
    # Claude helpers
    # ------------------------------------------------------------------

    def _parse_company_and_role(self, user_text: str) -> dict:
        prompt = (
            "Extract the company name and job role title from this message. "
            "Return a JSON object with keys 'company' and 'role'. "
            "Use null for any field you cannot determine.\n\n"
            f"Message: {user_text}\n\nJSON:"
        )
        raw = self._call_claude([{"role": "user", "content": prompt}])
        try:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, AttributeError):
            return {"company": None, "role": None}

    def _call_claude(self, messages: list[dict], max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text
