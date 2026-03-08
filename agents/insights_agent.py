"""
Insights Agent – core conversation logic.

Tier behaviour
──────────────
Premium  Immediately share full synopsis (including who contributed insights)
         after confirmation.  If role/company not found, ask for info.

Pro      Ask user to share what they know first (AWAITING_SHARE), then reveal
         full synopsis minus contributor attribution.
         If not found, ask for info.

Free     Roles:     Ask user to share first; after they share, confirm we have
                    the role but share no details; ask questions; mention upgrade.
         Companies: Immediately share public info (no confidential notes, no
                    insights); then ask follow-up questions.
         If not found, ask for info.
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

        if mode == "free":
            greeting = (
                f"Hey {user_name}! I'm the Insights agent.\n\n"
                "Ask me about any company or role you're exploring and I'll share what public information we have. "
                "The more you share with me, the more our community learns together.\n\n"
                "Upgrade to Pro to unlock full community insights."
            )
        elif mode == "pro":
            greeting = (
                f"Hey {user_name}! I'm the Insights agent.\n\n"
                "Tell me about a company or role you're exploring and share what you've learned. "
                "If it matches our database I'll pull up everything we know — "
                "and your insights help everyone in the community."
            )
        else:  # premium
            greeting = (
                f"Hey {user_name}! I'm the Insights agent.\n\n"
                "Tell me which company or role you want to know about and I'll pull up everything we have, "
                "including community insights and who contributed them.\n\n"
                "For example: \"Airtable\" or \"VP of RevOps at Airtable\"."
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

        # If we have a role but no company, resolve company from the role's linked field.
        if role_record and not company_record:
            linked = role_record["fields"].get("Company", [])
            if linked:
                company_record = self.db.get_company(linked[0])

        if not role_record and not company_record:
            entity = company_name or role_title
            return (
                f"I don't have \"{entity}\" in our database yet. "
                "Tell me what you know about it and I'll add it — "
                "what's the company, the role, and anything you've learned from your conversations?"
            )

        if company_record:
            state.company_record_id = company_record["id"]
            state.company_name = company_record["fields"].get("Company Name", company_name)
        elif company_name:
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
            # Treat as a new query
            state.phase = Phase.IDENTIFY
            state.company_record_id = None
            state.company_name = None
            state.role_record_id = None
            state.role_title = None
            return self._handle_identify(state, user_text)

        # ── Confirmed — branch by tier ─────────────────────────────────────
        mode = state.mode
        entity = state.role_title or state.company_name

        # FREE + company: immediately share public synopsis, then follow-up
        if mode == "free" and not state.role_record_id:
            co_rec = self.db.get_company(state.company_record_id)
            state.phase = Phase.COMPANY_FOUND
            return self._generate_company_synopsis(co_rec, mode="free")

        # FREE + role  | PRO (any): go to AWAITING_SHARE
        if mode in ("free", "pro"):
            state.phase = Phase.AWAITING_SHARE
            if mode == "free":
                return (
                    f"Tell me what you already know about \"{entity}\" from your research or conversations. "
                    "Share what you've learned and I'll let you know what we have."
                )
            else:  # pro
                return (
                    f"Great! Before I pull up what we have on {entity}, "
                    "what have you already learned from your conversations or research? "
                    "Your insights help the whole community."
                )

        # PREMIUM: immediately reveal full synopsis
        if state.role_title and state.company_name:
            role_rec, co_rec = self.db.find_role_for_company(state.role_title, state.company_name)
            if role_rec:
                state.role_record_id = role_rec["id"]
                state.company_record_id = co_rec["id"] if co_rec else state.company_record_id
                state.phase = Phase.ROLE_FOUND
                return self._generate_role_synopsis(co_rec, role_rec, mode="premium")

        if state.role_record_id:
            role_rec = self.db.find_role_by_id(state.role_record_id)
            co_rec = self.db.get_company(state.company_record_id) if state.company_record_id else None
            state.phase = Phase.ROLE_FOUND
            return self._generate_role_synopsis(co_rec, role_rec, mode="premium")

        co_rec = self.db.get_company(state.company_record_id)
        state.phase = Phase.COMPANY_FOUND
        return self._generate_company_synopsis(co_rec, mode="premium")

    def _handle_awaiting_share(self, state: ConversationState, user_text: str) -> str:
        """User has shared what they know — respond based on tier."""
        mode = state.mode
        entity = state.role_title or state.company_name or "this"

        if state.role_record_id:
            role_rec = self.db.find_role_by_id(state.role_record_id)
            co_rec = self.db.get_company(state.company_record_id) if state.company_record_id else None

            if mode == "free":
                # Confirm we have it; don't reveal details; ask questions; mention upgrade
                state.phase = Phase.ROLE_FOUND
                ack_prompt = (
                    f"The user shared this about {entity}: \"{user_text}\"\n\n"
                    "1. Warmly acknowledge their contribution in 1 sentence.\n"
                    "2. Confirm that we do have this role in our database.\n"
                    "3. Do NOT reveal any details we have about the role.\n"
                    "4. Ask 1-2 specific questions to gather more useful information for the community.\n"
                    "5. Briefly mention that upgrading to Pro lets them see everything we know.\n"
                    "Keep it short, friendly, and conversational. No markdown."
                )
                return self._call_claude([{"role": "user", "content": ack_prompt}])

            # Pro or Premium: show role synopsis then acknowledge
            synopsis = self._generate_role_synopsis(co_rec, role_rec, mode=mode) if role_rec else ""
            state.phase = Phase.ROLE_FOUND

        else:
            co_rec = self.db.get_company(state.company_record_id)

            if mode == "free":
                # Public synopsis already shown — now ask for supplemental confidential info
                state.phase = Phase.COMPANY_FOUND
                ack_prompt = (
                    f"The user shared this about {entity}: \"{user_text}\"\n\n"
                    "1. Warmly thank them for the insight in 1 sentence.\n"
                    "2. Ask 1-2 focused follow-up questions to gather insider information "
                    "(e.g. culture, interview process, hiring manager, recent news).\n"
                    "Keep it short and friendly. No markdown."
                )
                return self._call_claude([{"role": "user", "content": ack_prompt}])

            synopsis = self._generate_company_synopsis(co_rec, mode=mode) if co_rec else ""
            state.phase = Phase.COMPANY_FOUND

        ack_prompt = (
            f"The user just shared this about {entity}: \"{user_text}\"\n\n"
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

    def _generate_company_synopsis(self, company_record: dict, mode: str = "premium") -> str:
        roles = self.db.get_company_roles(company_record["id"])
        prompt = build_company_synopsis_prompt(company_record, roles, [], mode=mode)
        synopsis = self._call_claude([{"role": "user", "content": prompt}])
        return synopsis + "\n\nWhat would you like to know more about? You can ask me anything or look up a specific role."

    def _generate_role_synopsis(self, company_record: Optional[dict], role_record: dict, mode: str = "premium") -> str:
        prompt = build_role_synopsis_prompt(role_record, company_record or {}, [], mode=mode)
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
