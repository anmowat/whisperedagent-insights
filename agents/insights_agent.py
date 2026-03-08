"""
Insights Agent – core conversation logic.

Tier behaviour
──────────────
Premium  Immediately share brief synopsis after confirmation, then dialog to
         fill gaps.  Includes contributor attribution in insights.

Pro      Ask user to share first (AWAITING_SHARE), then brief synopsis minus
         contributor attribution; dialog to fill gaps.

Free     Roles:     Ask user to share first; confirm we have it but share no
                    details; ask questions; mention upgrade.
         Companies: Immediately share brief public snapshot; then dialog for
                    supplemental confidential info.

All tiers: data shared by users is accumulated in state.suggested_updates
           (visible in the UI panel but NOT written to Airtable).
"""

import json
import logging
import os
from typing import Optional

import anthropic

from database.airtable_client import AirtableClient
from agents.state import ConversationState, Phase, StateManager
from prompts.synopsis import build_company_synopsis_prompt, build_role_synopsis_prompt
from prompts.data_collection import (
    build_data_extraction_prompt,
    build_gap_question_prompt,
    build_structured_merge_prompt,
    get_role_gaps,
    get_company_gaps,
)

logger = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are the Insights agent for a professional community platform that tracks companies and open roles.
Your goal is to help community members and to learn from them — every conversation is a chance to fill in gaps in our knowledge.

Guidelines:
- Be warm, concise, and conversational. You are a knowledgeable colleague, not a database.
- Keep responses SHORT. 2-4 sentences is usually right. Leave room for dialogue.
- Ask ONE question at a time. Never list questions.
- Ask open, natural questions — not "what is the team size?" but "what's the team setup like there?"
- Bold the sentence containing your question using **double asterisks like this?** — do not bold anything else.
- Do not use any other markdown (no ##, no -, no _).
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
                "Upgrade to Pro to unlock full community insights."
            )
        elif mode == "pro":
            greeting = (
                f"Hey {user_name}! I'm the Insights agent.\n\n"
                "Tell me about a company or role you're exploring. "
                "Share what you've learned and I'll pull up what we have — "
                "your insights help the whole community."
            )
        else:  # premium
            greeting = (
                f"Hey {user_name}! I'm the Insights agent.\n\n"
                "Which company or role do you want to know about? "
                "I'll share what we have and we can compare notes.\n\n"
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
        parsed = self._parse_company_and_role(user_text)
        company_name = parsed.get("company")
        role_title = parsed.get("role")

        if not company_name and not role_title:
            return (
                "I didn't catch a company or role name there. "
                "**Could you try again? For example: \"Acme Corp\" or \"Product Manager at Acme Corp\".**"
            )

        company_record = None
        role_record = None

        # Fall back to the company already in state if the message didn't name one
        effective_company = company_name or state.company_name

        if role_title and effective_company:
            role_record, company_record = self.db.find_role_for_company(role_title, effective_company)
        elif role_title:
            role_record = self.db.find_role(role_title)
        elif company_name:
            company_record = self.db.find_company(company_name)

        if role_record and not company_record:
            linked = role_record["fields"].get("Company", [])
            if linked:
                company_record = self.db.get_company(linked[0])

        if not role_record and not company_record:
            entity = company_name or role_title
            return (
                f"I don't have \"{entity}\" in our database yet. "
                "Tell me what you know about it — what's the company, the role, "
                "and what have you learned from your conversations?"
            )

        if company_record:
            state.company_record_id = company_record["id"]
            state.company_name = company_record["fields"].get("Company Name", company_name)
            raw_domain = company_record["fields"].get("Domain") or ""
            state.company_domain = self._ensure_https(raw_domain.strip())
        elif company_name:
            state.company_name = company_name

        if role_record:
            state.role_record_id = role_record["id"]
            state.role_title = role_record["fields"].get("Title", role_title)
            state.role_app_page = (role_record["fields"].get("App Page") or "").strip()

        # Go straight to the tier-appropriate response — no confirmation step needed.
        # The hyperlinked company/role name in the response serves as implicit confirmation.
        return self._dispatch_after_match(state, user_text)

    def _handle_confirming(self, state: ConversationState, user_text: str) -> str:
        """Legacy phase handler — state should no longer enter CONFIRMING in normal flow."""
        return self._dispatch_after_match(state, user_text)

    def _dispatch_after_match(self, state: ConversationState, user_text: str) -> str:
        """
        Route to the tier-appropriate response immediately after a company/role is matched.
        Uses the hyperlinked company/role name as implicit confirmation — no round-trip needed.
        """
        mode = state.mode
        entity_ref = self._role_ref(state) or self._company_ref(state)

        roles_list_intent = (
            not state.role_record_id
            and state.company_record_id
            and self._is_roles_list_intent(user_text)
        )

        # FREE + company: role-listing intent → confirm existence only; otherwise show public snapshot
        if mode == "free" and not state.role_record_id:
            state.phase = Phase.COMPANY_FOUND
            if roles_list_intent:
                roles = self.db.get_company_roles(state.company_record_id)
                count = len(roles)
                noun = "role" if count == 1 else "roles"
                co_ref = self._company_ref(state)
                if count:
                    return (
                        f"We do have {count} {noun} tracked for {co_ref}, "
                        "but the details are only available on Pro and above. "
                        "**Upgrade to Pro to see the role titles and hiring details.**"
                    )
                return f"I don't have any roles tracked for {co_ref} at the moment."
            co_rec = self.db.get_company(state.company_record_id)
            return self._generate_company_synopsis(co_rec, mode="free", state=state)

        # FREE + role | PRO (any): role-listing intent for pro → confirm existence only; otherwise share-first flow
        if mode in ("free", "pro"):
            if mode == "pro" and roles_list_intent:
                state.phase = Phase.COMPANY_FOUND
                roles = self.db.get_company_roles(state.company_record_id)
                count = len(roles)
                noun = "role" if count == 1 else "roles"
                co_ref = self._company_ref(state)
                if count:
                    return (
                        f"We do have {count} {noun} tracked for {co_ref}. "
                        "**Upgrade to Premium to see the full breakdown with hiring manager and location details — "
                        "or is there a specific role you've already heard about?**"
                    )
                return f"I don't have any roles tracked for {co_ref} at the moment."
            state.phase = Phase.AWAITING_SHARE
            if mode == "free":
                return (
                    f"Tell me what you already know about {entity_ref} from your research or conversations. "
                    "**Share what you've learned and I'll let you know what we have.**"
                )
            else:  # pro
                return (
                    f"Great — before I pull up what we have on {entity_ref}, "
                    "**what have you already learned from your conversations or research?**"
                )

        # PREMIUM: role-listing intent → show full roles list
        if roles_list_intent:
            state.phase = Phase.COMPANY_FOUND
            return self._list_company_roles(state)

        # PREMIUM: role matched → show synopsis
        if state.role_record_id:
            role_rec = self.db.find_role_by_id(state.role_record_id)
            co_rec = self.db.get_company(state.company_record_id) if state.company_record_id else None
            state.phase = Phase.ROLE_FOUND
            return self._generate_role_synopsis(co_rec, role_rec, mode="premium", state=state)

        co_rec = self.db.get_company(state.company_record_id)
        state.phase = Phase.COMPANY_FOUND
        return self._generate_company_synopsis(co_rec, mode="premium", state=state)

    def _handle_awaiting_share(self, state: ConversationState, user_text: str) -> str:
        """User has shared what they know — respond based on tier."""
        mode = state.mode
        entity = state.role_title or state.company_name or "this"

        # Extract any structured data from what they shared
        self._extract_and_accumulate(state, user_text)

        if state.role_record_id:
            role_rec = self.db.find_role_by_id(state.role_record_id)
            co_rec = self.db.get_company(state.company_record_id) if state.company_record_id else None

            if mode == "free":
                state.phase = Phase.ROLE_FOUND
                ack_prompt = (
                    f"The user shared this about {entity}: \"{user_text}\"\n\n"
                    "1. Warmly acknowledge their contribution in 1 sentence.\n"
                    "2. Confirm that we do have this role in our database.\n"
                    "3. Do NOT reveal any details we have about the role.\n"
                    "4. Ask ONE natural follow-up question to learn more.\n"
                    "5. In a final sentence, mention they can upgrade to Pro to see what we know.\n"
                    "Bold only the question sentence using **double asterisks**. No other markdown."
                )
                return self._call_claude([{"role": "user", "content": ack_prompt}])

            synopsis = self._generate_role_synopsis(co_rec, role_rec, mode=mode, state=state) if role_rec else ""
            state.phase = Phase.ROLE_FOUND

        else:
            co_rec = self.db.get_company(state.company_record_id)

            if mode == "free":
                state.phase = Phase.COMPANY_FOUND
                ack_prompt = (
                    f"The user shared this about {entity}: \"{user_text}\"\n\n"
                    "1. Warmly thank them for the insight in 1 sentence.\n"
                    "2. Ask ONE focused follow-up question to gather more insider info "
                    "(culture, hiring process, leadership, recent changes).\n"
                    "Bold only the question sentence using **double asterisks**. No other markdown."
                )
                return self._call_claude([{"role": "user", "content": ack_prompt}])

            synopsis = self._generate_company_synopsis(co_rec, mode=mode, state=state) if co_rec else ""
            state.phase = Phase.COMPANY_FOUND

        ack_prompt = (
            f"The user just shared this about {entity}: \"{user_text}\"\n\n"
            "Acknowledge what they shared in 1 sentence (be specific and appreciative), "
            "then transition naturally into the synopsis below. "
            "Bold only the question sentence using **double asterisks**. No other markdown.\n\n"
            f"Synopsis:\n{synopsis}"
        )
        return self._call_claude([{"role": "user", "content": ack_prompt}])

    def _handle_followup(self, state: ConversationState, user_text: str) -> str:
        """Answer follow-up questions, extract data, and probe gaps conversationally."""
        # Role listing intent — tier-appropriate response
        if state.company_record_id and self._is_roles_list_intent(user_text):
            if state.mode == "premium":
                return self._list_company_roles(state)
            else:
                roles = self.db.get_company_roles(state.company_record_id)
                count = len(roles)
                noun = "role" if count == 1 else "roles"
                co_ref = self._company_ref(state)
                if not count:
                    return f"I don't have any roles tracked for {co_ref} at the moment."
                if state.mode == "free":
                    return (
                        f"We do have {count} {noun} tracked for {co_ref}, "
                        "but the details are only available on Pro and above. "
                        "**Upgrade to Pro to unlock the role titles and hiring details.**"
                    )
                # pro
                return (
                    f"We do have {count} {noun} tracked for {co_ref}. "
                    "**Upgrade to Premium to see the full breakdown — "
                    "or is there a specific role you've already come across?**"
                )

        # Check if they're asking about a new entity — reset and re-identify if so
        parsed = self._parse_company_and_role(user_text)
        new_company = parsed.get("company")
        new_role = parsed.get("role")
        current_company = (state.company_name or "").lower()
        current_role = (state.role_title or "").lower()

        switching_company = (
            new_company
            and new_company.lower() != current_company
            and new_company.lower() not in current_company
        )
        # Role switch: new role mentioned, and either no company in message or it's the same company
        same_company_mentioned = (
            new_company
            and (new_company.lower() == current_company or new_company.lower() in current_company)
        )
        switching_role = (
            new_role
            and not switching_company
            and (not new_company or same_company_mentioned)
            and new_role.lower() != current_role
        )

        if switching_company:
            state.phase = Phase.IDENTIFY
            state.company_record_id = None
            state.company_name = None
            state.role_record_id = None
            state.role_title = None
            state.suggested_updates = {}
            return self._handle_identify(state, user_text)

        if switching_role:
            # Keep company context — user is asking about a different role at the same company
            state.phase = Phase.IDENTIFY
            state.role_record_id = None
            state.role_title = None
            return self._handle_identify(state, user_text)

        # Extract structured data from what the user said (silent — never raises)
        self._extract_and_accumulate(state, user_text)

        # Determine remaining gaps to guide the next question
        role_fields = {}
        company_fields = {}
        if state.role_record_id:
            role_rec = self.db.find_role_by_id(state.role_record_id)
            role_fields = (role_rec or {}).get("fields", {})
        if state.company_record_id:
            co_rec = self.db.get_company(state.company_record_id)
            company_fields = (co_rec or {}).get("fields", {})

        merged_role = {**role_fields, **{k: v for k, v in state.suggested_updates.get("role", {}).items() if v}}
        merged_company = {**company_fields, **{k: v for k, v in state.suggested_updates.get("company", {}).items() if v}}

        role_gaps = get_role_gaps(merged_role) if state.role_record_id else []
        company_gaps = get_company_gaps(merged_company)

        # Pass gap hints via a dynamic system prompt — NEVER inject into the user's message
        # (injecting into user messages confuses Claude into producing greeting-like responses)
        system = SYSTEM_PROMPT
        if role_gaps or company_gaps:
            top = [desc for _, desc in (role_gaps + company_gaps)][:2]
            system = (
                SYSTEM_PROMPT
                + "\n\nAfter your response, ask ONE natural follow-up question that could surface: "
                + "; ".join(top)
                + ". Frame it as a conversational question, not a form field prompt."
            )

        # Premium + role: ensure Claude always gives a brief overview of what we know
        if state.mode == "premium" and role_fields:
            key_facts = {k: v for k, v in merged_role.items() if v and k in ("Title", "HM Name", "HQ Location", "Find", "Notes")}
            system = system + (
                "\n\nIMPORTANT: You are discussing the role below with a premium member. "
                "Always reference what we know about it (briefly) before asking your question.\n"
                f"ROLE DATA: {json.dumps(key_facts)}"
            )

        return self._call_claude(state.messages, system=system)

    # ------------------------------------------------------------------
    # Link helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_https(url: str) -> str:
        if not url:
            return url
        return url if url.startswith(("http://", "https://")) else "https://" + url

    def _company_ref(self, state: ConversationState) -> str:
        """Return a markdown hyperlink for the company, or plain name if no domain."""
        name = state.company_name or "the company"
        url = state.company_domain or ""
        return f"[{name}]({url})" if url else name

    def _role_ref(self, state: ConversationState) -> str:
        """Return a markdown hyperlink for the role, or plain title if no app page."""
        title = state.role_title or ""
        url = state.role_app_page or ""
        if not title:
            return ""
        return f"[{title}]({url})" if url else title

    # ------------------------------------------------------------------
    # Roles listing (premium only)
    # ------------------------------------------------------------------

    def _is_roles_list_intent(self, text: str) -> bool:
        low = text.lower()
        return any(p in low for p in [
            "what roles", "which roles", "list roles", "any roles", "open roles",
            "roles do you have", "roles you have", "tell me about the roles",
            "roles in our", "roles in the", "what positions", "any positions",
            "open positions", "what openings", "any openings",
        ])

    def _list_company_roles(self, state: ConversationState) -> str:
        from prompts.synopsis import build_roles_listing_prompt
        co_rec = self.db.get_company(state.company_record_id)
        roles = self.db.get_company_roles(state.company_record_id)
        open_roles = [r for r in roles if (r["fields"].get("Status") or "open").lower() != "closed"]
        closed_roles = [r for r in roles if (r["fields"].get("Status") or "").lower() == "closed"]
        company_url = state.company_domain or ""
        prompt = build_roles_listing_prompt(co_rec or {}, open_roles, closed_roles, company_url=company_url)
        return self._call_claude([{"role": "user", "content": prompt}])

    # ------------------------------------------------------------------
    # Synopsis generators
    # ------------------------------------------------------------------

    def _generate_company_synopsis(self, company_record: dict, mode: str = "premium",
                                    state: Optional[ConversationState] = None) -> str:
        roles = self.db.get_company_roles(company_record["id"])
        company_url = (state.company_domain or "") if state else ""
        prompt = build_company_synopsis_prompt(company_record, roles, [], mode=mode, company_url=company_url)
        return self._call_claude([{"role": "user", "content": prompt}])

    def _generate_role_synopsis(self, company_record: Optional[dict], role_record: dict, mode: str = "premium",
                                 state: Optional[ConversationState] = None) -> str:
        role_fields = (role_record or {}).get("fields", {})
        company_fields = (company_record or {}).get("fields", {})
        role_gaps = get_role_gaps(role_fields)
        company_gaps = get_company_gaps(company_fields)
        all_gaps = role_gaps + company_gaps
        top_gap = all_gaps[0][1] if all_gaps else None
        company_url = (state.company_domain or "") if state else ""
        role_url = (state.role_app_page or "") if state else ""
        prompt = build_role_synopsis_prompt(role_record, company_record or {}, [], mode=mode, top_gap=top_gap,
                                            company_url=company_url, role_url=role_url)
        return self._call_claude([{"role": "user", "content": prompt}])

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_and_accumulate(self, state: ConversationState, user_text: str) -> None:
        """Extract structured field values from user_text and accumulate in state.suggested_updates."""
        role_name = state.role_title or ""
        company_name = state.company_name or ""
        if not role_name and not company_name:
            return

        prompt = build_data_extraction_prompt(user_text, role_name, company_name)
        raw = self._call_claude([{"role": "user", "content": prompt}], max_tokens=512)

        try:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            extracted = json.loads(cleaned)
        except (json.JSONDecodeError, AttributeError, ValueError):
            logger.debug("data extraction parse failed: %r", raw[:200])
            return

        updates = state.suggested_updates
        if "role" not in updates:
            updates["role"] = {}
        if "company" not in updates:
            updates["company"] = {}

        updates["role_name"] = role_name
        updates["company_name"] = company_name

        # Fetch current Airtable field values so the merge baseline includes
        # any information already stored in the database (not just prior
        # suggested_updates accumulated this session).
        airtable_role_fields: dict = {}
        airtable_company_fields: dict = {}
        if state.role_record_id:
            role_rec = self.db.find_role_by_id(state.role_record_id)
            airtable_role_fields = (role_rec or {}).get("fields", {})
        if state.company_record_id:
            co_rec = self.db.get_company(state.company_record_id)
            airtable_company_fields = (co_rec or {}).get("fields", {})

        for field, value in (extracted.get("role") or {}).items():
            if value is None:
                continue
            if field == "Compensation":
                # Number field in Airtable — store as int, ignore non-numeric values
                try:
                    updates["role"]["Compensation"] = int(value)
                except (TypeError, ValueError):
                    pass
            elif field == "Notes":
                # Prefer already-merged suggested value; fall back to Airtable
                existing = updates["role"].get("Notes") or airtable_role_fields.get("Notes", "")
                merged = self._structured_merge("role_notes", existing, value)
                updates["role"]["Notes"] = merged
            else:
                # Prefer already-merged session value; fall back to Airtable — never discard existing info
                existing = str(updates["role"].get(field) or airtable_role_fields.get(field) or "")
                updates["role"][field] = (existing + "\n" + value).strip() if existing else value

        for field, value in (extracted.get("company") or {}).items():
            if value:
                if field == "Confidential Notes":
                    # Prefer already-merged suggested value; fall back to Airtable
                    existing = (
                        updates["company"].get("Confidential Notes")
                        or airtable_company_fields.get("Confidential Notes", "")
                    )
                    merged = self._structured_merge("company_notes", existing, value)
                    updates["company"]["Confidential Notes"] = merged
                else:
                    # Prefer already-merged session value; fall back to Airtable — never discard existing info
                    existing = str(updates["company"].get(field) or airtable_company_fields.get(field) or "")
                    updates["company"][field] = (existing + "\n" + value).strip() if existing else value

    def _structured_merge(self, schema_type: str, existing: str, new_info: str) -> str:
        """
        Merge *new_info* into *existing* using the structured schema for the given type.
        Falls back to simple concatenation if the Claude call fails.
        """
        prompt = build_structured_merge_prompt(schema_type, existing, new_info)
        try:
            result = self._call_claude(
                [{"role": "user", "content": prompt}],
                max_tokens=512,
                system="You are a concise database editor. Follow instructions exactly.",
            )
            return result.strip()
        except Exception:
            logger.debug("structured merge failed; falling back to concat")
            return (existing + "\n" + new_info).strip() if existing else new_info

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

    def _call_claude(self, messages: list[dict], max_tokens: int = 1024, system: str = None) -> str:
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system or SYSTEM_PROMPT,
            messages=messages,
        )
        return response.content[0].text
