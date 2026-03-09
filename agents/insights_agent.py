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
    build_simple_field_merge_prompt,
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
        elif state.phase == Phase.DISAMBIGUATING:
            reply = self._handle_disambiguating(state, user_text)
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
        match_type = "none"

        # Fall back to the company already in state if the message didn't name one
        effective_company = company_name or state.company_name

        if role_title and effective_company:
            role_record, company_record, match_type = self.db.find_role_for_company(role_title, effective_company)
        elif role_title:
            role_record = self.db.find_role(role_title)
        elif company_name:
            company_record = self.db.find_company(company_name)

        if role_record and not company_record:
            linked = role_record["fields"].get("Company", [])
            if linked:
                company_record = self.db.get_company(linked[0])

        # For weak/no DB matches run semantic matching.
        # With 2+ company roles we use semantic as a *filter* (drop the clearly
        # unrelated) rather than a picker: any roles that survive go to
        # disambiguation so the user decides — we never silently pick one when
        # there is genuine ambiguity.
        if company_record and role_title and match_type in ("notes", "none"):
            company_roles = self.db.get_company_roles(company_record["id"])
            if company_roles:
                if len(company_roles) == 1:
                    # Only one role at the company — proceed with it.
                    role_record = company_roles[0]
                else:
                    # Multiple roles — filter out clearly unrelated ones, then
                    # disambiguate if 2+ remain (or use the sole survivor directly).
                    candidates = self._semantic_role_filter(role_title, company_roles)
                    if len(candidates) > 1:
                        return self._ask_disambiguate(state, company_record, candidates)
                    elif len(candidates) == 1:
                        role_record = candidates[0]
                    # 0 candidates: keep any notes hit, or fall through to not-found

        if not role_record and not company_record:
            entity = company_name or role_title
            return (
                f"I don't have \"{entity}\" in our database yet. "
                "Tell me what you know about it — what's the company, the role, "
                "and what have you learned from your conversations?"
            )

        # Premium: company found but no matching role → show other roles at this company.
        if not role_record and company_record and role_title and state.mode == "premium":
            return self._premium_role_not_found_response(state, company_record, role_title)

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

    # ------------------------------------------------------------------
    # Disambiguation helpers
    # ------------------------------------------------------------------

    def _ask_disambiguate(
        self, state: ConversationState, company_record: dict, candidates: list[dict]
    ) -> str:
        """
        Store candidate role IDs in state, set DISAMBIGUATING phase, and return a
        numbered question asking the user which role they meant.
        """
        state.company_record_id = company_record["id"]
        state.company_name = company_record["fields"].get("Company Name", "")
        raw_domain = company_record["fields"].get("Domain") or ""
        state.company_domain = self._ensure_https(raw_domain.strip())
        state.candidate_role_ids = [r["id"] for r in candidates]
        state.phase = Phase.DISAMBIGUATING

        co_ref = self._company_ref(state)
        lines = []
        for i, r in enumerate(candidates, 1):
            rf = r.get("fields", {})
            title = rf.get("Title", "Untitled")
            app_page = (rf.get("App Page") or "").strip()
            # Link the title to the App Page when available; always bold it.
            title_ref = f"[{title}]({app_page})" if app_page else title
            lines.append(f"{i}. **{title_ref}**")

        return (
            f"I found a couple of roles at {co_ref} that could be a match — "
            f"**which of these did you mean?**\n\n"
            + "\n".join(lines)
        )

    def _handle_disambiguating(self, state: ConversationState, user_text: str) -> str:
        """
        User has replied to the disambiguation question. Ask Claude which candidate
        they picked, then proceed as if that role was matched directly.
        """
        candidates = [
            self.db.find_role_by_id(rid)
            for rid in state.candidate_role_ids
            if rid
        ]
        candidates = [r for r in candidates if r]

        if not candidates:
            # Candidates expired — restart
            state.phase = Phase.IDENTIFY
            state.candidate_role_ids = []
            return self._handle_identify(state, user_text)

        # Build a summary list so Claude can resolve the user's selection
        summaries = []
        for i, r in enumerate(candidates, 1):
            rf = r.get("fields", {})
            summaries.append(f"{i}: {rf.get('Title', 'Untitled')}")

        prompt = (
            f'The user was asked to pick from these roles:\n'
            + "\n".join(summaries)
            + f'\n\nTheir reply: "{user_text}"\n\n'
            f'Which number did they pick? Return just the integer (1-based), '
            f'or null if unclear. Valid JSON only.'
        )
        chosen = None
        try:
            raw = self._call_claude([{"role": "user", "content": prompt}], max_tokens=8)
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            idx = json.loads(cleaned)
            if isinstance(idx, int) and 1 <= idx <= len(candidates):
                chosen = candidates[idx - 1]
        except Exception:
            logger.debug("_handle_disambiguating parse failed for %r", user_text)

        if not chosen:
            # Couldn't parse — re-ask
            co_ref = self._company_ref(state)
            lines = []
            for i, r in enumerate(candidates, 1):
                rf = r.get("fields", {})
                title = rf.get("Title", "Untitled")
                app_page = (rf.get("App Page") or "").strip()
                title_ref = f"[{title}]({app_page})" if app_page else title
                lines.append(f"{i}. **{title_ref}**")
            return (
                "Sorry, I didn't catch which one — "
                f"**could you pick a number or name from the list?**\n\n"
                + "\n".join(lines)
            )

        # Got a clear choice — proceed exactly as if the role was found directly
        state.candidate_role_ids = []
        state.role_record_id = chosen["id"]
        state.role_title = chosen["fields"].get("Title", "")
        state.role_app_page = (chosen["fields"].get("App Page") or "").strip()
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
        # Check if they're asking about a new entity — reset and re-identify if so
        # NOTE: this must run BEFORE the roles-listing check so that "roles we have at
        # MaintainX" (while currently discussing 11x) switches company context first.
        parsed = self._parse_company_and_role(user_text)
        new_company = parsed.get("company")
        new_role = parsed.get("role")
        current_company = (state.company_name or "").lower()
        current_role = (state.role_title or "").lower()

        # Require a minimum length to avoid pronouns / continuation words ("that", "it", "this")
        # being mistakenly treated as a new company or role name.
        _MIN_LEN = 4
        switching_company = (
            new_company
            and len(new_company) >= _MIN_LEN
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
            and len(new_role) >= _MIN_LEN
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

        # Role listing intent — now safe to check because company context is confirmed correct
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
            key_facts = {k: v for k, v in merged_role.items() if v and k in ("Title", "Function", "HM Name", "HQ Location", "Find", "Notes")}
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
            "roles do you have", "roles you have", "roles we have", "tell me roles",
            "tell me about the roles", "tell me the roles",
            "roles in our", "roles in the", "roles at", "roles for",
            "what positions", "any positions", "open positions",
            "what openings", "any openings",
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
            elif field == "Location":
                location_text = str(value)
                # Map to valid Airtable picklist values
                valid_options = self.db.get_location_options()
                if valid_options:
                    mapped = self._map_location_to_picklist(location_text, valid_options)
                    if mapped:
                        existing_loc = updates["role"].get("HQ Location") or airtable_role_fields.get("HQ Location") or []
                        if isinstance(existing_loc, str):
                            existing_loc = [existing_loc] if existing_loc else []
                        # Merge: add new values, preserve existing, deduplicate
                        merged_loc = list(dict.fromkeys(existing_loc + mapped))
                        updates["role"]["HQ Location"] = merged_loc
                # Always also capture the full detail text in Notes → Details
                existing_notes = updates["role"].get("Notes") or airtable_role_fields.get("Notes", "")
                merged_notes = self._structured_merge("role_notes", existing_notes, f"Location detail: {location_text}")
                updates["role"]["Notes"] = merged_notes
            else:
                # Prefer already-merged session value; fall back to Airtable — synthesize, never just concat
                existing = str(updates["role"].get(field) or airtable_role_fields.get(field) or "")
                updates["role"][field] = self._simple_merge(field, existing, value) if existing else value

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
                    # Prefer already-merged session value; fall back to Airtable — synthesize, never just concat
                    existing = str(updates["company"].get(field) or airtable_company_fields.get(field) or "")
                    updates["company"][field] = self._simple_merge(field, existing, value) if existing else value

    def _map_location_to_picklist(self, location_text: str, valid_options: list[str]) -> list[str]:
        """Map a free-text location description to one or more valid Airtable HQ Location picklist values."""
        options_str = "\n".join(f"- {o}" for o in valid_options)
        prompt = (
            f'The following location description comes from a job posting or conversation:\n'
            f'"{location_text}"\n\n'
            f'Match it to one or more values from this picklist:\n{options_str}\n\n'
            f'Return ONLY a JSON array of matching picklist values using the exact strings above. '
            f'If nothing fits, return []. No preamble.'
        )
        try:
            raw = self._call_claude([{"role": "user", "content": prompt}], max_tokens=128)
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(cleaned)
            if isinstance(result, list):
                return [v for v in result if v in valid_options]
        except Exception:
            logger.debug("_map_location_to_picklist failed for %r", location_text)
        return []

    def _semantic_role_filter(self, role_query: str, roles: list[dict]) -> list[dict]:
        """
        Filter *roles* to those plausibly related to *role_query*, used when there is
        no strong title match and 2+ roles exist at the company.

        Intentionally INCLUSIVE — we are filtering OUT clearly unrelated roles, not
        picking a single best match. Any role that is even loosely in the same
        function/domain is kept so the user can make the final call via disambiguation.
        Returns the filtered list (may be empty, 1, or many).
        """
        summaries = []
        for i, r in enumerate(roles):
            rf = r.get("fields", {})
            title = rf.get("Title", "Untitled")
            function = rf.get("Function", "")
            notes_snippet = (rf.get("Notes") or "")[:200]
            line = f"{i}: {title}"
            if function:
                line += f" ({function})"
            if notes_snippet:
                line += f" — {notes_snippet}"
            summaries.append(line)

        prompt = (
            f'A user in a GTM/RevOps professional community is asking about: "{role_query}"\n\n'
            f'Roles at this company:\n' + "\n".join(summaries) + "\n\n"
            f'Remove only roles that are CLEARLY in a completely different function '
            f'(e.g. a Finance, HR, or pure Engineering role when the query is about RevOps/GTM). '
            f'Keep everything that could plausibly be what the user means — including '
            f'adjacent roles like GTM AI, Revenue Intelligence, Sales/Marketing Ops, '
            f'GTM Strategy, or Growth. When in doubt, KEEP the role. '
            f'Return a JSON array of the indices to KEEP. '
            f'Return [] only if nothing is remotely related. Valid JSON only — no preamble.'
        )
        try:
            raw = self._call_claude([{"role": "user", "content": prompt}], max_tokens=32)
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(cleaned)
            if isinstance(result, list):
                kept = [roles[i] for i in result if isinstance(i, int) and 0 <= i < len(roles)]
                logger.info(
                    "_semantic_role_filter: %r → kept %s",
                    role_query,
                    [r["fields"].get("Title") for r in kept],
                )
                return kept
        except Exception:
            logger.debug("_semantic_role_filter failed for %r", role_query)
        # On failure, return all roles (safest — let user pick)
        return roles

    def _semantic_role_match(self, role_query: str, roles: list[dict]) -> list[dict]:
        """
        Ask Claude which role(s) in *roles* best match *role_query* using semantic understanding
        (e.g. "RevOps" → "Director of GTM Strategy/Ops" and/or "Head of GTM AI").

        Returns a list of matching role records (0 = no match, 1 = confident single match,
        2+ = multiple plausible matches that need user disambiguation).
        """
        summaries = []
        for i, r in enumerate(roles):
            rf = r.get("fields", {})
            title = rf.get("Title", "Untitled")
            function = rf.get("Function", "")
            notes_snippet = (rf.get("Notes") or "")[:200]
            line = f"{i}: {title}"
            if function:
                line += f" ({function})"
            if notes_snippet:
                line += f" — {notes_snippet}"
            summaries.append(line)

        prompt = (
            f'A user is looking for: "{role_query}"\n\n'
            f'Available roles:\n' + "\n".join(summaries) + "\n\n"
            f'Which indices plausibly match? Be INCLUSIVE — if a query is broad or uses '
            f'an umbrella term (e.g. "RevOps", "GTM", "Sales Ops") return ALL roles that '
            f'could reasonably be what the user meant, even if one feels like a stronger '
            f'match than another. Only exclude roles that are clearly unrelated. '
            f'Return a JSON array of integer indices. Return [] only if nothing fits at all. '
            f'Valid JSON only — no preamble.'
        )
        try:
            raw = self._call_claude([{"role": "user", "content": prompt}], max_tokens=32)
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = json.loads(cleaned)
            if isinstance(result, list):
                matched = [roles[i] for i in result if isinstance(i, int) and 0 <= i < len(roles)]
                logger.info(
                    "_semantic_role_match: %r → %s",
                    role_query,
                    [r["fields"].get("Title") for r in matched],
                )
                return matched
        except Exception:
            logger.debug("_semantic_role_match failed for %r", role_query)
        return []

    def _premium_role_not_found_response(
        self, state: ConversationState, company_record: dict, role_query: str
    ) -> str:
        """
        Premium fallback: the specific role wasn't found but the company is known.
        Set company context in state and show the other roles we're tracking there.
        """
        state.company_record_id = company_record["id"]
        state.company_name = company_record["fields"].get("Company Name", "")
        raw_domain = company_record["fields"].get("Domain") or ""
        state.company_domain = self._ensure_https(raw_domain.strip())
        state.phase = Phase.COMPANY_FOUND

        roles = self.db.get_company_roles(company_record["id"])
        co_ref = self._company_ref(state)

        if not roles:
            return (
                f"I couldn't find a **{role_query}** role at {co_ref} — "
                "and we don't have any other roles tracked there right now. "
                "Tell me what you know about it and I'll capture it."
            )

        open_roles = [r for r in roles if (r["fields"].get("Status") or "open").lower() != "closed"]
        display_roles = open_roles or roles  # fall back to all if everything is closed
        role_lines = []
        for r in display_roles:
            rf = r.get("fields", {})
            line = f"- **{rf.get('Title', 'Untitled')}**"
            if rf.get("Function"):
                line += f" ({rf['Function']})"
            role_lines.append(line)

        return (
            f"I couldn't find a **{role_query}** role at {co_ref}, "
            f"but here are the other roles we're tracking there:\n\n"
            + "\n".join(role_lines)
            + "\n\nIs one of these what you were looking for?"
        )

    def _simple_merge(self, field_name: str, existing: str, new_info: str) -> str:
        """
        Synthesize *new_info* into *existing* for a plain text field, preserving all unique
        facts while eliminating redundancy. Falls back to concatenation on failure.
        """
        if not existing:
            return new_info
        prompt = build_simple_field_merge_prompt(field_name, existing, new_info)
        try:
            result = self._call_claude(
                [{"role": "user", "content": prompt}],
                max_tokens=256,
                system="You are a concise database editor. Follow instructions exactly.",
            )
            return result.strip()
        except Exception:
            logger.debug("simple merge failed for field %r; falling back to concat", field_name)
            return (existing + "; " + new_info).strip()

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
            "Extract the company name and job role title from this message.\n"
            "Rules:\n"
            "- 'company': extract any business, organisation, or product name, even if "
            "lowercase or abbreviated (e.g. 'maintainx', 'openai', '11x', 'acme corp'). "
            "Return null only if no company is mentioned.\n"
            "- 'role': extract only a specific job title (e.g. 'VP of Sales', 'Head of GTM AI'). "
            "Do NOT extract generic words like 'role', 'roles', 'job', 'position', or "
            "pronouns like 'that', 'this', 'it', 'one'. Return null if no real job title is mentioned.\n"
            "Return a JSON object with keys 'company' and 'role'.\n\n"
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
