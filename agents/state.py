"""
In-memory conversation state manager for the Insights agent.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Phase(Enum):
    IDENTIFY = auto()        # Waiting for user to name a company/role
    CONFIRMING = auto()      # Showing "is this the role you mean?" before synopsis
    AWAITING_SHARE = auto()  # Basic mode: waiting for user to share what they know
    COMPANY_FOUND = auto()   # Synopsis shown, answering follow-ups
    ROLE_FOUND = auto()      # Role synopsis shown, answering follow-ups


@dataclass
class ConversationState:
    user_id: str
    user_name: str = ""
    mode: str = "premium"    # "free", "pro", or "premium"

    phase: Phase = Phase.IDENTIFY

    company_record_id: Optional[str] = None
    company_name: Optional[str] = None
    role_record_id: Optional[str] = None
    role_title: Optional[str] = None

    # Suggested field updates accumulated from the conversation (not written to Airtable)
    suggested_updates: dict = field(default_factory=dict)

    # Full message history for Claude
    messages: list = field(default_factory=list)

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})


class StateManager:
    def __init__(self):
        self._store: dict[str, ConversationState] = {}

    def get(self, user_id: str) -> Optional[ConversationState]:
        return self._store.get(user_id)

    def get_or_create(self, user_id: str, user_name: str = "", mode: str = "premium") -> ConversationState:
        if user_id not in self._store:
            self._store[user_id] = ConversationState(
                user_id=user_id,
                user_name=user_name,
                mode=mode,
            )
        return self._store[user_id]

    def reset(self, user_id: str, user_name: str = "", mode: str = "premium") -> ConversationState:
        state = ConversationState(user_id=user_id, user_name=user_name, mode=mode)
        self._store[user_id] = state
        return state

    def delete(self, user_id: str) -> None:
        self._store.pop(user_id, None)
