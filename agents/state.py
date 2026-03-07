"""
In-memory conversation state manager for the Insights agent.

Each user has a ConversationState that tracks:
- Which phase of the flow they are in
- Which company/role they are discussing
- The collected data so far
- The full message history for the Claude conversation
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class Phase(Enum):
    """Phases of the Insights agent conversation flow."""
    # Initial: ask what company/role the user is interested in
    IDENTIFY = auto()
    # Company found – show synopsis, ask for more info
    COMPANY_FOUND = auto()
    # Role found – show role briefing, ask for more info
    ROLE_FOUND = auto()
    # Neither found – collecting basic info to create new records
    CREATING_NEW = auto()
    # Collecting additional insights from the user
    COLLECTING_INSIGHTS = auto()
    # Conversation complete
    DONE = auto()


@dataclass
class ConversationState:
    user_id: str
    user_name: str = ""

    phase: Phase = Phase.IDENTIFY

    # Resolved entity IDs
    company_record_id: Optional[str] = None
    company_name: Optional[str] = None
    role_record_id: Optional[str] = None
    role_title: Optional[str] = None

    # Pending data to be written to Airtable once confirmed
    pending_company_fields: dict = field(default_factory=dict)
    pending_role_fields: dict = field(default_factory=dict)
    pending_insight_content: str = ""

    # Full message history for Claude (list of {"role": ..., "content": ...})
    messages: list = field(default_factory=list)

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})


class StateManager:
    """Simple in-process store; swap for Redis in production."""

    def __init__(self):
        self._store: dict[str, ConversationState] = {}

    def get(self, user_id: str) -> Optional[ConversationState]:
        return self._store.get(user_id)

    def get_or_create(self, user_id: str, user_name: str = "") -> ConversationState:
        if user_id not in self._store:
            self._store[user_id] = ConversationState(
                user_id=user_id,
                user_name=user_name,
            )
        return self._store[user_id]

    def reset(self, user_id: str, user_name: str = "") -> ConversationState:
        """Start a fresh conversation for this user."""
        state = ConversationState(
            user_id=user_id,
            user_name=user_name,
        )
        self._store[user_id] = state
        return state

    def delete(self, user_id: str) -> None:
        self._store.pop(user_id, None)
