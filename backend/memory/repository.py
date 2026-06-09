from __future__ import annotations

from typing import Protocol

from backend.memory.models import ChatMessage, ConversationMemory


class ConversationMemoryRepository(Protocol):
    def get_memory(self, thread_id: str) -> ConversationMemory:
        ...

    def save_memory(self, memory: ConversationMemory) -> None:
        ...

    def append_turns(
        self,
        thread_id: str,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        max_recent_turns: int,
    ) -> ConversationMemory:
        ...

    def clear_memory(self, thread_id: str) -> None:
        ...
