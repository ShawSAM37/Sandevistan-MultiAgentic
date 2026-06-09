from __future__ import annotations

import threading

from backend.memory.models import ChatMessage, ConversationMemory, utc_now_iso
from backend.memory.repository import ConversationMemoryRepository


class InMemoryConversationMemoryRepository(ConversationMemoryRepository):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, ConversationMemory] = {}

    def get_memory(self, thread_id: str) -> ConversationMemory:
        with self._lock:
            memory = self._store.get(thread_id)

            if memory is None:
                memory = ConversationMemory(threadId=thread_id)
                self._store[thread_id] = memory

            return memory.model_copy(deep=True)

    def save_memory(self, memory: ConversationMemory) -> None:
        memory.updatedAt = utc_now_iso()

        with self._lock:
            self._store[memory.threadId] = memory.model_copy(deep=True)

    def append_turns(
        self,
        thread_id: str,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        max_recent_turns: int,
    ) -> ConversationMemory:
        with self._lock:
            memory = self._store.get(thread_id)

            if memory is None:
                memory = ConversationMemory(threadId=thread_id)

            turns = list(memory.recentTurns)
            turns.extend([user_message, assistant_message])

            if max_recent_turns > 0:
                turns = turns[-max_recent_turns:]

            memory.recentTurns = turns
            memory.updatedAt = utc_now_iso()

            self._store[thread_id] = memory.model_copy(deep=True)

            return memory.model_copy(deep=True)

    def clear_memory(self, thread_id: str) -> None:
        with self._lock:
            self._store.pop(thread_id, None)


_MEMORY_REPOSITORY = InMemoryConversationMemoryRepository()


def get_memory_repository() -> InMemoryConversationMemoryRepository:
    return _MEMORY_REPOSITORY
