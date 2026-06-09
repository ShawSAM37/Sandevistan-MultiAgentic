from __future__ import annotations

from backend.config import settings
from backend.memory.in_memory_repository import get_memory_repository as get_in_memory_repository
from backend.memory.repository import ConversationMemoryRepository


def get_memory_repository() -> ConversationMemoryRepository:
    backend = (settings.memory_backend or "in_memory").strip().lower()

    if backend in {"in_memory", "memory", "local"}:
        return get_in_memory_repository()

    if backend in {"azure_table", "table", "azure_tables"}:
        from backend.memory.azure_table_repository import get_azure_table_memory_repository

        return get_azure_table_memory_repository()

    raise ValueError(f"Unsupported MEMORY_BACKEND: {settings.memory_backend}")
