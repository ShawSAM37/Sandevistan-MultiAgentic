from __future__ import annotations

import json
from typing import Any

from azure.core.exceptions import ResourceNotFoundError
from azure.data.tables import TableServiceClient, UpdateMode

from backend.config import settings
from backend.memory.models import (
    ActiveConversationContext,
    ChatMessage,
    ConversationMemory,
    utc_now_iso,
)
from backend.memory.repository import ConversationMemoryRepository


MEMORY_ROW_KEY = "memory"


def _message_to_dict(message: ChatMessage) -> dict[str, Any]:
    return message.model_dump()


def _message_from_dict(data: dict[str, Any]) -> ChatMessage:
    return ChatMessage(**data)


class AzureTableConversationMemoryRepository(ConversationMemoryRepository):
    def __init__(
        self,
        connection_string: str,
        table_name: str,
    ) -> None:
        if not connection_string:
            raise ValueError("Azure Table connection string is required.")

        if not table_name:
            raise ValueError("Azure Table memory table name is required.")

        self._table_name = table_name
        self._service_client = TableServiceClient.from_connection_string(
            conn_str=connection_string
        )
        self._table_client = self._service_client.create_table_if_not_exists(
            table_name=table_name
        )

    def get_memory(self, thread_id: str) -> ConversationMemory:
        try:
            entity = self._table_client.get_entity(
                partition_key=thread_id,
                row_key=MEMORY_ROW_KEY,
            )
        except ResourceNotFoundError:
            memory = ConversationMemory(threadId=thread_id)
            self.save_memory(memory)
            return memory

        recent_turns_raw = entity.get("recentTurnsJson") or "[]"
        active_context_raw = entity.get("activeContextJson") or "{}"

        try:
            recent_turn_dicts = json.loads(recent_turns_raw)
        except json.JSONDecodeError:
            recent_turn_dicts = []

        try:
            active_context_dict = json.loads(active_context_raw)
        except json.JSONDecodeError:
            active_context_dict = {}

        return ConversationMemory(
            threadId=thread_id,
            conversationSummary=entity.get("conversationSummary") or "",
            recentTurns=[_message_from_dict(turn) for turn in recent_turn_dicts],
            activeContext=ActiveConversationContext(**active_context_dict),
            createdAt=entity.get("createdAt") or utc_now_iso(),
            updatedAt=entity.get("updatedAt") or utc_now_iso(),
        )

    def save_memory(self, memory: ConversationMemory) -> None:
        now = utc_now_iso()
        memory.updatedAt = now

        entity = {
            "PartitionKey": memory.threadId,
            "RowKey": MEMORY_ROW_KEY,
            "conversationSummary": memory.conversationSummary or "",
            "recentTurnsJson": json.dumps(
                [_message_to_dict(turn) for turn in memory.recentTurns],
                ensure_ascii=False,
            ),
            "activeContextJson": memory.activeContext.model_dump_json(),
            "createdAt": memory.createdAt,
            "updatedAt": memory.updatedAt,
        }

        self._table_client.upsert_entity(
            mode=UpdateMode.REPLACE,
            entity=entity,
        )

    def append_turns(
        self,
        thread_id: str,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        max_recent_turns: int,
    ) -> ConversationMemory:
        memory = self.get_memory(thread_id)

        turns = list(memory.recentTurns)
        turns.extend([user_message, assistant_message])

        if max_recent_turns > 0:
            turns = turns[-max_recent_turns:]

        memory.recentTurns = turns
        memory.updatedAt = utc_now_iso()

        self.save_memory(memory)

        return memory

    def clear_memory(self, thread_id: str) -> None:
        try:
            self._table_client.delete_entity(
                partition_key=thread_id,
                row_key=MEMORY_ROW_KEY,
            )
        except ResourceNotFoundError:
            return


_AZURE_TABLE_REPOSITORY: AzureTableConversationMemoryRepository | None = None


def get_azure_table_memory_repository() -> AzureTableConversationMemoryRepository:
    global _AZURE_TABLE_REPOSITORY

    if _AZURE_TABLE_REPOSITORY is None:
        _AZURE_TABLE_REPOSITORY = AzureTableConversationMemoryRepository(
            connection_string=settings.azure_table_connection_string or "",
            table_name=settings.azure_table_memory_table_name,
        )

    return _AZURE_TABLE_REPOSITORY
