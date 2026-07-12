from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


ChatRole = Literal["user", "assistant", "system"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    timestamp: str = Field(default_factory=utc_now_iso)
    requestId: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActiveConversationContext(BaseModel):
    machine: str | None = None
    baseMachine: str | None = None
    serialNumber: str | None = None
    manualType: str | None = None
    component: str | None = None
    intent: str | None = None


class ConversationMemory(BaseModel):
    threadId: str
    conversationSummary: str = ""
    recentTurns: list[ChatMessage] = Field(default_factory=list)
    activeContext: ActiveConversationContext = Field(default_factory=ActiveConversationContext)
    createdAt: str = Field(default_factory=utc_now_iso)
    updatedAt: str = Field(default_factory=utc_now_iso)
