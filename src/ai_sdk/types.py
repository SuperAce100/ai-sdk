from __future__ import annotations

"""Dataclass-based types mirroring the AI SDK Core *generateText* specification.

These lightweight structures offer static typing without introducing runtime
validation overhead from third-party libraries.  Each dataclass implements
``to_dict`` for easy JSON-ready serialization (including camelCase aliases).
"""

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, List, Literal, Optional, Union, Dict, TypedDict
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Message parts
# ---------------------------------------------------------------------------


def _alias(data: dict[str, Any]) -> dict[str, Any]:
    """Convert snake_case keys with explicit aliases to camelCase variants."""
    mapping = {
        "mime_type": "mimeType",
        "tool_call_id": "toolCallId",
        "tool_name": "toolName",
        "args_text_delta": "argsTextDelta",
        "text_delta": "textDelta",
        "is_error": "isError",
    }
    return {mapping.get(k, k): v for k, v in data.items() if v is not None}


@dataclass
class TextPart:
    text: str
    type: Literal["text"] = "text"

    def to_dict(self) -> dict[str, Any]:
        return _alias(asdict(self))


@dataclass
class ImagePart:
    image: Union[str, bytes]
    mime_type: Optional[str] = field(default=None, metadata={"alias": "mimeType"})
    type: Literal["image"] = "image"

    def to_dict(self) -> dict[str, Any]:
        return _alias(asdict(self))


@dataclass
class FilePart:
    data: Union[str, bytes]
    mime_type: str = field(metadata={"alias": "mimeType"})
    type: Literal["file"] = "file"

    def to_dict(self) -> dict[str, Any]:
        return _alias(asdict(self))


@dataclass
class ReasoningPart:
    text: str
    signature: Optional[str] = None
    type: Literal["reasoning"] = "reasoning"

    def to_dict(self) -> dict[str, Any]:
        return _alias(asdict(self))


@dataclass
class RedactedReasoningPart:
    data: str
    type: Literal["redacted-reasoning"] = "redacted-reasoning"

    def to_dict(self) -> dict[str, Any]:
        return _alias(asdict(self))


@dataclass
class ToolCallPart:
    tool_call_id: str
    tool_name: str
    args: Dict[str, Any]
    type: Literal["tool-call"] = "tool-call"

    def to_dict(self) -> dict[str, Any]:
        return _alias(asdict(self))


@dataclass
class ToolResultPart:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: Optional[bool] = None
    type: Literal["tool-result"] = "tool-result"

    def to_dict(self) -> dict[str, Any]:
        return _alias(asdict(self))


# Convenience unions
AnyUserContentPart = Union[TextPart, ImagePart, FilePart]
AnyAssistantContentPart = Union[
    TextPart,
    ReasoningPart,
    RedactedReasoningPart,
    ToolCallPart,
]

# ---------------------------------------------------------------------------
# Core Message hierarchy
# ---------------------------------------------------------------------------


@dataclass
class CoreMessage:
    """Base class – concrete subclasses provide a fixed ``role`` value."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class CoreSystemMessage(CoreMessage):
    content: str
    role: Literal["system"] = "system"

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class CoreUserMessage(CoreMessage):
    content: Union[str, List[AnyUserContentPart]]
    role: Literal["user"] = "user"

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.content, list):
            conv = [part.to_dict() for part in self.content]
        else:
            conv = self.content
        return {"role": self.role, "content": conv}


@dataclass
class CoreAssistantMessage(CoreMessage):
    content: Union[str, List[AnyAssistantContentPart]]
    role: Literal["assistant"] = "assistant"

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.content, list):
            conv = [part.to_dict() for part in self.content]
        else:
            conv = self.content
        return {"role": self.role, "content": conv}


@dataclass
class CoreToolMessage(CoreMessage):
    content: List[ToolResultPart]
    role: Literal["tool"] = "tool"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": [part.to_dict() for part in self.content],
        }


AnyMessage = Union[
    CoreSystemMessage,
    CoreUserMessage,
    CoreAssistantMessage,
    CoreToolMessage,
]
