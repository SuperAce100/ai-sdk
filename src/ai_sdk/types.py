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

# ---------------------------------------------------------------------------
# Additional result-related types
# ---------------------------------------------------------------------------


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class Source:
    id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    provider_metadata: Optional[Dict[str, Any]] = None
    source_type: Literal["url"] = "url"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sourceType"] = d.pop("source_type")
        return _alias(d)


@dataclass
class GeneratedFile:
    base64: Optional[str] = None
    uint8_array: Optional[bytes] = None
    mime_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return _alias(asdict(self))


# Tool calls & results -------------------------------------------------------


@dataclass
class ToolCall:
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _alias(asdict(self))


@dataclass
class ToolResult:
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    result: Any = None
    is_error: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return _alias(asdict(self))


# Reasoning detail -----------------------------------------------------------


@dataclass
class ReasoningDetail:
    type: Optional[Literal["text", "redacted"]] = None
    text: Optional[str] = None
    data: Optional[str] = None
    signature: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return _alias(asdict(self))
