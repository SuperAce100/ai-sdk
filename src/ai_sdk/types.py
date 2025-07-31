"""
Pydantic-based types mirroring the AI SDK Core *generateText* specification.

All models inherit from `_SDKBaseModel` which provides two conveniences:

1. `model_config = ConfigDict(frozen=True)` – makes all models immutable to
   mirror the behaviour of frozen dataclasses previously used.
2. `to_dict()` – wraps `model_dump(exclude_none=True)` and applies the explicit
   camelCase aliases required by the TypeScript SDK.

The public API of these classes stays **exactly** the same compared to the
previous dataclass-based implementation, so no changes are required in
existing downstream code.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Helper – explicit camelCase alias mapping (kept identical to old version)
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
        "source_type": "sourceType",
    }
    return {mapping.get(k, k): v for k, v in data.items() if v is not None}


class _SDKBaseModel(BaseModel):
    """Shared functionality for all SDK Pydantic models."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    def to_dict(self) -> dict[str, Any]:  # noqa: D401
        """Return a JSON-serialisable representation with camelCase aliases."""

        return _alias(self.model_dump(exclude_none=True))


# ---------------------------------------------------------------------------
# Message parts
# ---------------------------------------------------------------------------


class TextPart(_SDKBaseModel):
    text: str
    type: Literal["text"] = "text"


class ImagePart(_SDKBaseModel):
    image: Union[str, bytes]
    mime_type: Optional[str] = None
    type: Literal["image"] = "image"


class FilePart(_SDKBaseModel):
    data: Union[str, bytes]
    mime_type: str
    type: Literal["file"] = "file"


class ReasoningPart(_SDKBaseModel):
    text: str
    signature: Optional[str] = None
    type: Literal["reasoning"] = "reasoning"


class RedactedReasoningPart(_SDKBaseModel):
    data: str
    type: Literal["redacted-reasoning"] = "redacted-reasoning"


class ToolCallPart(_SDKBaseModel):
    tool_call_id: str
    tool_name: str
    args: Dict[str, Any]
    type: Literal["tool-call"] = "tool-call"


class ToolResultPart(_SDKBaseModel):
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: Optional[bool] = None
    type: Literal["tool-result"] = "tool-result"


# Convenience unions ---------------------------------------------------------

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


class CoreMessage(_SDKBaseModel):
    """Base class – concrete subclasses provide a fixed ``role`` value."""

    role: str

    # Each concrete subclass overrides *to_dict* because the shape of the
    # ``content`` field differs depending on the role.
    def to_dict(self) -> dict[str, Any]:  # noqa: D401
        raise NotImplementedError


class CoreSystemMessage(CoreMessage):
    content: str
    role: Literal["system"] = "system"

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


class CoreUserMessage(CoreMessage):
    content: Union[str, List[AnyUserContentPart]]
    role: Literal["user"] = "user"

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.content, list):
            conv = [part.to_dict() for part in self.content]
        else:
            conv = self.content
        return {"role": self.role, "content": conv}


class CoreAssistantMessage(CoreMessage):
    content: Union[str, List[AnyAssistantContentPart]]
    role: Literal["assistant"] = "assistant"

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.content, list):
            conv = [part.to_dict() for part in self.content]
        else:
            conv = self.content
        return {"role": self.role, "content": conv}


class CoreToolMessage(CoreMessage):
    content: List[ToolResultPart]
    role: Literal["tool"] = "tool"

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": [p.to_dict() for p in self.content]}


AnyMessage = Union[
    CoreSystemMessage,
    CoreUserMessage,
    CoreAssistantMessage,
    CoreToolMessage,
]

# ---------------------------------------------------------------------------
# On-step callback result structure
# ---------------------------------------------------------------------------


class ResponseMetadata(_SDKBaseModel):
    """Lightweight view of a provider response used by the on_step callback."""

    id: str | None = None
    model: str | None = None
    timestamp: datetime | None = None
    headers: Optional[Dict[str, str]] = None
    body: Any = None


class TokenUsage(_SDKBaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCall(_SDKBaseModel):
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(_SDKBaseModel):
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    result: Any = None
    is_error: Optional[bool] = None


class ReasoningDetail(_SDKBaseModel):
    type: Optional[Literal["text", "redacted"]] = None
    text: Optional[str] = None
    data: Optional[str] = None
    signature: Optional[str] = None


class OnStepFinishResult(_SDKBaseModel):
    """Detailed information passed to *on_step* callbacks."""

    step_type: Literal["initial", "continue", "tool-result"]
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None
    text: str = ""
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None
    warnings: Optional[List[str]] = None
    response: Optional[ResponseMetadata] = None
    is_continued: bool = False
    provider_metadata: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Additional result-related types
# ---------------------------------------------------------------------------


class Source(_SDKBaseModel):
    id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    provider_metadata: Optional[Dict[str, Any]] = None
    source_type: Literal["url"] = "url"


class GeneratedFile(_SDKBaseModel):
    base64: Optional[str] = None
    uint8_array: Optional[bytes] = None
    mime_type: Optional[str] = None
