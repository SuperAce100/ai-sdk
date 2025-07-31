from typing import Any, AsyncIterator, Dict, Optional, List

from pydantic import BaseModel

from ai_sdk import generate_object
from ai_sdk.providers.language_model import LanguageModel


# ---------------------------------------------------------------------------
# Dummy provider WITHOUT native generate_object (tests fallback path)
# ---------------------------------------------------------------------------


class DummyNoNative(LanguageModel):
    """Implements only generate_text / stream_text to trigger JSON-parsing fallback."""

    def __init__(self, return_json: str) -> None:
        self._return_json = return_json
        self.last_system: Optional[str] = None
        self.last_messages: Optional[List[dict[str, Any]]] = None

    # --------------------------------------------------
    # LanguageModel interface – minimal stub implementation
    # --------------------------------------------------

    def generate_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Capture the system/messages arguments so tests can assert the schema instruction
        self.last_system = system
        self.last_messages = messages  # type: ignore[assignment]
        return {
            "text": self._return_json,
            "finish_reason": "stop",
        }

    def stream_text(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async def _gen():  # pragma: no cover – not needed here
            yield self._return_json

        return _gen()


# ---------------------------------------------------------------------------
# Dummy provider WITH native generate_object supported
# ---------------------------------------------------------------------------


class DummyNative(LanguageModel):
    """Simulates provider that supports native structured outputs."""

    def __init__(self, obj: BaseModel):
        self._obj = obj
        self.generate_object_called = False

    # ---- native structured output ----------------------------------------

    def generate_object(
        self,
        *,
        schema: type[BaseModel],
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self.generate_object_called = True
        return {
            "object": self._obj,
            "raw_text": self._obj.model_dump_json(),
            "finish_reason": "stop",
        }

    # ---- required abstract stubs -----------------------------------------

    def generate_text(self, *, prompt=None, system=None, messages=None, **kwargs):  # type: ignore[override]
        raise RuntimeError("Should not be called when native path exists")

    def stream_text(self, *, prompt=None, system=None, messages=None, **kwargs):  # type: ignore[override]
        async def _gen():
            yield ""

        return _gen()


# ---------------------------------------------------------------------------
# Shared schema
# ---------------------------------------------------------------------------


class AckSchema(BaseModel):
    ack: str


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_object_fallback_injects_schema_instruction():
    """When provider lacks generate_object, the helper should inject schema prompt."""

    dummy_json = '{"ack": "yes"}'
    provider = DummyNoNative(return_json=dummy_json)

    result = generate_object(model=provider, schema=AckSchema, prompt="irrelevant")

    assert isinstance(result.object, AckSchema)
    assert result.object.ack == "yes"

    # The system prompt should contain the JSON-schema instruction snippet.
    assert provider.last_system is not None
    assert "Respond ONLY with valid JSON" in provider.last_system
    # messages path not used in this test (prompt-only scenario)


def test_generate_object_native_path_used():
    """If provider implements generate_object, helper should use it and not fallback."""

    ack_instance = AckSchema(ack="ok")
    provider = DummyNative(obj=ack_instance)

    result = generate_object(model=provider, schema=AckSchema, prompt="does not matter")

    assert provider.generate_object_called is True
    assert result.object == ack_instance
