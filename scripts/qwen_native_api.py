"""Opt-in Qwen-Agent adapter for native OpenAI-compatible tool calls.

Select ``model_type=oai_native_tools`` after importing this module. Legacy
tool parameter lists are translated at the request boundary, and optional
stream fragments are normalized for Qwen-Agent's string accumulator.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterator

from qwen_agent.llm.base import register_llm
from qwen_agent.llm.oai import TextChatAtOAI


def _native_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = deepcopy(tools)
    for tool in converted:
        function = tool.get("function", {})
        parameters = function.get("parameters")
        if not isinstance(parameters, list):
            continue
        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter in parameters:
            name = parameter.pop("name")
            if parameter.pop("required", False):
                required.append(name)
            properties[name] = parameter
        function["parameters"] = {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    return converted


def _normalize_stream(stream: Any) -> Iterator[Any]:
    try:
        for chunk in stream:
            for choice in chunk.choices:
                for tool_call in choice.delta.tool_calls or []:
                    function = tool_call.function
                    if function is not None:
                        if function.name is None:
                            function.name = ""
                        if function.arguments is None:
                            function.arguments = ""
            yield chunk
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()


@register_llm("oai_native_tools")
class NativeToolsChatAtOAI(TextChatAtOAI):
    """Use native tools without modifying Qwen-Agent or its legacy adapter."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__(cfg)
        self.use_raw_api = True
        create = self._chat_complete_create

        def native_create(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("tools") is not None:
                kwargs["tools"] = _native_tools(kwargs["tools"])
                kwargs.setdefault("parallel_tool_calls", False)
            response = create(*args, **kwargs)
            return _normalize_stream(response) if kwargs.get("stream") else response

        self._chat_complete_create = native_create
