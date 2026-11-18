from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from openai.types.chat import ChatCompletionChunk
from qwen_agent.llm import get_chat_model
from qwen_agent.llm.schema import Message

from scripts.qwen_native_api import NativeToolsChatAtOAI


class _Stream:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.chunks)

    def close(self):
        self.closed = True


def _chunk(tool_calls=None, content=None):
    return ChatCompletionChunk(
        id="chat-smoke",
        created=0,
        model="test-qwen",
        object="chat.completion.chunk",
        choices=[{
            "index": 0,
            "delta": {"content": content, "tool_calls": tool_calls},
            "finish_reason": None,
        }],
    )


class NativeToolsAPITests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model": "test-qwen",
            "model_type": "oai_native_tools",
            "model_server": "http://unused.invalid/v1",
            "api_key": "unused-test-key",
        }

    def test_name_only_then_argument_deltas_accumulate_and_close(self):
        stream = _Stream([
            _chunk([{
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "save_to_memory", "arguments": None},
            }]),
            _chunk([{
                "index": 0,
                "function": {"name": None, "arguments": '{"note":'},
            }]),
            _chunk([{
                "index": 0,
                "function": {"arguments": '"remember X"}'},
            }]),
            _chunk(content="done"),
        ])
        client = Mock()
        client.chat.completions.create.return_value = stream
        with patch("qwen_agent.llm.oai.openai.OpenAI", return_value=client):
            model = get_chat_model(self.config)
            outputs = list(model._chat_stream(
                messages=[Message(role="user", content="Remember X.")],
                delta_stream=False,
                generate_cfg={},
            ))

        self.assertIsInstance(model, NativeToolsChatAtOAI)
        self.assertTrue(model.use_raw_api)
        final = outputs[-1]
        call = next(message for message in final if message.function_call)
        self.assertEqual(call.function_call.name, "save_to_memory")
        self.assertEqual(call.function_call.arguments, '{"note":"remember X"}')
        self.assertEqual(call.extra["function_id"], "call-1")
        self.assertEqual(next(message.content for message in final if message.content), "done")
        self.assertTrue(stream.closed)

    def test_legacy_parameters_converted_without_mutating_request(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "save_to_memory",
                "description": "Remember a note.",
                "parameters": [
                    {"name": "note", "type": "string", "description": "The note", "required": True},
                    {"name": "tag", "type": "string", "enum": ["personal"]},
                ],
            },
        }]
        original = deepcopy(tools)
        response = object()
        client = Mock()
        client.chat.completions.create.return_value = response
        with patch("qwen_agent.llm.oai.openai.OpenAI", return_value=client):
            model = NativeToolsChatAtOAI(self.config)
            actual = model._chat_complete_create(
                model="test-qwen", messages=[], tools=tools, stream=False
            )

        sent = client.chat.completions.create.call_args.kwargs["tools"]
        self.assertEqual(sent[0]["function"]["parameters"], {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "The note"},
                "tag": {"type": "string", "enum": ["personal"]},
            },
            "required": ["note"],
        })
        self.assertEqual(tools, original)
        self.assertIs(actual, response)
        self.assertFalse(client.chat.completions.create.call_args.kwargs["parallel_tool_calls"])

    def test_existing_json_schema_is_preserved(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "save_to_memory",
                "parameters": {
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                    "required": ["note"],
                    "additionalProperties": False,
                },
            },
        }]
        original = deepcopy(tools)
        client = Mock()
        with patch("qwen_agent.llm.oai.openai.OpenAI", return_value=client):
            model = NativeToolsChatAtOAI(self.config)
            model._chat_complete_create(model="test-qwen", messages=[], tools=tools, stream=False)
        self.assertEqual(client.chat.completions.create.call_args.kwargs["tools"], original)
        self.assertEqual(tools, original)


if __name__ == "__main__":
    unittest.main()
