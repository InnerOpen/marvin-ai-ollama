"""Ollama's wire format: message and tool translation, errors, embeddings, model listing and pulls
(ported from Marvin core's tests/test_provider_tool_calling.py, now over httpx.MockTransport)."""

import json

import httpx
import pytest
from fake_server import FakeOllamaServer
from marvin_integration_sdk.ai import CompletionOptions, ImagePart, Message, ToolCall, ToolDefinition

from marvin_ai_ollama import OllamaProvider

TOOLS = [ToolDefinition(name="get_entry", description="Fetch an entry", input_schema={"type": "object"})]


@pytest.fixture
def server():
    return FakeOllamaServer()


@pytest.fixture
def provider(server):
    return OllamaProvider(http_client=server.client())


def test_it_translates_a_tool_round_trip():
    messages = [
        Message(role="assistant", content="", tool_calls=[ToolCall(id="call_0", name="get_entry", arguments={"slug": "about"})]),
        Message(role="tool", content='{"title":"About"}', tool_call_id="call_0"),
    ]
    api = OllamaProvider()._to_api_tool_messages(messages)
    # Ollama assistant tool_calls: function.arguments is an OBJECT (not a JSON string); the id is echoed
    assert api[0]["tool_calls"][0] == {"id": "call_0", "function": {"name": "get_entry", "arguments": {"slug": "about"}}}
    # the tool result is a plain role="tool" message pointing at its call
    assert api[1] == {"role": "tool", "content": '{"title":"About"}', "tool_call_id": "call_0"}


def test_a_tool_result_without_a_call_id_is_a_plain_tool_message():
    assert OllamaProvider()._to_api_tool_messages([Message(role="tool", content="x")]) == [{"role": "tool", "content": "x"}]


def test_complete_with_tools_synthesizes_ids():
    """Older Ollama servers return no call id."""
    seen = []

    def answer(request):
        seen.append(json.loads(request.content))
        body = {
            "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_entry", "arguments": {"slug": "home"}}}]},
            "prompt_eval_count": 20,
            "eval_count": 7,
        }
        return httpx.Response(200, json=body)

    provider = OllamaProvider(http_client=httpx.Client(transport=httpx.MockTransport(answer)))
    result = provider.complete_with_tools([Message(role="user", content="hi")], "llama3.1", TOOLS)
    assert result.tool_calls == [ToolCall(id="call_0", name="get_entry", arguments={"slug": "home"})]
    assert result.stop_reason == "tool_calls"
    assert result.total_tokens == 27
    # the tools were forwarded in the payload, and tool_choice (which Ollama has none of) was not
    assert seen[0]["tools"][0]["function"]["name"] == "get_entry" and "tool_choice" not in seen[0]


def test_ids_the_server_returns_are_kept(provider, server):
    server.reply_tool_calls([ToolCall(id="call_x1", name="get_entry", arguments={"slug": "home"})])
    assert provider.complete_with_tools([Message(role="user", content="hi")], "m", TOOLS).tool_calls[0].id == "call_x1"


def test_tool_arguments_given_as_a_json_string_are_decoded():
    body = {"message": {"content": "", "tool_calls": [{"id": "abc", "function": {"name": "get_entry", "arguments": '{"slug": "a"}'}}]}}
    provider = OllamaProvider(http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))))
    result = provider.complete_with_tools([Message(role="user", content="hi")], "m", TOOLS)
    assert result.tool_calls == [ToolCall(id="abc", name="get_entry", arguments={"slug": "a"})]


def test_a_plain_answer_has_no_tool_stop_reason(provider, server):
    server.reply_text("done")
    result = provider.complete_with_tools([Message(role="user", content="hi")], "m", TOOLS)
    assert (result.content, result.tool_calls, result.stop_reason) == ("done", [], "stop")


def test_images_travel_in_a_separate_list(provider, server):
    server.reply_text("a pixel")
    provider.complete([Message("user", ["What is this?", ImagePart(data="QUJD", mime_type="image/png")])], "llava")
    assert server.last_json()["messages"] == [{"role": "user", "content": "What is this?", "images": ["QUJD"]}]


def test_temperature_goes_in_options_and_structured_output_in_format(provider, server):
    schema = {"type": "object", "properties": {"tags": {"type": "array"}}}
    server.reply_text(json.dumps({"tags": []}))
    provider.complete_structured([Message("user", "Tag.")], "m", schema, CompletionOptions(temperature=0.2))
    sent = server.last_json()
    assert sent["format"] == schema and sent["options"] == {"temperature": 0.2} and sent["stream"] is False


def test_a_model_without_tool_support_gets_an_actionable_error(provider, server):
    server.reply_error(400, "registry.ollama.ai/library/gemma:latest does not support tools")
    with pytest.raises(RuntimeError, match="Pick a tool-capable model"):
        provider.complete_with_tools([Message("user", "hi")], "gemma", TOOLS)


def test_other_chat_errors_carry_ollamas_message(provider, server):
    server.reply_error(404, "model 'nope' not found")
    with pytest.raises(RuntimeError, match=r"HTTP 404\): model 'nope' not found"):
        provider.complete([Message("user", "hi")], "nope")


def test_embeddings_call_the_endpoint_once_per_text(provider, server):
    server.reply_embeddings([[1.0], [2.0]])
    assert provider.embed(["a", "b"], "nomic-embed-text") == [[1.0], [2.0]]
    assert [r.url.path for r in server.requests] == ["/api/embeddings", "/api/embeddings"]
    assert server.last_json() == {"model": "nomic-embed-text", "prompt": "b"}


def test_a_missing_embedding_model_says_how_to_pull_it(provider, server):
    server.reply_error(404, 'model "nomic-embed-text" not found, try pulling it first')
    with pytest.raises(RuntimeError, match="ollama pull nomic-embed-text"):
        provider.embed(["a"], "nomic-embed-text")


def test_pulling_a_model_forwards_progress(provider, server):
    server.reply_pull({"status": "pulling manifest"}, {"status": "downloading", "completed": 5, "total": 10}, {"status": "success"})
    seen = []
    provider.pull_model("llama3.2", on_progress=seen.append)
    assert [u["status"] for u in seen] == ["pulling manifest", "downloading", "success"]
    assert server.last_json() == {"model": "llama3.2", "stream": True}


def test_a_failed_pull_raises(provider, server):
    server.reply_pull({"status": "pulling manifest"}, {"error": "pull model manifest: file does not exist"})
    with pytest.raises(RuntimeError, match="file does not exist"):
        provider.pull_model("nope")


def test_test_connection_counts_models(provider, server):
    server.reply_models(["llama3.2", "qwen2.5"])
    assert provider.test_connection() == (True, "Connected — 2 models available")
