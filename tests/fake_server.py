"""A fake Ollama server for the conformance kit: the provider talks to it through ``httpx.MockTransport``,
so its requests and the answers it parses go through httpx exactly as against a real server."""

from __future__ import annotations

import json
from collections import deque

import httpx
from marvin_integration_sdk.ai import ToolCall
from marvin_integration_sdk.ai.fake import FakeTransport


class FakeOllamaServer(FakeTransport):
    """Queued neutral replies, answered in the wire shape of whichever endpoint the provider calls."""

    def __init__(self) -> None:
        self.replies: deque[dict] = deque()
        self.requests: list[httpx.Request] = []

    # ── the kit's vocabulary ─────────────────────────────────────────────────

    def reply_text(self, text: str, *, prompt_tokens: int = 11, completion_tokens: int = 7) -> None:
        self.replies.append({"text": text, "usage": (prompt_tokens, completion_tokens)})

    def reply_tool_calls(self, calls: list[ToolCall], *, prompt_tokens: int = 11, completion_tokens: int = 7) -> None:
        self.replies.append({"tool_calls": list(calls), "usage": (prompt_tokens, completion_tokens)})

    def reply_models(self, ids: list[str]) -> None:
        self.replies.append({"models": list(ids)})

    def reply_embeddings(self, vectors: list[list[float]]) -> None:
        self.replies.append({"embeddings": [list(v) for v in vectors]})

    def reply_error(self, status: int, message: str) -> None:
        self.replies.append({"error": (status, message)})

    def reply_pull(self, *updates: dict) -> None:
        """/api/pull streams one JSON object per line."""
        self.replies.append({"pull": list(updates)})

    def last_request(self) -> str:
        return self.requests[-1].content.decode() if self.requests else ""

    def last_json(self) -> dict:
        return json.loads(self.last_request() or "{}")

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    # ── the server ───────────────────────────────────────────────────────────

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.popleft() if self.replies else {}
        if "error" in reply:
            status, message = reply["error"]
            return httpx.Response(status, json={"error": message})
        path = request.url.path
        body = json.loads(request.content or b"{}") if request.method == "POST" else {}
        if path == "/api/chat":
            return httpx.Response(200, json=_chat(reply, body.get("model", "")))
        if path == "/api/embeddings":
            # One text per request: each call takes the next vector off the queued reply.
            vectors = reply.get("embeddings", [])
            if len(vectors) > 1:
                self.replies.appendleft({"embeddings": vectors[1:]})
            return httpx.Response(200, json={"embedding": vectors[0] if vectors else []})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": i, "model": i, "size": 0} for i in reply.get("models", [])]})
        if path == "/api/pull":
            return httpx.Response(200, content="".join(json.dumps(u) + "\n" for u in reply.get("pull", [])).encode())
        return httpx.Response(404, json={"error": f"no fake for {path}"})


def _chat(reply: dict, model: str) -> dict:
    pt, ct = reply.get("usage", (0, 0))
    # `arguments` is an object, not a JSON string. Current Ollama servers return an id per call.
    calls = [{"id": c.id, "function": {"name": c.name, "arguments": c.arguments}} for c in reply.get("tool_calls", [])]
    message = {"role": "assistant", "content": reply.get("text", ""), **({"tool_calls": calls} if calls else {})}
    return {
        "model": model,
        "created_at": "2026-01-01T00:00:00Z",
        "message": message,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": pt,
        "eval_count": ct,
    }
