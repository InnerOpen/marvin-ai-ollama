"""The SDK's AI provider conformance kit, over a fake Ollama server."""

import pytest
from fake_server import FakeOllamaServer
from marvin_integration_sdk.ai import Message
from marvin_integration_sdk.ai.testing import AIProviderContract

from marvin_ai_ollama import OllamaProvider, plugin


class TestOllama(AIProviderContract):
    @pytest.fixture
    def transport(self):
        return FakeOllamaServer()

    @pytest.fixture
    def provider(self, transport):
        return OllamaProvider(base_url="http://ollama.test:11434", http_client=transport.client())

    def test_it_talks_to_the_native_chat_api(self, provider, transport, model):
        transport.reply_text("hi")
        provider.complete([Message("user", "hi")], model)
        request = transport.requests[-1]
        assert (str(request.url), transport.last_json()["stream"]) == ("http://ollama.test:11434/api/chat", False)


def test_the_entry_point_declares_the_plugin():
    from importlib.metadata import entry_points

    eps = {ep.name: ep.load() for ep in entry_points(group="marvin.ai_providers")}
    assert eps["ollama"] is plugin
    assert (plugin.slug, plugin.provider) == ("ollama", OllamaProvider)
