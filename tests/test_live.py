"""One optional smoke test against a real Ollama server. Skipped unless OLLAMA_BASE_URL is set; CI has none."""

import os

import pytest
from marvin_integration_sdk.ai import CompletionOptions, Message

from marvin_ai_ollama import OllamaProvider

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("OLLAMA_BASE_URL"), reason="OLLAMA_BASE_URL is not set")
def test_a_real_completion():
    provider = OllamaProvider(base_url=os.environ["OLLAMA_BASE_URL"])
    model = os.environ.get("OLLAMA_MODEL") or provider.list_models()[0]
    result = provider.complete([Message("user", "Reply with the single word: towel")], model, CompletionOptions(temperature=0))
    assert result.content.strip()
