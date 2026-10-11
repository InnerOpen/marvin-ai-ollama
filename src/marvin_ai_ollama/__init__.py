"""Ollama for Marvin: self-hosted models as an AI provider plugin.

One entry point in ``marvin.ai_providers``: ``ollama``. It replaces Marvin core's built-in provider of
the same slug and reads the same setting (``OLLAMA_BASE_URL``) or a workspace's own base URL.
"""

from marvin_integration_sdk.ai import AIProviderPlugin

from .provider import OllamaProvider

__all__ = ["OllamaProvider", "plugin"]
__version__ = "0.1.0"

plugin = AIProviderPlugin(
    slug="ollama", name="Ollama", provider=OllamaProvider, description="Models served by your own Ollama (chat, tools, embeddings, pulls)."
)
