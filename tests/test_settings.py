"""Credentials, prices and capabilities: what Marvin reads to build, bill and offer the provider."""

from marvin_integration_sdk.ai import masked_credentials, read_credentials

from marvin_ai_ollama import OllamaProvider


def test_it_reads_the_setting_core_reads():
    assert [c.env("ollama") for c in OllamaProvider.credentials] == ["OLLAMA_BASE_URL"]
    assert masked_credentials(OllamaProvider.credentials, {"base_url": "http://x:11434"}) == {"base_url": "http://x:11434"}


def test_the_base_url_defaults_to_localhost():
    assert OllamaProvider.from_credentials(read_credentials(OllamaProvider.credentials, {}))._base_url == "http://localhost:11434"
    assert OllamaProvider.from_credentials({"base_url": ""})._base_url == "http://localhost:11434"
    assert OllamaProvider.from_credentials({"base_url": "http://gpu-box:11434/"})._base_url == "http://gpu-box:11434"


def test_a_base_url_that_already_ends_in_api_is_not_doubled():
    assert OllamaProvider(base_url="http://host:11434/api")._base_url == "http://host:11434"
    assert OllamaProvider(base_url="http://host:11434/api/")._base_url == "http://host:11434"


def test_runs_are_free():
    assert OllamaProvider.self_hosted and OllamaProvider.prices == {}
    assert OllamaProvider.estimate_cost("llama3", 10, 10) == 0.0


def test_capabilities_match_what_core_offered():
    assert OllamaProvider.capabilities() == {"vision": False, "structured_output": True, "embeddings": True, "tool_calls": True, "model_pull": True}
    assert OllamaProvider.default_embedding_model == "nomic-embed-text"
