# marvin-ai-ollama

[Ollama](https://ollama.com) for [Marvin](https://github.com/InnerOpen/marvin), as an AI provider plugin:
models you host yourself.

- chat, **tool calls** (whether a call yields tools depends on the loaded model: llama3.1+, qwen2.5…),
  **structured output** (Ollama's `format`) and **embeddings** (`/api/embeddings`, default model
  `nomic-embed-text`)
- **model pulls**: Marvin can download a model into your Ollama on demand (`/api/pull`)
- **self-hosted**: runs cost $0, so there is no price table
- plain HTTP through `httpx`; no vendor SDK. Images are not sent (`supports_vision` is false).

## Install

Marvin installs site-wide plugins at pod start from the chart's `plugins.packages` list. Only a
platform admin installs plugins; every workspace can then pick the provider in **Settings → AI**.

```yaml
plugins:
  packages:
    - https://github.com/InnerOpen/marvin-integration-sdk/archive/refs/heads/develop.tar.gz
    - https://github.com/InnerOpen/marvin-ai-ollama/archive/refs/heads/main.tar.gz
```

Requires Marvin with the AI provider contract (`marvin-integration-sdk` 0.8+). Marvin core still has a
built-in `ollama` provider; once this plugin is installed it **replaces** it (the log says `AI provider
plugin 'ollama' replaces core's built-in 'ollama'`) and reads the same settings, so nothing else
changes. **Admin → Plugins** lists `marvin-ai-ollama` with kind *AI provider*.

## Settings

| Setting | |
|---|---|
| `OLLAMA_BASE_URL` | Ollama's REST endpoint; default `http://localhost:11434`. A trailing `/api` is tolerated. |

No key is needed. A workspace can set its own base URL on its provider row.

## Develop

```sh
uv run --extra dev pytest -q
uv run --extra dev ruff check . && uv run --extra dev ruff format --check .
```

The tests never call a real Ollama. `tests/fake_server.py` is a fake Ollama server behind
`httpx.MockTransport` (the provider takes an optional `http_client`); the SDK's conformance kit
(`marvin_integration_sdk.ai.testing`) runs over it. `tests/test_live.py` is a smoke test against a
real server, skipped unless `OLLAMA_BASE_URL` is set (`OLLAMA_MODEL` picks the model).
