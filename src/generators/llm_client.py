"""Multi-provider LLM client for BT generation.

Supports four providers behind a single ``LLMClient`` interface:

  1. ``openai``    — OpenAI direct API (GPT-4, GPT-4o, etc.)
  2. ``anthropic`` — Anthropic direct API (Claude family)
  3. ``compatible`` — any OpenAI-compatible HTTP server. This covers
                     Together, Fireworks, Groq, OpenRouter, Mistral, DeepSeek
                     and similar hosted services. The same code path also
                     works for local servers (vLLM, llama.cpp server,
                     LM Studio, Ollama with OpenAI compat) — just point
                     ``base_url`` at ``http://localhost:PORT/v1``.
  4. ``local``     — alias of ``compatible`` with a sensible local default
                     ``base_url`` so users can write ``provider: local``
                     and have it Just Work for the common case.

The provider can be selected explicitly via constructor arg, via the
``model.provider`` config field, or implicitly inferred from the model
name (e.g. names starting with "gpt-" → openai, "claude-" → anthropic).

Environment variables (.env):

  OPENAI_API_KEY        — for provider=openai
  ANTHROPIC_API_KEY     — for provider=anthropic

  Compatible/local providers:
  TOGETHER_API_KEY / FIREWORKS_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY
  COMPATIBLE_API_KEY    — generic fallback (set this if you use a custom server)
  COMPATIBLE_BASE_URL   — explicit base URL for the compatible provider
  LOCAL_BASE_URL        — explicit base URL for provider=local
                          default: http://localhost:8000/v1
  LOCAL_API_KEY         — most local servers ignore this; default: "local"
"""

from __future__ import annotations

import os
import re
import time
import unicodedata

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Prompt sanitization (defense against the OpenAI 400 JSON-parse edge case
# triggered by control chars in prior LLM step outputs).
# ---------------------------------------------------------------------------
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_for_json(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = _CTRL_CHARS_RE.sub("", text)
    text = text.encode("utf-8", "replace").decode("utf-8", "replace")
    return text


# ---------------------------------------------------------------------------
# Provider inference from model name
# ---------------------------------------------------------------------------
def _infer_provider_from_model(model_name: str) -> str | None:
    if not model_name:
        return None
    name = model_name.lower()
    if name.startswith(("gpt-", "o1-", "o3-", "o4-", "chatgpt")):
        return "openai"
    if name.startswith("claude-"):
        return "anthropic"
    # Llama / Mixtral / Qwen / DeepSeek / Mistral hosted via OpenAI-compatible endpoints
    if any(t in name for t in ("llama", "mixtral", "qwen", "deepseek", "mistral", "phi")):
        return "compatible"
    return None


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------
class _OpenAIBackend:
    """OpenAI direct API or any OpenAI-compatible server (Together, vLLM, ...)."""

    def __init__(self, api_key: str, base_url: str | None = None):
        from openai import OpenAI
        self._OpenAI = OpenAI
        from openai import APIError, BadRequestError
        self._APIError = APIError
        self._BadRequestError = BadRequestError
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, model, messages, temperature, max_tokens, seed, **extra):
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        kwargs.update(extra)
        response = self.client.chat.completions.create(**kwargs)
        return {
            "content": response.choices[0].message.content or "",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            "model": getattr(response, "model", model),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
        }

    def is_retryable_400(self, exc: Exception) -> bool:
        if not isinstance(exc, self._BadRequestError):
            return False
        return "could not parse the json body" in str(exc).lower()

    def is_retryable_transient(self, exc: Exception) -> bool:
        return isinstance(exc, self._APIError)


class _AnthropicBackend:
    """Anthropic Messages API."""

    def __init__(self, api_key: str):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Run `pip install anthropic` "
                "or omit provider=anthropic."
            ) from e
        self.client = Anthropic(api_key=api_key)
        from anthropic import APIError, BadRequestError
        self._APIError = APIError
        self._BadRequestError = BadRequestError

    def chat(self, model, messages, temperature, max_tokens, seed, **extra):
        # Anthropic separates system from user/assistant turns.
        system = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_msgs.append({"role": m["role"], "content": m["content"]})

        response = self.client.messages.create(
            model=model,
            system=system,
            messages=user_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Anthropic content is a list of TextBlock objects.
        text = "".join(getattr(b, "text", "") for b in response.content)
        usage = response.usage
        return {
            "content": text,
            "usage": {
                "prompt_tokens": getattr(usage, "input_tokens", 0),
                "completion_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0),
            },
            "model": response.model,
            "system_fingerprint": None,  # Anthropic does not expose this
        }

    def is_retryable_400(self, exc: Exception) -> bool:
        return False

    def is_retryable_transient(self, exc: Exception) -> bool:
        return isinstance(exc, self._APIError)


# ---------------------------------------------------------------------------
# Provider configuration helpers (env-driven defaults)
# ---------------------------------------------------------------------------
def _resolve_compatible_creds(provider_hint: str | None = None) -> tuple[str, str]:
    """Find an API key + base URL for an OpenAI-compatible endpoint.

    Order of resolution:
      1. COMPATIBLE_API_KEY + COMPATIBLE_BASE_URL (explicit, recommended)
      2. Well-known providers in order of preference: Together, Fireworks, Groq, OpenRouter
      3. raise — no creds found

    A provider_hint like "together" / "fireworks" / "groq" / "openrouter"
    forces a specific provider if its creds are present.
    """
    explicit_key = os.getenv("COMPATIBLE_API_KEY")
    explicit_url = os.getenv("COMPATIBLE_BASE_URL")
    if explicit_key and explicit_url:
        return explicit_key, explicit_url

    candidates = [
        ("together", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
        ("fireworks", "FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1"),
        ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
        ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
        ("mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1"),
        ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    ]
    if provider_hint:
        candidates = [c for c in candidates if c[0] == provider_hint.lower()]
    for _name, env_key, base_url in candidates:
        key = os.getenv(env_key)
        if key:
            return key, base_url

    raise RuntimeError(
        "No OpenAI-compatible provider credentials found. Set one of: "
        "COMPATIBLE_API_KEY+COMPATIBLE_BASE_URL, TOGETHER_API_KEY, "
        "FIREWORKS_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, "
        "MISTRAL_API_KEY, DEEPSEEK_API_KEY."
    )


def _resolve_local_creds() -> tuple[str, str]:
    """Local OpenAI-compatible server (vLLM, llama.cpp server, LM Studio, Ollama)."""
    base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("LOCAL_API_KEY", "local")
    return api_key, base_url


# ---------------------------------------------------------------------------
# Public LLMClient
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 4,
        retry_backoff: float = 2.0,
        seed: int | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4")
        self.temperature = (
            temperature if temperature is not None
            else float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        )
        self.max_tokens = max_tokens or int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.seed = seed if seed is not None else (
            int(os.getenv("LLM_SEED")) if os.getenv("LLM_SEED") else None
        )

        # Resolve provider: explicit > env > inferred from model name > openai default
        self.provider = (
            provider
            or os.getenv("LLM_PROVIDER")
            or _infer_provider_from_model(self.model)
            or "openai"
        ).lower()

        self.backend = self._build_backend(provider_arg=provider, base_url=base_url, api_key=api_key)

    def _build_backend(self, provider_arg, base_url, api_key):
        if self.provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            return _OpenAIBackend(api_key=key, base_url=base_url)

        if self.provider == "anthropic":
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            return _AnthropicBackend(api_key=key)

        if self.provider == "compatible":
            if api_key and base_url:
                key, url = api_key, base_url
            else:
                # If user passed --provider compatible without specifying creds,
                # try named providers from env.
                key, url = _resolve_compatible_creds(provider_hint=os.getenv("COMPATIBLE_PROVIDER"))
                if base_url:
                    url = base_url
                if api_key:
                    key = api_key
            return _OpenAIBackend(api_key=key, base_url=url)

        if self.provider == "local":
            default_key, default_url = _resolve_local_creds()
            return _OpenAIBackend(
                api_key=api_key or default_key,
                base_url=base_url or default_url,
            )

        raise ValueError(
            f"Unknown provider '{self.provider}'. "
            "Choose from: openai, anthropic, compatible, local"
        )

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """Send a prompt to the configured backend and return response metadata.

        Retries on transient errors and on the OpenAI HTTP 400
        "could not parse the JSON body" edge case (caused by stray control
        chars in prompts built from prior LLM step output).
        """
        system_prompt = _sanitize_for_json(system_prompt)
        user_prompt = _sanitize_for_json(user_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_err: Exception | None = None
        start = time.time()
        for attempt in range(self.max_retries):
            try:
                response = self.backend.chat(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                )
                break
            except Exception as e:
                last_err = e
                if self.backend.is_retryable_400(e) or self.backend.is_retryable_transient(e):
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(self.retry_backoff ** attempt)
                else:
                    raise
        else:
            raise last_err if last_err else RuntimeError("retries exhausted")

        elapsed = time.time() - start
        return {
            "content": response["content"],
            "bt_xml": self._extract_xml(response["content"]),
            "usage": response["usage"],
            "elapsed_seconds": round(elapsed, 2),
            "model": response["model"],
            "provider": self.provider,
            "seed": self.seed,
            "system_fingerprint": response.get("system_fingerprint"),
        }

    @staticmethod
    def _extract_xml(text: str) -> str | None:
        """Extract a Behavior Tree XML payload from an LLM response.

        The model may emit XML in several inconsistent forms; we try them
        in order of decreasing specificity:
          1. ```xml ... ``` fenced block (preferred form, asked for in prompt)
          2. ``` ... ``` plain fenced block whose content begins with <root or <?xml
          3. <?xml ...?> declaration anywhere in the text
          4. <root ...> ... </root> bare element
          5. <BehaviorTree ...> ... </BehaviorTree> bare element
        """
        if not text:
            return None

        m = re.search(r"```xml\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        for m in re.finditer(r"```\s*\n?(.*?)\n?```", text, re.DOTALL):
            block = m.group(1).strip()
            if block.startswith("<root") or block.startswith("<?xml") or "<BehaviorTree" in block:
                return block

        m = re.search(r"<\?xml.*?\?>\s*(<root\b.*?</root>)", text, re.DOTALL)
        if m:
            return m.group(0).strip()

        m = re.search(r"<root\b[^>]*>.*?</root>", text, re.DOTALL)
        if m:
            return m.group(0).strip()

        m = re.search(r"<BehaviorTree\b[^>]*>.*?</BehaviorTree>", text, re.DOTALL)
        if m:
            return m.group(0).strip()

        return None
