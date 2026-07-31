"""
llm_providers.py
LLM provider abstraction used by Agent2's LLM-based conceptual-validation
check (agent2.py: check_documents_with_llm).

Providers:
  - DeloitteAgentProvider   (primary)
  - OllamaProvider          (fallback)

complete_with_fallback(prompt) tries the Deloitte Agent first and
automatically falls back to Ollama on any connection error, timeout, or
HTTP failure from Deloitte.

Configuration — environment variables (see .env.example, none of this is
hardcoded):

  DELOITTE_AGENT_URL          Primary GenW/Deloitte Agent completion endpoint.
  GENW_AGENT_URL              Alternate name for the same primary endpoint.
  LLM_CHECK_AGENT_URL         Alternate name for the same primary endpoint.
  DELOITTE_API_KEY            API key / bearer token for the primary endpoint.
  DELOITTE_TIMEOUT_SECONDS    Optional request timeout in seconds (default 120).

  OLLAMA_URL                  Base URL of the Ollama server (default: http://localhost:11434).
  OLLAMA_MODEL                Model name for the fallback (default: "llama3").
  OLLAMA_TIMEOUT_SECONDS       Optional request timeout in seconds (default 180).

NOTE on DeloitteAgentProvider.complete(): the exact request/response
contract of the internal "Deloitte Agent" API was not provided. The request
body ({"prompt": ...}) and the response-field probing (completion/response/
text/output/answer) are best-effort placeholders — update
DeloitteAgentProvider.complete() once the real API contract is known.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import time
from abc import ABC, abstractmethod
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    load_dotenv = None

print("[TRACE] Loaded llm_providers from:", __file__)

# Load backend-local environment variables (e.g. 24-06/.env) when present.
_DOTENV_CANDIDATES = [
    Path.cwd() / ".env",
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path(__file__).resolve().parent.parent / "24-06" / ".env",
]
for _candidate in _DOTENV_CANDIDATES:
    if load_dotenv is not None and _candidate.exists():
        load_dotenv(_candidate, override=False)
        break


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot produce a completion."""


class LLMProvider(ABC):
    name = "base"

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the raw text completion for `prompt`. Raise LLMProviderError on failure."""


class DeloitteAgentProvider(LLMProvider):
    """Primary provider. Configure via DELOITTE_AGENT_URL / GENW_AGENT_URL / LLM_CHECK_AGENT_URL."""

    name = "deloitte_agent"

    def __init__(self):
        self.url = ""
        for env_key in (
            "DELOITTE_AGENT_URL",
            "GENW_AGENT_URL",
            "LLM_CHECK_AGENT_URL",
        ):
            value = os.environ.get(env_key, "").strip()
            if value:
                self.url = value
                break
        self.api_key = os.environ.get("DELOITTE_API_KEY", "").strip()
        self.timeout = float(os.environ.get("DELOITTE_TIMEOUT_SECONDS", "120"))

    def is_configured(self) -> bool:
        return bool(self.url)

    def complete(self, prompt: str) -> str:
        if not self.url:
            raise LLMProviderError("DELOITTE_AGENT_URL is not configured.")

        payload = json.dumps({"prompt": prompt}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status >= 400:
                    raise LLMProviderError(f"Deloitte Agent returned HTTP {resp.status}")
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise LLMProviderError(f"Deloitte Agent HTTP error: {e.code} {e.reason}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            # Covers connection-refused, DNS failure, and socket timeouts.
            raise LLMProviderError(f"Deloitte Agent unreachable: {e}") from e

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return body

        for key in ("completion", "response", "text", "output", "answer"):
            if isinstance(data, dict) and key in data:
                return str(data[key])
        return json.dumps(data)


class OllamaProvider(LLMProvider):
    """Fallback provider. Configure via OLLAMA_URL / OLLAMA_MODEL."""

    name = "ollama"

    def __init__(self):
        self.base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = "llama3.2"
        self.timeout = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180"))

    def complete(self, prompt: str) -> str:
        print("[TRACE] Entered OllamaProvider.complete")
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Grammar-constrained JSON decoding: Ollama guarantees
            # syntactically valid JSON output in this mode, which avoids
            # the truncated/empty-response failure mode of free-form
            # generation and is usually faster to converge on.
            "format": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "rule_id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["PASS", "WARN", "FAIL"],
                        },
                        "evidence": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["index", "rule_id", "status", "evidence", "reasoning"],
                    "additionalProperties": False,
                },
            },
            # temperature 0 for deterministic, lower-latency decoding —
            # this is a rule-verdict task, not creative generation.
            "options": {"temperature": 0},
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        prompt_preview = prompt[:100] + ("..." if len(prompt) > 100 else "")
        req = urllib.request.Request(f"{self.base_url}/api/generate", data=payload, headers=headers, method="POST")
        try:
            start = time.time()
            print("[TRACE] About to send HTTP request to Ollama")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                print("[TRACE] HTTP request returned")
                print(f"[TRACE] Ollama request took {time.time() - start:.2f} seconds")
                if resp.status >= 400:
                    raise LLMProviderError(f"Ollama returned HTTP {resp.status}")
                body = resp.read().decode("utf-8")
                print("[TRACE] Response body read")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            raise LLMProviderError(f"Ollama HTTP error: {e.code} {e.reason}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise LLMProviderError(f"Ollama unreachable: {e}") from e
        except Exception as e:
            import traceback
            print("[OLLAMA ERROR]", repr(e))
            traceback.print_exc()
            raise

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            print("[TRACE] Leaving OllamaProvider.complete")
            return body
        print("[TRACE] Leaving OllamaProvider.complete")
        return str(data.get("response", body))


def complete_with_fallback(prompt: str) -> tuple[str, str]:
    """
    Try the Deloitte Agent first. On any connection error, timeout, or HTTP
    failure (or if it isn't configured), automatically fall back to Ollama.

    Returns (completion_text, provider_name_used).
    Raises LLMProviderError only if BOTH providers fail.
    """
    print("[TRACE] Entered complete_with_fallback")
    deloitte = DeloitteAgentProvider()
    errors: list[str] = []

    if deloitte.is_configured():
        print("[TRACE] Trying Deloitte provider")
        try:
            return deloitte.complete(prompt), deloitte.name
        except LLMProviderError as e:
            errors.append(str(e))
    else:
        errors.append("Deloitte Agent not configured (DELOITTE_AGENT_URL unset) — falling back to Ollama.")

    ollama = OllamaProvider()
    print("[TRACE] Trying Ollama provider")
    try:
        print(f"[TRACE] Prompt length: {len(prompt)} characters")
    
        result = ollama.complete(prompt), ollama.name
        print("[TRACE] Ollama provider returned successfully")
        return result
    except LLMProviderError as e:
        errors.append(str(e))

    print("[TRACE] All providers failed")
    raise LLMProviderError("All LLM providers failed: " + " | ".join(errors))
