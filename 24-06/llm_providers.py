"""
Compatibility-local LLM provider helper for the 24-06 backend.

This module mirrors the runtime contract expected by source_agent2.py and
provides a local import path without depending on the duplicate
Credit-Risk-Poc-main tree.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    load_dotenv = None

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
    name = "ollama"

    def __init__(self):
        self.base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.timeout = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180"))

    def complete(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "rule_id": {"type": "string"},
                        "status": {"type": "string", "enum": ["PASS", "WARN", "FAIL"]},
                        "evidence": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["index", "rule_id", "status", "evidence", "reasoning"],
                    "additionalProperties": False,
                },
            },
            "options": {"temperature": 0},
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        req = urllib.request.Request(f"{self.base_url}/api/generate", data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status >= 400:
                    raise LLMProviderError(f"Ollama returned HTTP {resp.status}")
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            raise LLMProviderError(f"Ollama HTTP error: {e.code} {e.reason}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise LLMProviderError(f"Ollama unreachable: {e}") from e
        except Exception:
            raise

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return body
        return str(data.get("response", body))


def complete_with_fallback(prompt: str) -> tuple[str, str]:
    deloitte = DeloitteAgentProvider()
    errors: list[str] = []

    if deloitte.is_configured():
        try:
            return deloitte.complete(prompt), deloitte.name
        except LLMProviderError as e:
            errors.append(str(e))
    else:
        errors.append("Deloitte Agent not configured (DELOITTE_AGENT_URL unset) — falling back to Ollama.")

    ollama = OllamaProvider()
    try:
        return ollama.complete(prompt), ollama.name
    except LLMProviderError as e:
        errors.append(str(e))

    raise LLMProviderError("All LLM providers failed: " + " | ".join(errors))
