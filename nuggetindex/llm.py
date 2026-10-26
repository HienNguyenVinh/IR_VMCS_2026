"""LLM client for Stage 1 candidate extraction with disk caching.

Supports OpenAI and Gemini (via OpenAI-compatible endpoint).
Uses local disk cache (cache/llm_cache.json) to ensure reproducible,
instant, and zero-cost runs after initial ingestion.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_env_file(filepath: str | Path = ".env") -> None:
    """Simple .env loader if dotenv is not loaded."""
    p = Path(filepath)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


class LLMClient:
    """OpenAI-compatible client with disk-backed cache."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        cache_path: str | Path = "cache/llm_cache.json",
    ):
        load_env_file()
        self.api_key = (
            api_key
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        
        # Configure endpoint defaults
        gemini_key = os.environ.get("GEMINI_API_KEY")
        default_base_url = None
        default_model = "gpt-4o-mini"
        if gemini_key and (self.api_key == gemini_key or not os.environ.get("OPENAI_API_KEY")):
            default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            default_model = "gemini-2.5-flash"

        self.base_url = base_url or os.environ.get("LLM_BASE_URL") or default_base_url
        self.model = model or os.environ.get("LLM_MODEL") or default_model
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, str] = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _cache_key(self, prompt: str, system_prompt: str) -> str:
        raw = f"{self.model}:::{system_prompt}:::{prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def complete(self, prompt: str, system_prompt: str = "") -> str:
        """Call LLM or return cached response."""
        ck = self._cache_key(prompt, system_prompt)
        if ck in self.cache:
            return self.cache[ck]

        if not self.api_key:
            raise RuntimeError(
                "No API key provided. Set GEMINI_API_KEY, OPENAI_API_KEY, or LLM_API_KEY in .env or environment."
            )

        # Call API using openai client or urllib
        response_text = self._call_api(prompt, system_prompt)
        self.cache[ck] = response_text
        self._save_cache()
        return response_text

    def _call_api(self, prompt: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try using openai package if available
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            # Use json_object response format when possible
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
            except Exception:
                # Some models or proxies might not support response_format
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except ImportError:
            # Fallback to requests/urllib
            import urllib.request
            url = (self.base_url or "https://api.openai.com/v1/").rstrip("/") + "/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()

    def complete_json(self, prompt: str, system_prompt: str = "") -> Any:
        """Call LLM and parse JSON response safely."""
        text = self.complete(prompt, system_prompt)
        # Strip potential markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback regex extraction of JSON object/array
            match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise ValueError(f"Could not parse JSON from LLM output:\n{text}")

