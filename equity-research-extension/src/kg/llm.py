"""LLM clients for the KGQA evaluation (Colab): Gemini (google-genai) and Groq (OpenAI-compatible API, hosts open-weight
models such as openai/gpt-oss-120b and Qwen). Every reply is cached on disk by (model, prompt), so a rerun costs
nothing and an interrupted run resumes where it stopped. API keys come from the caller (Colab secrets) and are never
written to disk."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path


class LLMUnavailable(RuntimeError):
    """The API refused for good (daily quota used up, bad key, unknown model). Stops a run instead of being scored."""


_FATAL = re.compile(r"per ?day|perday|\bRPD\b|\bTPD\b|api key|invalid_api_key|model_not_found|does not exist|not found",
                    re.I)


class _Cached:
    name = "llm"

    def __init__(self, model: str, cache_dir, pause: float, temperature: float):
        self.model, self.pause, self.temperature = model, pause, temperature
        self.cache = Path(cache_dir) / f"{model.replace('/', '__')}.jsonl"
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.memo = {}
        if self.cache.exists():
            for line in self.cache.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.memo[r["key"]] = r["text"]
        self.calls = 0

    def _generate(self, prompt: str, json_mode: bool) -> str:
        raise NotImplementedError

    def __call__(self, prompt: str, json_mode: bool = True, retries: int = 6) -> str:
        key = hashlib.sha256(f"{self.model}\n{json_mode}\n{prompt}".encode()).hexdigest()
        if key in self.memo:
            return self.memo[key]
        wait = 15.0
        for attempt in range(retries):
            try:
                time.sleep(self.pause)
                text = self._generate(prompt, json_mode) or ""
                break
            except Exception as e:  # noqa: BLE001 - per-minute limits and transient errors: back off and retry
                msg = str(e)
                if _FATAL.search(msg) or attempt == retries - 1:
                    raise LLMUnavailable(f"{self.model}: {msg[:400]}") from e
                print(f"  {type(e).__name__}: {msg[:120]} - retrying in {wait:.0f}s")
                time.sleep(wait)
                wait = min(wait * 2, 120)
        self.calls += 1
        self.memo[key] = text
        with self.cache.open("a") as f:
            f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
        return text


class Gemini(_Cached):
    """Google Gemini API. Free-tier daily limits are per model and small for Flash (20 requests/day seen)."""

    def __init__(self, api_key: str, model: str, cache_dir, pause: float = 4.0, temperature: float = 0.0):
        from google import genai  # pip install google-genai

        self.client = genai.Client(api_key=api_key)
        super().__init__(model, cache_dir, pause, temperature)

    def _generate(self, prompt, json_mode):
        from google.genai import types

        cfg = types.GenerateContentConfig(temperature=self.temperature,
                                          response_mime_type="application/json" if json_mode else "text/plain")
        return self.client.models.generate_content(model=self.model, contents=prompt, config=cfg).text


class Groq(_Cached):
    """Groq's OpenAI-compatible endpoint. Free tier, per model: about 30 requests/min, 1,000 requests/day,
    8K tokens/min, 200K tokens/day for gpt-oss-120b. A planner call is about 1K tokens, so the default pause keeps
    under 8K tokens/min."""

    def __init__(self, api_key: str, model: str, cache_dir, pause: float = 8.0, temperature: float = 0.0,
                 reasoning_effort: str | None = "low"):
        from openai import OpenAI  # pip install openai

        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.reasoning_effort = reasoning_effort if "gpt-oss" in model else None
        super().__init__(model, cache_dir, pause, temperature)

    def _generate(self, prompt, json_mode):
        kw = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        r = self.client.chat.completions.create(model=self.model, temperature=self.temperature,
                                                messages=[{"role": "user", "content": prompt}], **kw)
        text = r.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # Qwen thinking blocks
