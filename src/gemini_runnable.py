import os
from typing import Any, Optional

import requests
from pydantic import Field
from langchain_core.runnables import RunnableSerializable

try:
    import google.generativeai as genai
except Exception:
    genai = None


class GeminiRunnable(RunnableSerializable):
    """Call Google Generative API via REST using an API key.

    Returns a dict with 'content'.

    Environment variables:
    - `GOOGLE_API_KEY`: API key for HTTP requests (preferred for REST path).
    - `GEMINI_MODEL`: model id (default: `models/text-bison-001`).
    - `GEMINI_ENDPOINT`: optional base endpoint override.
    """
    
    model: str = Field(default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-pro"))
    temperature: float = Field(default=0.4)
    max_tokens: int = Field(default=500)
    api_key: Optional[str] = Field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY"))
    base: str = Field(default_factory=lambda: os.environ.get("GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta"))

    def _render_prompt(self, inp: Any) -> str:
        try:
            if isinstance(inp, dict):
                return str(inp.get("input") or inp)
            return str(inp)
        except Exception:
            return ""

    def invoke(self, inputs: Any, config: Optional[dict] = None) -> dict:
        prompt = self._render_prompt(inputs)

        if not self.api_key:
            return {"content": "[Gemini error] GOOGLE_API_KEY not set"}

        # Remove 'models/' prefix from self.model if present to avoid duplication
        model_name = self.model.removeprefix("models/")
        url = f"{self.base}/models/{model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(self.temperature),
                "maxOutputTokens": int(self.max_tokens),
            }
        }
        headers = {"Content-Type": "application/json"}

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {"content": f"[Gemini error] {e}"}

        # Gemini response: candidates[0].content.parts[0].text
        generated = ""
        if isinstance(data, dict):
            candidates = data.get("candidates", [])
            if candidates and isinstance(candidates, list) and len(candidates) > 0:
                first = candidates[0]
                if isinstance(first, dict):
                    content = first.get("content", {})
                    parts = content.get("parts", [])
                    if parts and len(parts) > 0:
                        generated = parts[0].get("text", "")
                    else:
                        generated = str(first)
                else:
                    generated = str(first)
            else:
                generated = str(data)

        return {"content": generated}
