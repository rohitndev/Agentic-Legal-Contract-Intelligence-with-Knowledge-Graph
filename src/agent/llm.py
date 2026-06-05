"""LLM client.

Wraps the Groq API (Mixtral 8x7B) when ``GROQ_API_KEY`` is set, and otherwise
falls back to a deterministic template engine so risk narratives, clause
rewrites, and negotiation replies are always produced — no network required.
"""

from __future__ import annotations

from typing import Optional

from ..config import settings


class LLMClient:
    def __init__(self) -> None:
        self.backend = "template"
        self._client = None
        if settings.groq_enabled:
            self._init_groq()

    def _init_groq(self) -> None:  # pragma: no cover - requires API key
        try:
            from groq import Groq

            self._client = Groq(api_key=settings.groq_api_key)
            self.backend = "groq"
        except Exception:
            self._client = None
            self.backend = "template"

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> str:
        if self._client is not None:  # pragma: no cover - requires API key
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self._client.chat.completions.create(
                model=settings.groq_model, messages=messages, max_tokens=max_tokens, temperature=0.2
            )
            return resp.choices[0].message.content.strip()
        return self._template(prompt)

    # ------------------------------------------------------------------
    def _template(self, prompt: str) -> str:
        """Heuristic, offline narrative generator keyed off the prompt intent."""
        low = prompt.lower()
        if "rewrite" in low or "redline" in low:
            return (
                "Suggested redline: add a mutual liability cap equal to fees paid in the "
                "preceding 12 months, convert unilateral rights to mutual notice-based "
                "rights, and carve out indemnity for gross negligence and willful misconduct."
            )
        if "summar" in low:
            return (
                "These clauses concentrate material legal exposure and should be "
                "negotiated as a package to limit aggregate downside."
            )
        if "narrative" in low or "explain" in low:
            return (
                "This clause materially shifts risk toward the reviewing party. Industry "
                "practice favours a negotiated cap and mutual obligations."
            )
        return (
            "Based on the contract context, the proposed term increases risk; a balanced, "
            "mutual formulation with defined limits is recommended."
        )
