from __future__ import annotations

import os
from typing import Optional


class OptionalOpenAIClient:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")

    def is_available(self) -> bool:
        return self.enabled and bool(self.api_key)

    def generate_markdown(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.is_available():
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return getattr(response, "output_text", None)
        except Exception as exc:  # pragma: no cover - network path
            return f"OpenAI commentary unavailable: {exc}"

