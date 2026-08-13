from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
       
        kwargs = {"model": self.model, "input": prompt}
        if system_prompt is not None:
            kwargs["instructions"] = system_prompt

        response = self.client.responses.create(**kwargs)

        return response.output_text.strip()

    def generate_text_with_image(self, prompt: str, image_url: str) -> str:
    
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
        )

        return response.output_text.strip()