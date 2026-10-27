"""Ollama model wrapper. Each call is an independent single-turn request."""
import time
from datetime import datetime
from typing import Optional

import ollama

from .config import ModelConfig


class OllamaModel:
    def __init__(self, config: ModelConfig):
        self.config = config

    def generate(self, prompt: str, seed: Optional[int] = None, fmt: Optional[str] = None) -> dict:
        c = self.config
        options = {
            "temperature": c.temperature,
            "top_p": c.top_p,
            "num_predict": c.max_tokens,
        }
        if seed is not None:
            options["seed"] = seed
        start = time.time()
        try:
            resp = ollama.chat(
                model=c.model_path,
                messages=[{"role": "user", "content": prompt}],
                options=options,
                think=c.think,
                format=fmt,
            )
            return {
                "text": resp["message"]["content"],
                "input_tokens": resp.get("prompt_eval_count", 0),
                "output_tokens": resp.get("eval_count", 0),
                "latency_ms": (time.time() - start) * 1000,
                "timestamp": datetime.now().isoformat(),
                "finish_reason": resp.get("done_reason", "unknown"),
            }
        except Exception as e:  # keep the failure in the record
            return {
                "text": "",
                "error": str(e),
                "latency_ms": (time.time() - start) * 1000,
                "timestamp": datetime.now().isoformat(),
                "finish_reason": "error",
            }
