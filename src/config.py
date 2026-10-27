"""Experiment configuration (models adjusted for local hardware, see 08 guide)."""
import os
from dataclasses import dataclass, field
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelConfig:
    name: str
    model_path: str          # Ollama tag
    think: bool = False      # qwen3 must run with thinking disabled
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512


@dataclass
class ExperimentConfig:
    models: List[ModelConfig] = field(default_factory=lambda: [
        ModelConfig("Qwen25", "qwen2.5:3b"),
        ModelConfig("Qwen3", "qwen3:1.7b", think=False),
    ])
    n_repeats_dev: int = 2
    n_repeats_main: int = 3
    n_repeats_adversarial: int = 3
    random_seed: int = 42
    data_dir: str = os.path.join(ROOT, "data")
    results_dir: str = os.path.join(ROOT, "results")


DEFAULT_CONFIG = ExperimentConfig()
