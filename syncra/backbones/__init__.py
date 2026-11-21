"""Backbone registry. Each backbone runs in its own software environment (see requirements/)."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch

from .base import Backbone

REGISTRY = {
    "qwen2.5-omni": ("qwen_omni", "Qwen25Omni"),
    "qwen3-omni": ("qwen_omni", "Qwen3Omni"),
    "minicpm-o-4.5": ("minicpm_o", "MiniCPMo"),
    "nemotron-3-nano-omni": ("nemotron", "NemotronOmni"),
}


def load(name: str, model_path: str | Path, device: torch.device, train: bool = False) -> Backbone:
    module, cls = REGISTRY[name]
    return getattr(importlib.import_module(f"{__name__}.{module}"), cls)(model_path, device, train=train)
