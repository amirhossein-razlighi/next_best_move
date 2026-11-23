"""LoRA adapters on the frozen DiT (peft), and the checkpoint files the paper's runs wrote.

A checkpoint directory holds pytorch_lora_weights.safetensors (diffusers' LoRA format: keys prefixed "transformer.") and, for
training checkpoints, state.pt (optimizer, LR schedule, step). The released weights are exactly these files.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

WEIGHTS_NAME = "pytorch_lora_weights.safetensors"


def weights_file(checkpoint: str | os.PathLike) -> Path:
    """The LoRA weights of a checkpoint directory (or the weights file itself)."""
    p = Path(checkpoint)
    return p if p.is_file() else p / WEIGHTS_NAME


def add_trainable_lora(transformer: torch.nn.Module, rank: int, alpha: float, targets: Sequence[str], dropout: float) -> None:
    """A fresh adapter for training: Gaussian init of A (B = 0), trainable parameters cast to float32."""
    transformer.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, init_lora_weights="gaussian",
                                       target_modules=list(targets)))
    cast_training_params(transformer, dtype=torch.float32)


def load_weights(transformer: torch.nn.Module, checkpoint: str | os.PathLike) -> None:
    """Load a checkpoint's LoRA tensors into the transformer's existing adapter; every LoRA parameter of the model must be in
    the file and the file must hold nothing else."""
    sd = {k.removeprefix("transformer."): v for k, v in load_file(str(weights_file(checkpoint))).items()}
    params = {n.replace(".default", ""): p for n, p in transformer.named_parameters() if "lora_" in n}
    if set(params) != set(sd):
        only_model, only_file = sorted(set(params) - set(sd)), sorted(set(sd) - set(params))
        raise ValueError(f"LoRA keys of {checkpoint} do not match the model's adapter: model only {only_model[:3]} "
                         f"({len(only_model)}), file only {only_file[:3]} ({len(only_file)}); check model.lora in the config")
    res = set_peft_model_state_dict(transformer, sd, adapter_name="default")
    assert not getattr(res, "unexpected_keys", None), res.unexpected_keys


def load_for_inference(transformer: torch.nn.Module, checkpoint: str | os.PathLike, rank: int, alpha: float,
                       targets: Sequence[str], dtype: torch.dtype) -> None:
    """Adapter with the run's rank / alpha / target modules, the checkpoint's weights, everything in the model dtype."""
    transformer.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, target_modules=list(targets)))
    load_weights(transformer, checkpoint)
    transformer.to(dtype)


def state_dict(transformer: torch.nn.Module) -> dict:
    return get_peft_model_state_dict(transformer)
