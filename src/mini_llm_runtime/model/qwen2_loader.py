from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file
import torch
import torch.nn as nn


def map_qwen2_checkpoint_key(checkpoint_key: str) -> str:
    """Map a Hugging Face Qwen2 parameter name to the local model name."""

    projection_names = ("q_proj", "k_proj", "v_proj")

    for projection_name in projection_names:
        source = f".self_attn.{projection_name}."
        target = f".self_attn.qkv_proj.{projection_name}."

        if source in checkpoint_key:
            return checkpoint_key.replace(source, target)

    return checkpoint_key

@dataclass(frozen=True)
class Qwen2CheckpointValidation:
    missing_keys: list[str]
    unexpected_keys: list[str]
    shape_mismatches: list[
        tuple[str, tuple[int, ...], tuple[int, ...]]
    ]

    @property
    def is_valid(self) -> bool:
        return not (
            self.missing_keys
            or self.unexpected_keys
            or self.shape_mismatches
        )


def validate_qwen2_checkpoint(
    checkpoint_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
    *,
    tie_word_embeddings: bool,
) -> Qwen2CheckpointValidation:
    """Validate mapped checkpoint names and shapes without loading weights."""
    present_local_keys: set[str] = set()
    unexpected_keys: list[str] = []
    shape_mismatches: list[
        tuple[str, tuple[int, ...], tuple[int, ...]]
    ] = []

    for checkpoint_key, checkpoint_tensor in checkpoint_state.items():
        local_key = map_qwen2_checkpoint_key(checkpoint_key)
        if local_key not in local_state:
            unexpected_keys.append(checkpoint_key)
            continue

        present_local_keys.add(local_key)
        checkpoint_shape = tuple(checkpoint_tensor.shape)
        local_shape = tuple(local_state[local_key].shape)
        if checkpoint_shape != local_shape:
            shape_mismatches.append(
                (local_key, checkpoint_shape, local_shape)
            )

    if tie_word_embeddings:
        embedding_key = "model.embed_tokens.weight"
        lm_head_key = "lm_head.weight"
        if embedding_key in present_local_keys:
            present_local_keys.add(lm_head_key)
        if lm_head_key in present_local_keys:
            present_local_keys.add(embedding_key)

    missing_keys = sorted(set(local_state) - present_local_keys)
    return Qwen2CheckpointValidation(
        missing_keys=missing_keys,
        unexpected_keys=sorted(unexpected_keys),
        shape_mismatches=sorted(shape_mismatches),
    )


def load_qwen2_checkpoint(
    model: nn.Module,
    checkpoint_state: Mapping[str, torch.Tensor],
    *,
    tie_word_embeddings: bool,
) -> Qwen2CheckpointValidation:
    """Strictly validate and copy an in-memory Qwen2 checkpoint."""
    local_state = model.state_dict()
    validation = validate_qwen2_checkpoint(
        checkpoint_state,
        local_state,
        tie_word_embeddings=tie_word_embeddings,
    )
    if not validation.is_valid:
        raise ValueError(
            "Invalid Qwen2 checkpoint: "
            f"missing_keys={validation.missing_keys}, "
            f"unexpected_keys={validation.unexpected_keys}, "
            f"shape_mismatches={validation.shape_mismatches}"
        )

    with torch.no_grad():
        for checkpoint_key, checkpoint_tensor in checkpoint_state.items():
            local_key = map_qwen2_checkpoint_key(checkpoint_key)
            local_state[local_key].copy_(checkpoint_tensor)

    return validation

def load_qwen2_safetensors(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    tie_word_embeddings: bool,
    device: str | torch.device = "cpu",
    ) -> Qwen2CheckpointValidation:
    """Load a local Qwen2 safetensors checkpoint into a model."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    checkpoint_state = load_file(
        str(checkpoint_path),
        device=str(device),
    )

    return load_qwen2_checkpoint(
        model,
        checkpoint_state,
        tie_word_embeddings=tie_word_embeddings,
    )


def validate_qwen2_safetensors_metadata(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    tie_word_embeddings: bool,
) -> Qwen2CheckpointValidation:
    """Validate safetensors keys and shapes without loading tensor data."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    local_state = model.state_dict()
    present_local_keys: set[str] = set()
    unexpected_keys: list[str] = []
    shape_mismatches: list[
        tuple[str, tuple[int, ...], tuple[int, ...]]
    ] = []

    with safe_open(
        str(checkpoint_path),
        framework="pt",
        device="cpu",
    ) as checkpoint:
        for checkpoint_key in checkpoint.keys():
            local_key = map_qwen2_checkpoint_key(checkpoint_key)
            if local_key not in local_state:
                unexpected_keys.append(checkpoint_key)
                continue

            present_local_keys.add(local_key)
            checkpoint_shape = tuple(
                checkpoint.get_slice(checkpoint_key).get_shape()
            )
            local_shape = tuple(local_state[local_key].shape)
            if checkpoint_shape != local_shape:
                shape_mismatches.append(
                    (local_key, checkpoint_shape, local_shape)
                )

    if tie_word_embeddings:
        embedding_key = "model.embed_tokens.weight"
        lm_head_key = "lm_head.weight"
        if embedding_key in present_local_keys:
            present_local_keys.add(lm_head_key)
        if lm_head_key in present_local_keys:
            present_local_keys.add(embedding_key)

    return Qwen2CheckpointValidation(
        missing_keys=sorted(set(local_state) - present_local_keys),
        unexpected_keys=sorted(unexpected_keys),
        shape_mismatches=sorted(shape_mismatches),
    )
