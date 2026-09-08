import pytest
import torch
from safetensors.torch import save_file

from mini_llm_runtime.model.qwen2 import (
    TinyQwen2Config,
    TinyQwen2ForCausalLM,
)
from mini_llm_runtime.model.qwen2_loader import (
    Qwen2CheckpointValidation,
    load_qwen2_checkpoint,
    load_qwen2_safetensors,
    map_qwen2_checkpoint_key,
    validate_qwen2_checkpoint,
    validate_qwen2_safetensors_metadata,
)


def test_checkpoint_validation_is_valid_when_report_is_empty():
    validation = Qwen2CheckpointValidation(
        missing_keys=[],
        unexpected_keys=[],
        shape_mismatches=[],
    )

    assert validation.is_valid


@pytest.mark.parametrize(
    ("missing_keys", "unexpected_keys", "shape_mismatches"),
    [
        (["model.norm.weight"], [], []),
        ([], ["unknown.weight"], []),
        ([], [], [("model.norm.weight", (32,), (64,))]),
    ],
)
def test_checkpoint_validation_is_invalid_when_report_has_errors(
    missing_keys,
    unexpected_keys,
    shape_mismatches,
):
    validation = Qwen2CheckpointValidation(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
    )

    assert not validation.is_valid


@pytest.mark.parametrize(
    ("checkpoint_key", "expected_local_key"),
    [
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.qkv_proj.q_proj.weight",
        ),
        (
            "model.layers.3.self_attn.k_proj.bias",
            "model.layers.3.self_attn.qkv_proj.k_proj.bias",
        ),
        (
            "model.layers.23.self_attn.v_proj.weight",
            "model.layers.23.self_attn.qkv_proj.v_proj.weight",
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ),
        (
            "model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ),
        (
            "model.norm.weight",
            "model.norm.weight",
        ),
    ],
)
def test_map_qwen2_checkpoint_key(
    checkpoint_key,
    expected_local_key,
):
    assert (
        map_qwen2_checkpoint_key(checkpoint_key)
        == expected_local_key
    )


def test_validate_qwen2_checkpoint_accepts_mapped_and_direct_keys():
    checkpoint_state = {
        "model.layers.0.self_attn.q_proj.weight": torch.empty(4, 4),
        "model.norm.weight": torch.empty(4),
    }
    local_state = {
        "model.layers.0.self_attn.qkv_proj.q_proj.weight": torch.empty(4, 4),
        "model.norm.weight": torch.empty(4),
    }

    validation = validate_qwen2_checkpoint(
        checkpoint_state,
        local_state,
        tie_word_embeddings=False,
    )

    assert validation.is_valid


def test_validate_qwen2_checkpoint_reports_all_error_categories():
    checkpoint_state = {
        "model.norm.weight": torch.empty(3),
        "unknown.weight": torch.empty(1),
    }
    local_state = {
        "model.norm.weight": torch.empty(4),
        "model.embed_tokens.weight": torch.empty(8, 4),
    }

    validation = validate_qwen2_checkpoint(
        checkpoint_state,
        local_state,
        tie_word_embeddings=False,
    )

    assert validation.missing_keys == ["model.embed_tokens.weight"]
    assert validation.unexpected_keys == ["unknown.weight"]
    assert validation.shape_mismatches == [
        ("model.norm.weight", (3,), (4,)),
    ]
    assert not validation.is_valid


def test_validate_qwen2_checkpoint_allows_missing_tied_lm_head():
    checkpoint_state = {
        "model.embed_tokens.weight": torch.empty(8, 4),
    }
    local_state = {
        "model.embed_tokens.weight": torch.empty(8, 4),
        "lm_head.weight": torch.empty(8, 4),
    }

    validation = validate_qwen2_checkpoint(
        checkpoint_state,
        local_state,
        tie_word_embeddings=True,
    )

    assert validation.is_valid


def test_validate_qwen2_checkpoint_requires_untied_lm_head():
    checkpoint_state = {
        "model.embed_tokens.weight": torch.empty(8, 4),
    }
    local_state = {
        "model.embed_tokens.weight": torch.empty(8, 4),
        "lm_head.weight": torch.empty(8, 4),
    }

    validation = validate_qwen2_checkpoint(
        checkpoint_state,
        local_state,
        tie_word_embeddings=False,
    )

    assert validation.missing_keys == ["lm_head.weight"]
    assert not validation.is_valid


def make_hugging_face_style_checkpoint(model):
    checkpoint_state = {}
    for local_key, tensor in model.state_dict().items():
        checkpoint_key = local_key.replace(
            ".self_attn.qkv_proj.",
            ".self_attn.",
        )
        checkpoint_state[checkpoint_key] = tensor.detach().clone()
    return checkpoint_state


def test_load_qwen2_checkpoint_copies_mapped_tied_weights():
    config = TinyQwen2Config(tie_word_embeddings=True)
    source_model = TinyQwen2ForCausalLM(config)
    checkpoint_state = make_hugging_face_style_checkpoint(source_model)
    del checkpoint_state["lm_head.weight"]

    target_model = TinyQwen2ForCausalLM(config)
    with torch.no_grad():
        for parameter in target_model.parameters():
            parameter.zero_()

    validation = load_qwen2_checkpoint(
        target_model,
        checkpoint_state,
        tie_word_embeddings=config.tie_word_embeddings,
    )

    assert validation.is_valid
    for source_parameter, target_parameter in zip(
        source_model.parameters(),
        target_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(target_parameter, source_parameter)
    assert (
        target_model.lm_head.weight
        is target_model.model.embed_tokens.weight
    )


def test_load_qwen2_checkpoint_does_not_mutate_model_when_invalid():
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    with pytest.raises(ValueError, match="Invalid Qwen2 checkpoint"):
        load_qwen2_checkpoint(
            model,
            {"unknown.weight": torch.empty(1)},
            tie_word_embeddings=config.tie_word_embeddings,
        )

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameters_before[name])


def test_load_qwen2_safetensors_from_local_file(tmp_path):
    config = TinyQwen2Config(tie_word_embeddings=True)
    source_model = TinyQwen2ForCausalLM(config)
    checkpoint_state = make_hugging_face_style_checkpoint(source_model)
    del checkpoint_state["lm_head.weight"]
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(checkpoint_state, str(checkpoint_path))

    target_model = TinyQwen2ForCausalLM(config)
    with torch.no_grad():
        for parameter in target_model.parameters():
            parameter.zero_()

    validation = load_qwen2_safetensors(
        target_model,
        checkpoint_path,
        tie_word_embeddings=config.tie_word_embeddings,
    )

    assert validation.is_valid
    for source_parameter, target_parameter in zip(
        source_model.parameters(),
        target_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(target_parameter, source_parameter)


def test_load_qwen2_safetensors_rejects_missing_file(tmp_path):
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    missing_path = tmp_path / "missing.safetensors"

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint file not found",
    ):
        load_qwen2_safetensors(
            model,
            missing_path,
            tie_word_embeddings=config.tie_word_embeddings,
        )


def test_validate_qwen2_safetensors_metadata_with_meta_model(tmp_path):
    config = TinyQwen2Config(tie_word_embeddings=True)
    source_model = TinyQwen2ForCausalLM(config)
    checkpoint_state = make_hugging_face_style_checkpoint(source_model)
    del checkpoint_state["lm_head.weight"]
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(checkpoint_state, str(checkpoint_path))

    with torch.device("meta"):
        meta_model = TinyQwen2ForCausalLM(config)

    validation = validate_qwen2_safetensors_metadata(
        meta_model,
        checkpoint_path,
        tie_word_embeddings=config.tie_word_embeddings,
    )

    assert validation.is_valid
