import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from mini_llm_runtime.model.qwen2 import (
    TinyQwen2Config,
    TinyQwen2ForCausalLM,
)
from mini_llm_runtime.model.qwen2_loader import (
    load_qwen2_safetensors,
)
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache


MODEL_DIR = Path("checkpoints/Qwen2.5-0.5B")
CONFIG_PATH = MODEL_DIR / "config.json"
CHECKPOINT_PATH = MODEL_DIR / "model.safetensors"

PROMPT_TOKEN_IDS = [10, 20, 30, 40]
DECODE_TOKEN_ID = 50

BLOCK_SIZE = 4
NUM_BLOCKS = 2


def print_comparison(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    difference = (actual - expected).abs()
    print(f"{name} actual top token:", int(actual.argmax().item()))
    print(f"{name} expected top token:", int(expected.argmax().item()))
    print(f"{name} max absolute difference:", difference.max().item())
    print(f"{name} mean absolute difference:", difference.mean().item())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this comparison")

    device = torch.device("cuda")
    prompt_input_ids = torch.tensor(
        [PROMPT_TOKEN_IDS],
        dtype=torch.long,
        device=device,
    )
    full_input_ids = torch.tensor(
        [[*PROMPT_TOKEN_IDS, DECODE_TOKEN_ID]],
        dtype=torch.long,
        device=device,
    )
    decode_input_ids = torch.tensor(
        [[DECODE_TOKEN_ID]],
        dtype=torch.long,
        device=device,
    )

    print("Loading Hugging Face model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    hf_model.eval()

    with torch.no_grad():
        hf_prefill_output = hf_model(
            prompt_input_ids,
            use_cache=True,
        )
        hf_prompt_logits = hf_prefill_output.logits[0, -1].float().cpu()
        hf_full_logits = (
            hf_model(full_input_ids).logits[0, -1].float().cpu()
        )
        hf_incremental_logits = hf_model(
            decode_input_ids,
            past_key_values=hf_prefill_output.past_key_values,
            use_cache=True,
        ).logits[0, -1].float().cpu()

    print("HF prompt top token:", int(hf_prompt_logits.argmax().item()))
    print("HF full top token:", int(hf_full_logits.argmax().item()))
    print("HF incremental top token:", int(hf_incremental_logits.argmax().item()))
    print_comparison(
        "HF incremental vs full",
        hf_incremental_logits,
        hf_full_logits,
    )

    del hf_prefill_output
    del hf_model
    torch.cuda.empty_cache()
    gc.collect()

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config_dict = json.load(file)
    config = TinyQwen2Config.from_dict(config_dict)

    print("Loading local model...")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            local_model = TinyQwen2ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    load_qwen2_safetensors(
        local_model,
        CHECKPOINT_PATH,
        tie_word_embeddings=config.tie_word_embeddings,
        device="cpu",
    )
    local_model.eval()

    with torch.no_grad():
        local_dense_logits = (
            local_model(full_input_ids)[0, -1].float().cpu()
        )

    print_comparison(
        "Dense full",
        local_dense_logits,
        hf_full_logits,
    )

    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    request_id = "qwen2.5-paged-check"
    total_context_len = len(PROMPT_TOKEN_IDS) + 1
    kv_cache.add_sequence(request_id)
    if not kv_cache.reserve_tokens(request_id, total_context_len):
        raise RuntimeError("Failed to reserve KV cache blocks")

    block_table = list(kv_cache.block_tables[request_id])
    prompt_positions = torch.arange(
        len(PROMPT_TOKEN_IDS),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    prompt_slot_mapping = torch.tensor(
        [
            block_table[position // BLOCK_SIZE] * BLOCK_SIZE
            + position % BLOCK_SIZE
            for position in range(len(PROMPT_TOKEN_IDS))
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        paged_prefill_logits = local_model.forward_prefill_with_cache(
            input_ids=prompt_input_ids,
            position_ids=prompt_positions,
            kv_cache=kv_cache,
            slot_mapping=prompt_slot_mapping,
        )[0, -1].float().cpu()

    print("Block table:", block_table)
    print("Prompt slot mapping:", prompt_slot_mapping.tolist())
    print_comparison(
        "Paged prefill",
        paged_prefill_logits,
        hf_prompt_logits,
    )

    decode_position = len(PROMPT_TOKEN_IDS)
    decode_position_ids = torch.tensor(
        [[decode_position]],
        dtype=torch.long,
        device=device,
    )
    decode_slot_mapping = torch.tensor(
        [
            block_table[decode_position // BLOCK_SIZE] * BLOCK_SIZE
            + decode_position % BLOCK_SIZE
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        paged_decode_logits = local_model.forward_decode_with_cache(
            input_ids=decode_input_ids,
            position_ids=decode_position_ids,
            kv_cache=kv_cache,
            slot_mapping=decode_slot_mapping,
            block_table=block_table,
            context_len=total_context_len,
        )[0, -1].float().cpu()

    print("Decode slot mapping:", decode_slot_mapping.tolist())
    print_comparison(
        "Paged decode vs full",
        paged_decode_logits,
        hf_full_logits,
    )
    print_comparison(
        "Paged decode vs HF incremental",
        paged_decode_logits,
        hf_incremental_logits,
    )

    torch.testing.assert_close(
        local_dense_logits,
        hf_full_logits,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        paged_prefill_logits,
        hf_prompt_logits,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        paged_decode_logits,
        hf_incremental_logits,
        rtol=2e-2,
        atol=2e-2,
    )

    print("Real Qwen2.5 paged prefill/decode alignment passed")


if __name__ == "__main__":
    main()
