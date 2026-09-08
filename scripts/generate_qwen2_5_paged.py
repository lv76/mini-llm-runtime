import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mini_llm_runtime.model.qwen2 import (
    TinyQwen2Config,
    TinyQwen2ForCausalLM,
)
from mini_llm_runtime.model.qwen2_loader import load_qwen2_safetensors
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache


MODEL_DIR = Path("checkpoints/Qwen2.5-0.5B")
CONFIG_PATH = MODEL_DIR / "config.json"
CHECKPOINT_PATH = MODEL_DIR / "model.safetensors"

PROMPT = "who are you?"
MAX_NEW_TOKENS = 10
BLOCK_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text through the local paged Qwen2.5 runtime."
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model, activation, and KV-cache dtype (default: bfloat16).",
    )
    parser.add_argument(
        "--allow-bf16-reduced-precision-reduction",
        action="store_true",
        help="Opt into faster but less accurate reduced-precision BF16 GEMM reductions.",
    )
    return parser.parse_args()


def slot_for_position(
    position: int,
    block_table: list[int],
) -> int:
    logical_block = position // BLOCK_SIZE
    block_offset = position % BLOCK_SIZE
    return block_table[logical_block] * BLOCK_SIZE + block_offset


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real-model generation")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        args.allow_bf16_reduced_precision_reduction
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    prompt_token_ids = tokenizer.encode(PROMPT)
    if not prompt_token_ids:
        raise RuntimeError("Tokenizer produced an empty prompt")

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = TinyQwen2Config.from_dict(json.load(file))

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = TinyQwen2ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)

    load_qwen2_safetensors(
        model,
        CHECKPOINT_PATH,
        tie_word_embeddings=config.tie_word_embeddings,
        device="cpu",
    )
    model.eval()

    max_context_len = len(prompt_token_ids) + MAX_NEW_TOKENS
    num_blocks = math.ceil(max_context_len / BLOCK_SIZE)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
        device=device,
    )
    request_id = "text-generation"
    kv_cache.add_sequence(request_id)
    if not kv_cache.reserve_tokens(request_id, max_context_len):
        raise RuntimeError("Failed to reserve KV cache capacity")
    block_table = list(kv_cache.block_tables[request_id])

    prompt_input_ids = torch.tensor(
        [prompt_token_ids],
        dtype=torch.long,
        device=device,
    )

    prompt_position_ids = torch.arange(
        len(prompt_token_ids),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    prompt_slot_mapping = torch.tensor(
        [
            slot_for_position(position, block_table)
            for position in range(len(prompt_token_ids))
        ],
        dtype=torch.long,
        device=device,
    )#确认slot编号

    with torch.no_grad():
        prefill_logits = model.forward_prefill_with_cache(
            input_ids=prompt_input_ids,
            position_ids=prompt_position_ids,
            kv_cache=kv_cache,
            slot_mapping=prompt_slot_mapping,
        )

    generated_token_ids = [
        int(prefill_logits[0, -1].argmax(dim=-1).item())
    ]

    for generation_index in range(1, MAX_NEW_TOKENS):
        if generated_token_ids[-1] == config.eos_token_id:
            break

        position = len(prompt_token_ids) + generation_index - 1
        context_len = position + 1
        decode_input_ids = torch.tensor(
            [[generated_token_ids[-1]]],
            dtype=torch.long,
            device=device,
        )
        decode_position_ids = torch.tensor(
            [[position]],
            dtype=torch.long,
            device=device,
        )
        decode_slot_mapping = torch.tensor(
            [slot_for_position(position, block_table)],
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            decode_logits = model.forward_decode_with_cache(
                input_ids=decode_input_ids,
                position_ids=decode_position_ids,
                kv_cache=kv_cache,
                slot_mapping=decode_slot_mapping,
                block_table=block_table,
                context_len=context_len,
            )

        generated_token_ids.append(
            int(decode_logits[0, -1].argmax(dim=-1).item())
        )

    generated_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    )

    print("Prompt:", PROMPT)
    print("Dtype:", args.dtype)
    print(
        "BF16 reduced-precision reduction:",
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    )
    print("Prompt token IDs:", prompt_token_ids)
    print("Block table:", block_table)
    print("Generated token IDs:", generated_token_ids)
    print("Generated text:", generated_text)
    print("Full text:", PROMPT + generated_text)


if __name__ == "__main__":
    main()
