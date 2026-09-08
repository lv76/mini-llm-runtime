import json
from pathlib import Path

import pytest
import torch

from mini_llm_runtime.model.qwen2 import (
    TinyQwen2Config,
    TinyQwen2ForCausalLM,
)
from mini_llm_runtime.model.qwen2_loader import load_qwen2_safetensors
from mini_llm_runtime.runtime.model_runner import (
    run_generation_step,
    run_greedy_generation,
)
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache
from mini_llm_runtime.runtime.request import FinishReason, Request, RequestStatus
from mini_llm_runtime.runtime.scheduler import Scheduler


MODEL_DIR = Path("checkpoints/Qwen2.5-0.5B")
CONFIG_PATH = MODEL_DIR / "config.json"
CHECKPOINT_PATH = MODEL_DIR / "model.safetensors"

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA is required for the real Qwen2.5 integration test",
    ),
    pytest.mark.skipif(
        not CONFIG_PATH.is_file() or not CHECKPOINT_PATH.is_file(),
        reason="Local Qwen2.5-0.5B checkpoint is required",
    ),
]


@pytest.fixture(scope="module")
def real_qwen2_5():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = TinyQwen2Config.from_dict(json.load(file))

    device = torch.device("cuda")
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
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
    return model, config


def dense_greedy_tokens(model, prompt_token_ids, max_new_tokens):
    token_ids = list(prompt_token_ids)
    generated = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_ids = torch.tensor(
                [token_ids], dtype=torch.long, device=device
            )
            next_token = int(model(input_ids)[0, -1].argmax().item())
            generated.append(next_token)
            token_ids.append(next_token)

    return generated


def test_real_qwen2_5_dynamic_continuous_batching(real_qwen2_5):
    model, config = real_qwen2_5
    device = next(model.parameters()).device
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=8,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    scheduler = Scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=2,
        kv_cache=kv_cache,
    )

    request_b = Request("B", [20, 21], 2, 0.0)
    request_a = Request("A", [10, 11, 12], 4, 1.0)
    expected_b = dense_greedy_tokens(model, [20, 21], 2)
    expected_a = dense_greedy_tokens(model, [10, 11, 12], 4)

    scheduler.add_request(request_b)
    first_step = run_generation_step(model, scheduler)
    assert first_step.model_batch.input_token_ids.device == device
    assert request_b.output_token_ids == expected_b[:1]
    b_physical_blocks = set(request_b.block_table)

    scheduler.add_request(request_a)
    mixed_step = run_generation_step(model, scheduler)
    assert mixed_step.model_batch.request_ids == ["A", "B"]
    assert mixed_step.model_batch.request_phases == [
        RequestStatus.PREFILLING,
        RequestStatus.DECODING,
    ]
    assert request_b.output_token_ids == expected_b
    assert request_b.is_finished
    assert request_b.request_id not in kv_cache.block_tables
    assert request_a.output_token_ids == expected_a[:1]

    request_c = Request("C", [30, 31], 1, 2.0)
    expected_c = dense_greedy_tokens(model, [30, 31], 1)
    scheduler.add_request(request_c)
    reuse_step = run_generation_step(model, scheduler)
    assert reuse_step.model_batch.request_ids == ["C", "A"]
    assert int(reuse_step.model_batch.block_tables[0, 0]) in b_physical_blocks
    assert request_c.output_token_ids == expected_c
    assert request_c.is_finished
    assert request_a.output_token_ids == expected_a[:2]

    while scheduler.running or scheduler.waiting:
        run_generation_step(model, scheduler)

    assert request_a.output_token_ids == expected_a
    assert request_a.is_finished
    assert scheduler.running == []
    assert list(scheduler.waiting) == []
    assert len(kv_cache.free_blocks) == kv_cache.num_blocks


def test_real_qwen2_5_waits_for_kv_capacity(real_qwen2_5):
    model, config = real_qwen2_5
    device = next(model.parameters()).device
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=2,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    scheduler = Scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=2,
        kv_cache=kv_cache,
    )

    request_b = Request("B", [20, 21], 2, 0.0)
    request_a = Request("A", [10, 11, 12, 13, 14], 1, 1.0)
    expected_b = dense_greedy_tokens(model, request_b.prompt_token_ids, 2)
    expected_a = dense_greedy_tokens(model, request_a.prompt_token_ids, 1)

    scheduler.add_request(request_b)
    first_step = run_generation_step(model, scheduler)

    assert first_step.scheduler_output.prefill_requests == [request_b]
    assert request_b.output_token_ids == expected_b[:1]
    assert request_b.status is RequestStatus.DECODING
    assert len(request_b.block_table) == 1
    b_physical_block = request_b.block_table[0]
    assert len(kv_cache.free_blocks) == 1

    scheduler.add_request(request_a)
    assert list(scheduler.waiting) == [request_a]
    assert request_a.block_table == []

    blocked_step = run_generation_step(model, scheduler)

    assert blocked_step.scheduler_output.decode_requests == [request_b]
    assert blocked_step.scheduler_output.prefill_requests == []
    assert blocked_step.model_batch.request_ids == ["B"]
    assert request_b.output_token_ids == expected_b
    assert request_b.is_finished
    assert request_b.request_id not in kv_cache.block_tables

    assert list(scheduler.waiting) == [request_a]
    assert request_a.status is RequestStatus.WAITING
    assert request_a.block_table == []
    assert request_a.num_computed_tokens == 0
    assert len(kv_cache.free_blocks) == 2

    admitted_step = run_generation_step(model, scheduler)

    assert admitted_step.scheduler_output.prefill_requests == [request_a]
    assert admitted_step.scheduler_output.decode_requests == []
    assert admitted_step.model_batch.request_ids == ["A"]
    allocated_blocks = admitted_step.model_batch.block_tables[0].tolist()
    assert len(allocated_blocks) == 2
    assert b_physical_block in allocated_blocks
    assert request_a.output_token_ids == expected_a
    assert request_a.is_finished

    assert list(scheduler.waiting) == []
    assert scheduler.running == []
    assert len(kv_cache.free_blocks) == kv_cache.num_blocks


def test_real_qwen2_5_grows_across_multiple_blocks(real_qwen2_5):
    model, config = real_qwen2_5
    device = next(model.parameters()).device
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=4,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    scheduler = Scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=1,
        kv_cache=kv_cache,
    )
    request = Request("growing", [10, 11, 12], 7, 0.0)
    expected_tokens = dense_greedy_tokens(
        model,
        request.prompt_token_ids,
        request.max_new_tokens,
    )

    scheduler.add_request(request)

    observed_block_counts = []
    observed_context_lens = []

    while scheduler.waiting or scheduler.running:
        step = run_generation_step(model, scheduler)

        valid_blocks = step.model_batch.block_tables[0]
        num_valid_blocks = int((valid_blocks >= 0).sum().item())
        observed_block_counts.append(num_valid_blocks)
        observed_context_lens.append(
            int(step.model_batch.context_lens[0].item())
        )

    assert observed_context_lens == [3, 4, 5, 6, 7, 8, 9]
    assert observed_block_counts == [1, 1, 2, 2, 2, 2, 3]

    assert request.output_token_ids == expected_tokens
    assert request.is_finished
    assert scheduler.running == []
    assert list(scheduler.waiting) == []
    assert len(kv_cache.free_blocks) == kv_cache.num_blocks


def test_real_qwen2_5_admits_waiting_requests_in_fifo_order(real_qwen2_5):
    model, config = real_qwen2_5
    device = next(model.parameters()).device
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=2,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    scheduler = Scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=1,
        kv_cache=kv_cache,
    )

    request_b = Request("B", [20, 21], 2, 0.0)
    request_a = Request("A", [10, 11, 12, 13, 14], 1, 1.0)
    request_c = Request("C", [30, 31], 1, 2.0)
    expected_b = dense_greedy_tokens(model, request_b.prompt_token_ids, 2)
    expected_a = dense_greedy_tokens(model, request_a.prompt_token_ids, 1)
    expected_c = dense_greedy_tokens(model, request_c.prompt_token_ids, 1)

    scheduler.add_request(request_b)
    first_step = run_generation_step(model, scheduler)
    assert first_step.scheduler_output.prefill_requests == [request_b]
    assert request_b.status is RequestStatus.DECODING

    scheduler.add_request(request_a)
    scheduler.add_request(request_c)
    assert list(scheduler.waiting) == [request_a, request_c]

    b_finish_step = run_generation_step(model, scheduler)
    assert b_finish_step.scheduler_output.decode_requests == [request_b]
    assert b_finish_step.scheduler_output.prefill_requests == []
    assert request_b.output_token_ids == expected_b
    assert request_b.is_finished
    assert list(scheduler.waiting) == [request_a, request_c]

    a_step = run_generation_step(model, scheduler)
    assert a_step.scheduler_output.prefill_requests == [request_a]
    assert a_step.scheduler_output.decode_requests == []
    assert a_step.model_batch.request_ids == ["A"]
    assert request_a.output_token_ids == expected_a
    assert request_a.is_finished
    assert list(scheduler.waiting) == [request_c]

    c_step = run_generation_step(model, scheduler)
    assert c_step.scheduler_output.prefill_requests == [request_c]
    assert c_step.scheduler_output.decode_requests == []
    assert c_step.model_batch.request_ids == ["C"]
    assert request_c.output_token_ids == expected_c
    assert request_c.is_finished

    assert list(scheduler.waiting) == []
    assert scheduler.running == []
    assert len(kv_cache.free_blocks) == kv_cache.num_blocks


def test_real_qwen2_5_finishes_on_eos(real_qwen2_5):
    model, config = real_qwen2_5
    device = next(model.parameters()).device
    prompt = [10, 11, 12]
    eos_token_id = dense_greedy_tokens(model, prompt, 1)[0]
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=2,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    scheduler = Scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=1,
        kv_cache=kv_cache,
    )
    request = Request(
        request_id="eos",
        prompt_token_ids=prompt,
        max_new_tokens=10,
        arrival_time=0.0,
    )
    scheduler.add_request(request)

    finished = run_greedy_generation(
        model,
        scheduler,
        eos_token_id=eos_token_id,
    )

    assert finished == [request]
    assert request.output_token_ids == [eos_token_id]
    assert request.finish_reason is FinishReason.EOS
    assert request.is_finished
    assert scheduler.running == []
    assert list(scheduler.waiting) == []
    assert request.request_id not in kv_cache.block_tables
    assert len(kv_cache.free_blocks) == kv_cache.num_blocks
