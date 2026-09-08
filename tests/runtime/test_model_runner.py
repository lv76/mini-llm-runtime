import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from mini_llm_runtime.model.qwen2 import TinyQwen2Config, TinyQwen2ForCausalLM
from mini_llm_runtime.runtime.model_runner import (
    ModelBatch,
    build_model_batch,
    execute_model_batch,
    greedy_sample,
    run_generation_step,
    run_greedy_generation,
)
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache
from mini_llm_runtime.runtime.request import FinishReason, Request, RequestStatus
from mini_llm_runtime.runtime.scheduler import Scheduler, SchedulerOutput


def dense_greedy_tokens(model, prompt_token_ids, max_new_tokens):
    full_token_ids = list(prompt_token_ids)
    output_token_ids = []

    for _ in range(max_new_tokens):
        logits = model(torch.tensor([full_token_ids], dtype=torch.long))
        next_token_id = int(logits[0, -1].argmax().item())
        output_token_ids.append(next_token_id)
        full_token_ids.append(next_token_id)

    return output_token_ids


@pytest.mark.parametrize("mixed,interleaved", [(False, False), (True, False), (True, True)])
@torch.no_grad()
def test_runner_groups_decode_and_preserves_packed_order(mixed, interleaved):
    torch.manual_seed(23)
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config).eval()
    cache = PagedKVCache(2, 16, 2, 2, config.head_dim, dtype=torch.float32)
    cache.k_cache.zero_()
    cache.v_cache.zero_()
    scheduler = Scheduler(max_num_batched_tokens=16, max_num_seqs=3, kv_cache=cache)
    for request_id, prompt in [("B", [20, 21]), ("C", [30, 31, 32, 33])]:
        scheduler.add_request(Request(request_id, prompt, 5, 0.0))
    run_generation_step(model, scheduler)
    if mixed:
        scheduler.add_request(Request("A", [10, 11, 12], 2, 1.0))
    batch = build_model_batch(scheduler.schedule(), block_size=2)
    if interleaved:
        # Move the prefill chunk between decode chunks to catch scatter mistakes.
        order = [2, 0, 1]
        chunks = [
            torch.arange(int(batch.query_start_loc[i]), int(batch.query_start_loc[i + 1]))
            for i in order
        ]
        tokens = torch.cat(chunks)
        batch = replace(
            batch,
            request_ids=[batch.request_ids[i] for i in order],
            request_phases=[batch.request_phases[i] for i in order],
            input_token_ids=batch.input_token_ids[tokens],
            positions=batch.positions[tokens],
            slot_mapping=batch.slot_mapping[tokens],
            context_lens=batch.context_lens[order],
            block_tables=batch.block_tables[order],
            query_start_loc=torch.tensor([0] + [len(chunk) for chunk in chunks]).cumsum(0),
        )

    reference_cache = copy.deepcopy(cache)
    expected_chunks = []
    for i, phase in enumerate(batch.request_phases):
        start, end = int(batch.query_start_loc[i]), int(batch.query_start_loc[i + 1])
        args = dict(
            input_ids=batch.input_token_ids[start:end][None, :],
            position_ids=batch.positions[start:end][None, :],
            kv_cache=reference_cache,
            slot_mapping=batch.slot_mapping[start:end],
        )
        if phase is RequestStatus.PREFILLING:
            logits = model.forward_prefill_with_cache(**args)
        else:
            logits = model.forward_decode_with_cache(
                **args, block_table=batch.block_tables[i].tolist(),
                context_len=int(batch.context_lens[i]),
            )
        expected_chunks.append(logits.squeeze(0))
    expected = torch.cat(expected_chunks)
    with (
        patch.object(model, "forward_decode_batch_with_cache", wraps=model.forward_decode_batch_with_cache) as decode,
        patch.object(model, "forward_decode_with_cache", side_effect=AssertionError("single decode called")),
        patch.object(model, "forward_prefill_with_cache", wraps=model.forward_prefill_with_cache) as prefill,
    ):
        actual = execute_model_batch(model, batch, cache)
    assert decode.call_count == 1
    assert decode.call_args.kwargs["input_ids"].shape == (2, 1)
    assert prefill.call_count == int(mixed)
    torch.testing.assert_close(actual, expected)
    assert greedy_sample(actual, batch) == greedy_sample(expected, batch)
    torch.testing.assert_close(cache.k_cache, reference_cache.k_cache)
    torch.testing.assert_close(cache.v_cache, reference_cache.v_cache)


def test_build_model_batch_with_prefill_and_decode():
    request_a = Request(
        request_id="A",
        prompt_token_ids=[10, 11, 12],
        max_new_tokens=3,
        arrival_time=0.0,
    )
    request_a.start_prefill()
    request_a.block_table.extend([7, 2])

    request_b = Request(
        request_id="B",
        prompt_token_ids=[20, 21],
        max_new_tokens=3,
        arrival_time=1.0,
    )
    request_b.status = RequestStatus.DECODING
    request_b.num_computed_tokens = 2
    request_b.output_token_ids.append(30)
    request_b.block_table.extend([5])


    scheduler_output = SchedulerOutput(
        prefill_requests=[request_a],
        decode_requests=[request_b],
        num_scheduled_tokens={
            "A": 3,
            "B": 1,
        },
    )

    batch = build_model_batch(scheduler_output,block_size=4)

    assert batch.request_ids == ["A", "B"]
    assert batch.request_phases == [
        RequestStatus.PREFILLING,
        RequestStatus.DECODING,
    ]

    torch.testing.assert_close(
        batch.input_token_ids,
        torch.tensor([10, 11, 12, 30], dtype=torch.long),
    )
    torch.testing.assert_close(
        batch.positions,
        torch.tensor([0, 1, 2, 2], dtype=torch.long),
    )
    torch.testing.assert_close(
        batch.query_start_loc,
        torch.tensor([0, 3, 4], dtype=torch.long),
    )

    torch.testing.assert_close(
        batch.context_lens,
        torch.tensor([3, 3], dtype=torch.long),
    )

    torch.testing.assert_close(
      batch.block_tables,
      torch.tensor([[7, 2], [5, -1]], dtype=torch.long),
    )
    torch.testing.assert_close(
        batch.slot_mapping,
        torch.tensor([28, 29, 30, 22], dtype=torch.long),
    )

def test_build_empty_model_batch():
    batch = build_model_batch(SchedulerOutput(),block_size=4)

    assert batch.request_ids == []
    assert batch.request_phases == []
    torch.testing.assert_close(
        batch.input_token_ids, torch.tensor([], dtype=torch.long)
    )
    torch.testing.assert_close(
        batch.positions, torch.tensor([], dtype=torch.long)
    )
    torch.testing.assert_close(
        batch.query_start_loc, torch.tensor([0], dtype=torch.long)
    )
    torch.testing.assert_close(
        batch.context_lens, torch.tensor([], dtype=torch.long)
    )
    assert batch.block_tables.shape == (0, 0)
    torch.testing.assert_close(
        batch.slot_mapping, torch.tensor([], dtype=torch.long)
    )


def test_build_model_batch_rejects_decode_without_pending_token():
    request = Request(
        request_id="A",
        prompt_token_ids=[10, 11],
        max_new_tokens=3,
        arrival_time=0.0,
        status=RequestStatus.DECODING,
        num_computed_tokens=2,
    )
    scheduler_output = SchedulerOutput(
        decode_requests=[request],
        num_scheduled_tokens={"A": 1},
    )

    with pytest.raises(ValueError, match="has no pending token"):
        build_model_batch(scheduler_output,block_size=4)


@pytest.mark.parametrize("block_size", [0, -1])
def test_build_model_batch_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="block_size must be greater than 0"):
        build_model_batch(SchedulerOutput(), block_size=block_size)


def test_build_model_batch_rejects_insufficient_block_table():
    request = Request(
        request_id="A",
        prompt_token_ids=[10],
        max_new_tokens=1,
        arrival_time=0.0,
    )
    request.start_prefill()

    scheduler_output = SchedulerOutput(
        prefill_requests=[request],
        num_scheduled_tokens={"A": 1},
    )

    with pytest.raises(
        ValueError,
        match="Request A has no KV cache block for position 0",
    ):
        build_model_batch(scheduler_output, block_size=4)


def test_execute_model_batch_prefill_matches_dense():
    torch.manual_seed(0)
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=8,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    request = Request(
        request_id="A",
        prompt_token_ids=[10, 11, 12],
        max_new_tokens=2,
        arrival_time=0.0,
    )
    request.start_prefill()
    kv_cache.add_sequence(request.request_id)
    assert kv_cache.reserve_tokens(request.request_id, request.prompt_len)
    request.block_table = kv_cache.block_tables[request.request_id]

    scheduler_output = SchedulerOutput(
        prefill_requests=[request],
        num_scheduled_tokens={request.request_id: request.prompt_len},
    )
    batch = build_model_batch(
        scheduler_output,
        block_size=kv_cache.block_size,
    )

    actual = execute_model_batch(model, batch, kv_cache)
    expected = model(
        torch.tensor([request.prompt_token_ids], dtype=torch.long)
    ).squeeze(0)

    assert actual.shape == (request.prompt_len, config.vocab_size)
    torch.testing.assert_close(actual, expected)


def test_execute_model_batch_rejects_empty_batch():
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=1,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    batch = build_model_batch(
        SchedulerOutput(),
        block_size=kv_cache.block_size,
    )

    with pytest.raises(ValueError, match="empty ModelBatch"):
        execute_model_batch(model, batch, kv_cache)


@pytest.mark.parametrize(
    "request_phases",
    [
        [],
        [RequestStatus.PREFILLING, RequestStatus.PREFILLING],
    ],
)
def test_execute_model_batch_rejects_request_phase_length_mismatch(
    request_phases,
):
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=2,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    request = Request(
        request_id="A",
        prompt_token_ids=[10],
        max_new_tokens=1,
        arrival_time=0.0,
    )
    request.start_prefill()
    kv_cache.add_sequence(request.request_id)
    assert kv_cache.reserve_tokens(request.request_id, request.prompt_len)
    request.block_table = kv_cache.block_tables[request.request_id]
    batch = build_model_batch(
        SchedulerOutput(
            prefill_requests=[request],
            num_scheduled_tokens={request.request_id: request.prompt_len},
        ),
        block_size=kv_cache.block_size,
    )
    batch.request_phases = request_phases

    with pytest.raises(
        ValueError,
        match="request_phases length must match request_ids",
    ):
        execute_model_batch(model, batch, kv_cache)


@pytest.mark.parametrize(
    ("query_start_loc", "error_message"),
    [
        ([0], "one more entry than request_ids"),
        ([1, 2], "must start at 0"),
        ([0, 0], "must be strictly increasing"),
        ([0, 2], "must end at the packed token count"),
    ],
)
def test_execute_model_batch_rejects_invalid_query_start_loc(
    query_start_loc,
    error_message,
):
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=1,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    batch = ModelBatch(
        request_ids=["A"],
        input_token_ids=torch.tensor([10], dtype=torch.long),
        positions=torch.tensor([0], dtype=torch.long),
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.long),
        request_phases=[RequestStatus.PREFILLING],
        context_lens=torch.tensor([1], dtype=torch.long),
        block_tables=torch.tensor([[0]], dtype=torch.long),
        slot_mapping=torch.tensor([0], dtype=torch.long),
    )

    with pytest.raises(ValueError, match=error_message):
        execute_model_batch(model, batch, kv_cache)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_message"),
    [
        (
            "positions",
            torch.tensor([], dtype=torch.long),
            "positions length must match input_token_ids",
        ),
        (
            "slot_mapping",
            torch.tensor([], dtype=torch.long),
            "slot_mapping length must match input_token_ids",
        ),
        (
            "context_lens",
            torch.tensor([], dtype=torch.long),
            "context_lens length must match request_ids",
        ),
        (
            "block_tables",
            torch.empty((0, 1), dtype=torch.long),
            "block_tables rows must match request_ids",
        ),
        (
            "block_tables",
            torch.tensor([0], dtype=torch.long),
            "block_tables must be two-dimensional",
        ),
    ],
)
def test_execute_model_batch_rejects_inconsistent_tensor_lengths(
    field_name,
    invalid_value,
    error_message,
):
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=1,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    batch = ModelBatch(
        request_ids=["A"],
        input_token_ids=torch.tensor([10], dtype=torch.long),
        positions=torch.tensor([0], dtype=torch.long),
        query_start_loc=torch.tensor([0, 1], dtype=torch.long),
        request_phases=[RequestStatus.PREFILLING],
        context_lens=torch.tensor([1], dtype=torch.long),
        block_tables=torch.tensor([[0]], dtype=torch.long),
        slot_mapping=torch.tensor([0], dtype=torch.long),
    )
    setattr(batch, field_name, invalid_value)

    with pytest.raises(ValueError, match=error_message):
        execute_model_batch(model, batch, kv_cache)


@pytest.mark.parametrize(
    ("context_lens", "block_tables", "error_message"),
    [
        ([0], [[0]], "context_len must be greater than 0"),
        ([5], [[0]], "block_table has insufficient blocks"),
        ([1], [[-1]], "block_table contains an invalid physical block"),
        ([1], [[1]], "block_table contains an invalid physical block"),
    ],
)
def test_execute_model_batch_rejects_invalid_effective_block_table(
    context_lens,
    block_tables,
    error_message,
):
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=1,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    batch = ModelBatch(
        request_ids=["A"],
        input_token_ids=torch.tensor([10], dtype=torch.long),
        positions=torch.tensor([0], dtype=torch.long),
        query_start_loc=torch.tensor([0, 1], dtype=torch.long),
        request_phases=[RequestStatus.PREFILLING],
        context_lens=torch.tensor(context_lens, dtype=torch.long),
        block_tables=torch.tensor(block_tables, dtype=torch.long),
        slot_mapping=torch.tensor([0], dtype=torch.long),
    )

    with pytest.raises(ValueError, match=error_message):
        execute_model_batch(model, batch, kv_cache)


def make_single_token_model_batch():
    return ModelBatch(
        request_ids=["A"],
        input_token_ids=torch.tensor([10], dtype=torch.long),
        positions=torch.tensor([0], dtype=torch.long),
        query_start_loc=torch.tensor([0, 1], dtype=torch.long),
        request_phases=[RequestStatus.PREFILLING],
        context_lens=torch.tensor([1], dtype=torch.long),
        block_tables=torch.tensor([[0]], dtype=torch.long),
        slot_mapping=torch.tensor([0], dtype=torch.long),
    )


@pytest.mark.parametrize("field_name", ["input_token_ids", "context_lens"])
def test_execute_model_batch_rejects_non_long_index_tensor(field_name):
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=1,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    batch = make_single_token_model_batch()
    original = getattr(batch, field_name)
    setattr(batch, field_name, original.to(torch.int32))

    with pytest.raises(ValueError, match=f"{field_name} must use torch.long"):
        execute_model_batch(model, batch, kv_cache)


def test_execute_model_batch_rejects_batch_tensor_device_mismatch():
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=1,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    batch = make_single_token_model_batch()
    batch.positions = torch.empty((1,), dtype=torch.long, device="meta")

    with pytest.raises(ValueError, match="positions must be on the same device"):
        execute_model_batch(model, batch, kv_cache)


def test_execute_model_batch_rejects_kv_cache_dtype_mismatch():
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=1,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="KV cache dtype must match model dtype"):
        execute_model_batch(model, make_single_token_model_batch(), kv_cache)


def test_execute_model_batch_rejects_model_device_mismatch():
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config).to("meta")
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=1,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="ModelBatch and model must be on"):
        execute_model_batch(model, make_single_token_model_batch(), kv_cache)


def test_execute_model_batch_decode_matches_dense():
    torch.manual_seed(0)
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=8,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    request = Request(
        request_id="A",
        prompt_token_ids=[10, 11, 12],
        max_new_tokens=2,
        arrival_time=0.0,
    )
    request.start_prefill()
    kv_cache.add_sequence(request.request_id)
    assert kv_cache.reserve_tokens(request.request_id, request.prompt_len)
    request.block_table = kv_cache.block_tables[request.request_id]

    prefill_output = SchedulerOutput(
        prefill_requests=[request],
        num_scheduled_tokens={request.request_id: request.prompt_len},
    )
    prefill_batch = build_model_batch(
        prefill_output,
        block_size=kv_cache.block_size,
    )
    execute_model_batch(model, prefill_batch, kv_cache)

    decode_token = 30
    request.update_after_model(
        num_new_computed_tokens=request.prompt_len,
        sampled_token_id=decode_token,
    )
    assert kv_cache.reserve_tokens(
        request.request_id,
        request.num_computed_tokens + 1,
    )

    decode_output = SchedulerOutput(
        decode_requests=[request],
        num_scheduled_tokens={request.request_id: 1},
    )
    decode_batch = build_model_batch(
        decode_output,
        block_size=kv_cache.block_size,
    )
    actual = execute_model_batch(model, decode_batch, kv_cache)
    expected = model(
        torch.tensor([[10, 11, 12, decode_token]], dtype=torch.long)
    )[0, -1]

    assert actual.shape == (1, config.vocab_size)
    torch.testing.assert_close(actual[0], expected)


def test_greedy_sample_uses_each_requests_last_query():
    request_a = Request("A", [10, 11, 12], 1, 0.0)
    request_a.start_prefill()
    request_a.block_table.append(0)
    request_b = Request(
        "B",
        [20, 21],
        1,
        0.0,
        output_token_ids=[30],
        status=RequestStatus.DECODING,
        num_computed_tokens=2,
        block_table=[1],
    )
    batch = build_model_batch(
        SchedulerOutput(
            prefill_requests=[request_a],
            decode_requests=[request_b],
            num_scheduled_tokens={"A": 3, "B": 1},
        ),
        block_size=4,
    )
    packed_logits = torch.zeros(4, 8)
    packed_logits[0, 1] = 10
    packed_logits[1, 2] = 10
    packed_logits[2, 5] = 10
    packed_logits[3, 7] = 10

    assert greedy_sample(packed_logits, batch) == {"A": 5, "B": 7}


def test_run_greedy_generation_matches_dense_recomputation():
    torch.manual_seed(0)
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    prompt = [10, 11, 12]
    max_new_tokens = 3

    expected_tokens: list[int] = []
    full_tokens = list(prompt)
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([full_tokens], dtype=torch.long))
        next_token = int(logits[0, -1].argmax().item())
        expected_tokens.append(next_token)
        full_tokens.append(next_token)

    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=8,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    scheduler = Scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=1,
        kv_cache=kv_cache,
    )
    request = Request(
        request_id="A",
        prompt_token_ids=prompt,
        max_new_tokens=max_new_tokens,
        arrival_time=0.0,
    )
    scheduler.add_request(request)

    finished = run_greedy_generation(model, scheduler)

    assert finished == [request]
    assert request.output_token_ids == expected_tokens
    assert request.is_finished
    assert scheduler.running == []
    assert list(scheduler.waiting) == []
    assert request.request_id not in kv_cache.block_tables


def test_run_greedy_generation_finishes_on_eos():
    torch.manual_seed(0)
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    prompt = [10, 11, 12]
    eos_token_id = dense_greedy_tokens(model, prompt, 1)[0]
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=8,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
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


def test_run_generation_step_returns_prefill_result():
    torch.manual_seed(0)
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=8,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    scheduler = Scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=1,
        kv_cache=kv_cache,
    )
    request = Request(
        request_id="A",
        prompt_token_ids=[10, 11, 12],
        max_new_tokens=3,
        arrival_time=0.0,
    )
    scheduler.add_request(request)

    dense_logits = model(
        torch.tensor([request.prompt_token_ids], dtype=torch.long)
    )
    expected_token = int(dense_logits[0, -1].argmax().item())

    step_result = run_generation_step(model, scheduler)

    assert step_result.scheduler_output.prefill_requests == [request]
    assert step_result.scheduler_output.decode_requests == []
    assert step_result.scheduler_output.num_scheduled_tokens == {
        request.request_id: request.prompt_len,
    }

    assert step_result.model_batch.request_ids == [request.request_id]
    assert step_result.model_batch.request_phases == [
        RequestStatus.PREFILLING,
    ]
    torch.testing.assert_close(
        step_result.model_batch.query_start_loc,
        torch.tensor([0, request.prompt_len], dtype=torch.long),
    )
    assert step_result.packed_logits.shape == (
        request.prompt_len,
        config.vocab_size,
    )
    assert int(step_result.packed_logits[-1].argmax().item()) == expected_token

    assert step_result.sampled_token_ids == {
        request.request_id: expected_token,
    }
    assert step_result.finished_requests == []
    assert request.status is RequestStatus.DECODING
    assert request.num_computed_tokens == request.prompt_len
    assert request.output_token_ids == [expected_token]
    assert request.pending_token_id == expected_token


def test_run_generation_step_rejects_scheduler_without_requests():
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=1,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    scheduler = Scheduler(
        max_num_batched_tokens=4,
        max_num_seqs=1,
        kv_cache=kv_cache,
    )

    with pytest.raises(ValueError, match="no requests"):
        run_generation_step(model, scheduler)


def test_run_generation_step_rejects_no_progress_with_waiting_request():
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=0,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    scheduler = Scheduler(
        max_num_batched_tokens=4,
        max_num_seqs=1,
        kv_cache=kv_cache,
    )
    request = Request(
        request_id="A",
        prompt_token_ids=[10],
        max_new_tokens=1,
        arrival_time=0.0,
    )
    scheduler.add_request(request)

    with pytest.raises(RuntimeError, match="made no progress"):
        run_generation_step(model, scheduler)

    assert list(scheduler.waiting) == [request]
    assert scheduler.running == []


def test_execute_model_batch_mixed_prefill_decode_matches_dense():
    torch.manual_seed(0)
    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)
    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=16,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )
    scheduler = Scheduler(
        max_num_batched_tokens=5,
        max_num_seqs=2,
        kv_cache=kv_cache,
    )

    request_b = Request(
        request_id="B",
        prompt_token_ids=[20, 21],
        max_new_tokens=3,
        arrival_time=0.0,
    )
    scheduler.add_request(request_b)

    prefill_b_output = scheduler.schedule()
    prefill_b_batch = build_model_batch(
        prefill_b_output,
        block_size=kv_cache.block_size,
    )
    prefill_b_logits = execute_model_batch(
        model,
        prefill_b_batch,
        kv_cache,
    )
    sampled_b = greedy_sample(prefill_b_logits, prefill_b_batch)
    scheduler.update_after_model(prefill_b_output, sampled_b)
    pending_b_token = request_b.pending_token_id
    assert pending_b_token is not None

    request_a = Request(
        request_id="A",
        prompt_token_ids=[10, 11, 12],
        max_new_tokens=3,
        arrival_time=1.0,
    )
    scheduler.add_request(request_a)

    mixed_output = scheduler.schedule()
    mixed_batch = build_model_batch(
        mixed_output,
        block_size=kv_cache.block_size,
    )

    assert mixed_batch.request_ids == ["A", "B"]
    assert mixed_batch.request_phases == [
        RequestStatus.PREFILLING,
        RequestStatus.DECODING,
    ]
    torch.testing.assert_close(
        mixed_batch.query_start_loc,
        torch.tensor([0, 3, 4], dtype=torch.long),
    )
    assert request_a.block_table[0] != request_b.block_table[0]

    actual = execute_model_batch(model, mixed_batch, kv_cache)
    expected_a = model(
        torch.tensor([request_a.prompt_token_ids], dtype=torch.long)
    ).squeeze(0)
    expected_b = model(
        torch.tensor(
            [[*request_b.prompt_token_ids, pending_b_token]],
            dtype=torch.long,
        )
    )[0, -1]

    assert actual.shape == (4, config.vocab_size)
    torch.testing.assert_close(actual[:3], expected_a)
    torch.testing.assert_close(actual[3], expected_b)

    sampled = greedy_sample(actual, mixed_batch)
    assert sampled == {
        "A": int(expected_a[-1].argmax().item()),
        "B": int(expected_b.argmax().item()),
    }


def test_dynamic_continuous_batching_lifecycle():
    torch.manual_seed(0)

    config = TinyQwen2Config()
    model = TinyQwen2ForCausalLM(config)

    kv_cache = PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=8,
        block_size=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
    )

    scheduler = Scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=2,
        kv_cache=kv_cache,
    )

    request_b = Request(
        request_id="B",
        prompt_token_ids=[20, 21],
        max_new_tokens=2,
        arrival_time=0.0,
    )

    request_a = Request(
        request_id="A",
        prompt_token_ids=[10, 11, 12],
        max_new_tokens=4,
        arrival_time=1.0,
    )

    expected_b = dense_greedy_tokens(
        model,
        request_b.prompt_token_ids,
        request_b.max_new_tokens,
    )
    expected_a = dense_greedy_tokens(
        model,
        request_a.prompt_token_ids,
        request_a.max_new_tokens,
    )

    assert request_b.status is RequestStatus.WAITING
    assert request_a.status is RequestStatus.WAITING
    assert len(expected_b) == request_b.max_new_tokens
    assert len(expected_a) == request_a.max_new_tokens

    scheduler.add_request(request_b)
    first_step = run_generation_step(model, scheduler)

    assert first_step.scheduler_output.prefill_requests == [request_b]
    assert first_step.scheduler_output.decode_requests == []
    assert first_step.model_batch.request_ids == [request_b.request_id]
    assert first_step.model_batch.request_phases == [
        RequestStatus.PREFILLING,
    ]
    assert first_step.sampled_token_ids == {
        request_b.request_id: expected_b[0],
    }
    assert first_step.finished_requests == []
    assert request_b.status is RequestStatus.DECODING
    assert request_b.num_computed_tokens == request_b.prompt_len
    assert request_b.output_token_ids == expected_b[:1]
    assert request_b in scheduler.running
    assert list(scheduler.waiting) == []
    b_physical_blocks = set(request_b.block_table)

    scheduler.add_request(request_a)
    second_step = run_generation_step(model, scheduler)

    assert second_step.scheduler_output.prefill_requests == [request_a]
    assert second_step.scheduler_output.decode_requests == [request_b]
    assert second_step.model_batch.request_ids == [
        request_a.request_id,
        request_b.request_id,
    ]
    assert second_step.model_batch.request_phases == [
        RequestStatus.PREFILLING,
        RequestStatus.DECODING,
    ]
    torch.testing.assert_close(
        second_step.model_batch.query_start_loc,
        torch.tensor([0, request_a.prompt_len, request_a.prompt_len + 1]),
    )
    assert (
        second_step.model_batch.block_tables[0, 0].item()
        != second_step.model_batch.block_tables[1, 0].item()
    )
    assert second_step.sampled_token_ids == {
        request_a.request_id: expected_a[0],
        request_b.request_id: expected_b[1],
    }
    assert second_step.finished_requests == [request_b]
    assert request_b.is_finished
    assert request_b.output_token_ids == expected_b
    assert request_b not in scheduler.running
    assert request_b.request_id not in kv_cache.block_tables
    assert request_a.status is RequestStatus.DECODING
    assert request_a.output_token_ids == expected_a[:1]
    assert scheduler.running == [request_a]
    assert list(scheduler.waiting) == []

    request_c = Request(
        request_id="C",
        prompt_token_ids=[30, 31],
        max_new_tokens=1,
        arrival_time=2.0,
    )
    expected_c = dense_greedy_tokens(
        model,
        request_c.prompt_token_ids,
        request_c.max_new_tokens,
    )
    scheduler.add_request(request_c)

    reuse_step = run_generation_step(model, scheduler)

    assert reuse_step.model_batch.request_ids == [
        request_c.request_id,
        request_a.request_id,
    ]
    assert reuse_step.model_batch.request_phases == [
        RequestStatus.PREFILLING,
        RequestStatus.DECODING,
    ]
    c_physical_block = int(reuse_step.model_batch.block_tables[0, 0].item())
    assert c_physical_block in b_physical_blocks
    assert reuse_step.sampled_token_ids == {
        request_c.request_id: expected_c[0],
        request_a.request_id: expected_a[1],
    }
    assert reuse_step.finished_requests == [request_c]
    assert request_c.is_finished
    assert request_c.request_id not in kv_cache.block_tables
    assert request_a.output_token_ids == expected_a[:2]

    for output_index in range(2, request_a.max_new_tokens):
        decode_step = run_generation_step(model, scheduler)

        assert decode_step.scheduler_output.prefill_requests == []
        assert decode_step.scheduler_output.decode_requests == [request_a]
        assert decode_step.model_batch.request_ids == [request_a.request_id]
        assert decode_step.model_batch.request_phases == [
            RequestStatus.DECODING,
        ]
        assert decode_step.sampled_token_ids == {
            request_a.request_id: expected_a[output_index],
        }
        assert request_a.output_token_ids == expected_a[: output_index + 1]

        if output_index < request_a.max_new_tokens - 1:
            assert decode_step.finished_requests == []
            assert request_a.status is RequestStatus.DECODING
        else:
            assert decode_step.finished_requests == [request_a]

    assert request_a.is_finished
    assert request_a.output_token_ids == expected_a
    assert scheduler.running == []
    assert list(scheduler.waiting) == []
    assert kv_cache.block_tables == {}
    assert len(kv_cache.free_blocks) == kv_cache.num_blocks
