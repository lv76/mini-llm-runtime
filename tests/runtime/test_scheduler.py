import pytest
from mini_llm_runtime.runtime.request import Request, RequestStatus, FinishReason
from mini_llm_runtime.runtime.scheduler import Scheduler, SchedulerOutput
from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache

def make_request(
    request_id: str,
    prompt_token_ids: list[int],
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=4,
        arrival_time=0.0,
    )

def make_scheduler(
    max_num_batched_tokens: int = 8,
    max_num_seqs: int = 2,
    num_blocks: int = 16,
    block_size: int = 4,
) -> Scheduler:
    kv_cache = PagedKVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=4,
        device="cpu",
    )

    return Scheduler(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        kv_cache=kv_cache,
    )


def test_scheduler_output_empty():
    output = SchedulerOutput()

    assert output.prefill_requests == []
    assert output.decode_requests == []
    assert output.num_scheduled_tokens == {}
    assert output.scheduled_requests == []
    assert output.total_num_scheduled_tokens == 0
    assert output.is_empty is True


def test_scheduler_output_mixed_requests():
    request_a = make_request("A", [10, 11, 12])
    request_b = make_request("B", [20, 21])

    request_a.start_prefill()
    request_b.status = RequestStatus.DECODING

    output = SchedulerOutput(
        prefill_requests=[request_a],
        decode_requests=[request_b],
        num_scheduled_tokens={
            "A": 3,
            "B": 1,
        },
    )

    assert output.prefill_requests == [request_a]
    assert output.decode_requests == [request_b]
    assert output.num_scheduled_tokens == {"A": 3, "B": 1}

    assert output.scheduled_requests == [request_a, request_b]
    assert output.total_num_scheduled_tokens == 4
    assert output.is_empty is False


def test_scheduler_output_rejects_duplicate_request():
    request_a = make_request("A", [10, 11])

    with pytest.raises(ValueError):
        SchedulerOutput(
            prefill_requests=[request_a],
            decode_requests=[request_a],
            num_scheduled_tokens={"A": 1},
        )


def test_scheduler_output_rejects_missing_token_count():
    request_a = make_request("A", [10, 11])

    with pytest.raises(ValueError):
        SchedulerOutput(
            prefill_requests=[request_a],
            num_scheduled_tokens={},
        )


def test_scheduler_output_rejects_extra_token_count():
    with pytest.raises(ValueError):
        SchedulerOutput(
            num_scheduled_tokens={"A": 1},
        )


def test_scheduler_output_rejects_nonpositive_token_count():
    request_a = make_request("A", [10, 11])

    with pytest.raises(ValueError):
        SchedulerOutput(
            prefill_requests=[request_a],
            num_scheduled_tokens={"A": 0},
        )


def test_scheduler_initial_state():
    scheduler = make_scheduler()

    assert scheduler.max_num_batched_tokens == 8
    assert scheduler.max_num_seqs == 2
    assert list(scheduler.waiting) == []
    assert scheduler.running == []


def test_scheduler_add_request_preserves_fifo_order():
    scheduler = make_scheduler()
    request_a = make_request("A", [10, 11])
    request_b = make_request("B", [20, 21])

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    assert list(scheduler.waiting) == [request_a, request_b]
    assert scheduler.running == []
    assert request_a.status is RequestStatus.WAITING
    assert request_b.status is RequestStatus.WAITING


def test_scheduler_rejects_duplicate_request_id():
    scheduler = make_scheduler()
    request_a1 = make_request("A", [10, 11])
    request_a2 = make_request("A", [20, 21])

    scheduler.add_request(request_a1)

    with pytest.raises(ValueError):
        scheduler.add_request(request_a2)


def test_scheduler_rejects_nonwaiting_request():
    scheduler = make_scheduler()
    request_a = make_request("A", [10, 11])
    request_a.status = RequestStatus.DECODING

    with pytest.raises(ValueError):
        scheduler.add_request(request_a)

def test_scheduler_update_after_prefill_moves_request_to_decoding():
    scheduler = make_scheduler()
    request_a = make_request("A", [10, 11, 12])

    scheduler.add_request(request_a)
    scheduler_output = scheduler.schedule()

    finished_requests = scheduler.update_after_model(
        scheduler_output=scheduler_output,
        sampled_token_ids={"A": 100},
    )

    assert finished_requests == []
    assert request_a.status is RequestStatus.DECODING
    assert request_a.num_computed_tokens == 3
    assert request_a.output_token_ids == [100]
    assert request_a.pending_token_id == 100
    assert scheduler.running == [request_a]


def test_scheduler_update_after_model_removes_finished_request():
    scheduler = make_scheduler()
    request_a = make_request("A", [10, 11])

    scheduler.add_request(request_a)
    scheduler_output = scheduler.schedule()

    finished_requests = scheduler.update_after_model(
        scheduler_output=scheduler_output,
        sampled_token_ids={"A": 2},
        reached_eos_request_ids={"A"},
    )

    assert finished_requests == [request_a]
    assert request_a.status is RequestStatus.FINISHED
    assert request_a.is_finished is True
    assert scheduler.running == []

    # 完成后的 request_id 不再占用 Scheduler。
    replacement = make_request("A", [20, 21])
    scheduler.add_request(replacement)
    assert list(scheduler.waiting) == [replacement]

def test_scheduler_schedules_decode_request():
    scheduler = make_scheduler()
    request_a = make_request("A", [10, 11, 12])

    scheduler.add_request(request_a)

    prefill_output = scheduler.schedule()
    scheduler.update_after_model(
        scheduler_output=prefill_output,
        sampled_token_ids={"A": 100},
    )

    assert request_a.status is RequestStatus.DECODING
    assert request_a.pending_token_id == 100

    decode_output = scheduler.schedule()

    assert decode_output.prefill_requests == []
    assert decode_output.decode_requests == [request_a]
    assert decode_output.num_scheduled_tokens == {"A": 1}
    assert decode_output.total_num_scheduled_tokens == 1

    assert scheduler.running == [request_a]
    assert request_a.status is RequestStatus.DECODING
    assert request_a.num_computed_tokens == 3
    assert request_a.pending_token_id == 100

def test_scheduler_mixes_decode_and_new_prefill():
    scheduler = Scheduler(
        max_num_batched_tokens=5,
        max_num_seqs=2,
    )

    request_b = make_request("B", [20, 21])
    scheduler.add_request(request_b)

    first_output = scheduler.schedule()
    scheduler.update_after_model(
        scheduler_output=first_output,
        sampled_token_ids={"B": 100},
    )

    request_a = make_request("A", [10, 11, 12])
    scheduler.add_request(request_a)

    mixed_output = scheduler.schedule()

    assert mixed_output.decode_requests == [request_b]
    assert mixed_output.prefill_requests == [request_a]
    assert mixed_output.num_scheduled_tokens == {
        "B": 1,
        "A": 3,
    }
    assert mixed_output.total_num_scheduled_tokens == 4

    assert scheduler.running == [request_b, request_a]
    assert request_b.status is RequestStatus.DECODING
    assert request_a.status is RequestStatus.PREFILLING
    assert list(scheduler.waiting) == []

def test_scheduler_prioritizes_decode_over_prefill():
    scheduler = Scheduler(
        max_num_batched_tokens=3,
        max_num_seqs=3,
    )

    request_b = make_request("B", [20])
    request_c = make_request("C", [30])

    scheduler.add_request(request_b)
    scheduler.add_request(request_c)

    first_output = scheduler.schedule()
    scheduler.update_after_model(
        scheduler_output=first_output,
        sampled_token_ids={
            "B": 100,
            "C": 200,
        },
    )

    request_a = make_request("A", [10, 11])
    scheduler.add_request(request_a)

    second_output = scheduler.schedule()

    assert second_output.decode_requests == [request_b, request_c]
    assert second_output.prefill_requests == []
    assert second_output.num_scheduled_tokens == {
        "B": 1,
        "C": 1,
    }

    assert list(scheduler.waiting) == [request_a]
    assert request_a.status is RequestStatus.WAITING


def test_scheduler_full_lifecycle_finishes_on_length():
    scheduler = Scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=2,
        kv_cache=PagedKVCache(
            num_layers=1,
            num_blocks=16,
            block_size=4,
            num_kv_heads=1,
            head_dim=4,
            device="cpu",
        ),
    )

    request_a = Request(
        request_id="A",
        prompt_token_ids=[10, 11],
        max_new_tokens=3,
        arrival_time=0.0,
    )

    scheduler.add_request(request_a)

    # Step 1：prefill 计算两个 prompt token，并生成 token 100。
    prefill_output = scheduler.schedule()

    finished_requests = scheduler.update_after_model(
        scheduler_output=prefill_output,
        sampled_token_ids={"A": 100},
    )

    assert finished_requests == []
    assert request_a.status is RequestStatus.DECODING
    assert request_a.num_computed_tokens == 2
    assert request_a.output_token_ids == [100]

    # Step 2：计算 token 100，并生成 token 101。
    first_decode_output = scheduler.schedule()

    assert first_decode_output.decode_requests == [request_a]
    assert first_decode_output.num_scheduled_tokens == {"A": 1}

    finished_requests = scheduler.update_after_model(
        scheduler_output=first_decode_output,
        sampled_token_ids={"A": 101},
    )

    assert finished_requests == []
    assert request_a.status is RequestStatus.DECODING
    assert request_a.num_computed_tokens == 3
    assert request_a.output_token_ids == [100, 101]

    # Step 3：计算 token 101，并生成第三个输出 token 102。
    second_decode_output = scheduler.schedule()

    finished_requests = scheduler.update_after_model(
        scheduler_output=second_decode_output,
        sampled_token_ids={"A": 102},
    )

    assert finished_requests == [request_a]
    assert request_a.status is RequestStatus.FINISHED
    assert request_a.finish_reason is FinishReason.LENGTH
    assert request_a.num_computed_tokens == 4
    assert request_a.output_token_ids == [100, 101, 102]

    assert scheduler.running == []
    assert list(scheduler.waiting) == []


def test_scheduler_add_request_registers_kv_sequence():
    scheduler = make_scheduler()
    request_a = make_request("A", [10, 11, 12])

    scheduler.add_request(request_a)

    assert "A" in scheduler.kv_cache.block_tables
    assert scheduler.kv_cache.block_tables["A"] == []
    assert scheduler.kv_cache.seq_lens["A"] == 0

    assert request_a.block_table is scheduler.kv_cache.block_tables["A"]

    # waiting 请求只注册 sequence，暂时不占用物理 blocks。
    assert len(scheduler.kv_cache.free_blocks) == 16

def test_scheduler_prefill_reserves_kv_blocks():
    scheduler = make_scheduler(
        num_blocks=3,
        block_size=4,
    )
    request = make_request("A", [10, 11, 12, 13, 14])  # 5 tokens，需要 2 blocks
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.prefill_requests == [request]
    assert len(request.block_table) == 2
    assert request.block_table is scheduler.kv_cache.block_tables["A"]
    assert len(scheduler.kv_cache.free_blocks) == 1

    # Scheduler 只预留地址，还没有真正写入 K/V
    assert scheduler.kv_cache.seq_lens["A"] == 0

def test_scheduler_does_not_schedule_prefill_when_kv_cache_is_full():
    scheduler = make_scheduler(
        num_blocks=1,
        block_size=4,
    )
    request = make_request("A", [10, 11, 12, 13, 14])  # 需要 2 blocks
    scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.is_empty
    assert list(scheduler.waiting) == [request]
    assert scheduler.running == []
    assert request.status is RequestStatus.WAITING

    # reserve_tokens 必须是原子的，失败后不能占用部分 block
    assert request.block_table == []
    assert len(scheduler.kv_cache.free_blocks) == 1

def test_scheduler_decode_reserves_new_kv_block():
    scheduler = make_scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=1,
        num_blocks=2,
        block_size=4,
    )
    request = make_request("A", [10, 11, 12, 13])
    scheduler.add_request(request)

    prefill_output = scheduler.schedule()
    assert prefill_output.prefill_requests == [request]
    assert len(request.block_table) == 1

    request.update_after_model(
        num_new_computed_tokens=4,
        sampled_token_id=20,
    )

    decode_output = scheduler.schedule()

    assert decode_output.decode_requests == [request]
    assert decode_output.num_scheduled_tokens == {"A": 1}
    assert len(request.block_table) == 2
    assert len(scheduler.kv_cache.free_blocks) == 0

def test_scheduler_skips_decode_request_without_kv_capacity():
    scheduler = make_scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=2,
        num_blocks=2,
        block_size=4,
    )
    request_a = make_request("A", [10, 11, 12, 13])
    request_b = make_request("B", [20, 21])

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    scheduler.schedule()

    request_a.update_after_model(
        num_new_computed_tokens=4,
        sampled_token_id=30,
    )
    request_b.update_after_model(
        num_new_computed_tokens=2,
        sampled_token_id=40,
    )

    # 当前没有空闲 block：
    # A 的第 5 个 token 需要新 block，无法调度
    # B 的第 3 个 token 仍可使用已有 block
    assert len(scheduler.kv_cache.free_blocks) == 0

    output = scheduler.schedule()

    assert output.decode_requests == [request_b]
    assert output.num_scheduled_tokens == {"B": 1}
    assert request_a.status is RequestStatus.DECODING
    assert request_a in scheduler.running

def test_scheduler_frees_kv_blocks_when_request_finishes():
    scheduler = make_scheduler(
        max_num_batched_tokens=8,
        max_num_seqs=1,
        num_blocks=2,
        block_size=4,
    )
    request_a = Request(
        request_id="A",
        prompt_token_ids=[10, 11, 12, 13, 14],
        max_new_tokens=1,
        arrival_time=0.0,
    )
    scheduler.add_request(request_a)

    output_a = scheduler.schedule()

    allocated_blocks = set(request_a.block_table)
    assert len(allocated_blocks) == 2
    assert len(scheduler.kv_cache.free_blocks) == 0

    # 第一个采样 token 就达到 max_new_tokens
    finished_requests = scheduler.update_after_model(
        scheduler_output=output_a,
        sampled_token_ids={"A": 99},
    )

    assert finished_requests == [request_a]
    assert request_a.is_finished
    assert request_a not in scheduler.running
    assert "A" not in scheduler.request_ids

    assert "A" not in scheduler.kv_cache.block_tables
    assert "A" not in scheduler.kv_cache.seq_lens
    assert len(scheduler.kv_cache.free_blocks) == 2
    assert request_a.block_table == []

    # 新请求能够复用刚释放的物理 blocks
    request_b = Request(
        request_id="B",
        prompt_token_ids=[20, 21, 22, 23, 24],
        max_new_tokens=1,
        arrival_time=1.0,
    )
    scheduler.add_request(request_b)

    output_b = scheduler.schedule()

    assert output_b.prefill_requests == [request_b]
    assert set(request_b.block_table) == allocated_blocks