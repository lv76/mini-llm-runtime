import pytest

from mini_llm_runtime.runtime.request import Request, RequestStatus, FinishReason

def test_request_properties(): 
    request_a = Request(
        request_id="A",
        prompt_token_ids=[10, 11, 12, 13, 14],
        max_new_tokens=3,
        arrival_time=0.0,
    )

    assert request_a.prompt_len == 5
    assert request_a.num_output_tokens == 0
    assert request_a.num_all_tokens == 5
    assert request_a.pending_token_id is None
    assert request_a.is_finished is False
    print("Test 1 passed: Initial properties are correct.")

    request_a.status = RequestStatus.DECODING

    request_a.num_computed_tokens = 5
    request_a.output_token_ids.append(20)

    assert request_a.num_output_tokens == 1
    assert request_a.num_all_tokens == 6
    assert request_a.pending_token_id == 20
    assert request_a.is_finished is False
    print("test simulation is correct")

def test_request_mutable_lists_are_independent():
    request_a = Request(
        request_id="A",
        prompt_token_ids=[10, 11],
        max_new_tokens=3,
        arrival_time=0.0,
    )

    request_b = Request(
        request_id="B",
        prompt_token_ids=[20, 21],
        max_new_tokens=3,
        arrival_time=1.0,
    )

    request_a.output_token_ids.append(30)
    request_a.block_table.extend([7, 3])

    assert request_a.output_token_ids == [30]
    assert request_a.block_table == [7, 3]

    assert request_b.output_token_ids == []
    assert request_b.block_table == []

def test_request_finished_properties():
    request = Request(
        request_id="A",
        prompt_token_ids=[10, 11],
        max_new_tokens=1,
        arrival_time=0.0,
    )

    request.status = RequestStatus.FINISHED
    request.finish_reason = FinishReason.LENGTH

    assert request.is_finished
    assert request.finish_reason is FinishReason.LENGTH

def test_start_prefill():
    request = Request(
        request_id="A",
        prompt_token_ids=[10, 11, 12, 13, 14],
        max_new_tokens=3,
        arrival_time=0.0,
    )

    # BlockManager应在开始Prefill前完成block分配。
    request.block_table.extend([7, 3])

    request.start_prefill()

    assert request.status is RequestStatus.PREFILLING
    assert request.num_computed_tokens == 0
    assert request.output_token_ids == []
    assert request.block_table == [7, 3]


def test_start_prefill_rejects_invalid_status():
    request = Request(
        request_id="A",
        prompt_token_ids=[10, 11],
        max_new_tokens=3,
        arrival_time=0.0,
        status=RequestStatus.DECODING,
    )

    with pytest.raises(RuntimeError):
        request.start_prefill()

def test_update_after_model():
    request = Request(
        request_id="A",
        prompt_token_ids=[10, 11, 12, 13, 14],
        max_new_tokens=3,
        arrival_time=0.0,
    )

    request.block_table.extend([7, 3])
    request.start_prefill()

    # Prefill完成，采样20
    request.update_after_model(
        num_new_computed_tokens=5,
        sampled_token_id=20,
    )

    assert request.status is RequestStatus.DECODING
    assert request.num_computed_tokens == 5
    assert request.output_token_ids == [20]
    assert request.pending_token_id == 20
    assert request.finish_reason is None

    # 第一次Decode，输入20并采样21
    request.update_after_model(
        num_new_computed_tokens=1,
        sampled_token_id=21,
    )

    assert request.status is RequestStatus.DECODING
    assert request.num_computed_tokens == 6
    assert request.output_token_ids == [20, 21]

    # 第二次Decode，输入21并采样22
    request.update_after_model(
        num_new_computed_tokens=1,
        sampled_token_id=22,
    )

    assert request.status is RequestStatus.FINISHED
    assert request.finish_reason is FinishReason.LENGTH
    assert request.num_computed_tokens == 7
    assert request.output_token_ids == [20, 21, 22]

    # Request只标记完成，BlockManager尚未执行释放。
    assert request.block_table == [7, 3]

def test_update_after_model_finishes_on_eos():
    request = Request(
        request_id="A",
        prompt_token_ids=[10, 11],
        max_new_tokens=5,
        arrival_time=0.0,
    )

    request.block_table.append(7)
    request.start_prefill()

    request.update_after_model(
        num_new_computed_tokens=2,
        sampled_token_id=2,
        reached_eos=True,
    )

    assert request.status is RequestStatus.FINISHED
    assert request.finish_reason is FinishReason.EOS
    assert request.output_token_ids == [2]
    assert request.num_computed_tokens == 2

    # BlockManager还没有释放资源。
    assert request.block_table == [7]

def test_update_after_model_rejects_invalid_status():
    request = Request(
        request_id="A",
        prompt_token_ids=[10, 11],
        max_new_tokens=3,
        arrival_time=0.0,
    )

    with pytest.raises(RuntimeError):
        request.update_after_model(
            num_new_computed_tokens=2,
            sampled_token_id=20,
        )

    assert request.status is RequestStatus.WAITING
    assert request.num_computed_tokens == 0
    assert request.output_token_ids == []