import runpy
from pathlib import Path

import pytest
import torch

from mini_llm_runtime.model.qwen2 import TinyQwen2Config, TinyQwen2ForCausalLM


benchmark = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / 'benchmarks'
        / 'benchmark_scheduler_runtime.py'
    )
)
percentile = benchmark['percentile']
latency_summary = benchmark['latency_summary']
summarize_samples = benchmark['summarize_samples']
parse_positive_int_list = benchmark['parse_positive_int_list']
parse_nonnegative_float_list = benchmark['parse_nonnegative_float_list']
expand_request_values = benchmark['expand_request_values']
top2_snapshot = benchmark['top2_snapshot']
build_first_divergences = benchmark['build_first_divergences']
dense_greedy_tokens = benchmark['dense_greedy_tokens']
isolated_paged_greedy_tokens = benchmark['isolated_paged_greedy_tokens']
run_workload = benchmark['run_workload']


def test_percentile_uses_nearest_rank():
    values = [5.0, 1.0, 4.0, 2.0, 3.0]

    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 0.95) == 5.0


def test_latency_summary_reports_request_level_percentiles():
    assert latency_summary([1.0, 2.0, 9.0]) == {
        'mean': 4.0,
        'p50': 2.0,
        'p95': 9.0,
        'p99': 9.0,
        'max': 9.0,
    }
    assert latency_summary([]) == {
        'mean': 0.0,
        'p50': 0.0,
        'p95': 0.0,
        'p99': 0.0,
        'max': 0.0,
    }


def test_parse_and_expand_ragged_workload_values():
    assert parse_positive_int_list('8, 32,128') == [8, 32, 128]
    assert parse_nonnegative_float_list('0, 12.5,40') == [0.0, 12.5, 40.0]
    assert expand_request_values([16], 3, 'output_lens') == [16, 16, 16]
    assert expand_request_values([1, 2], 2, 'output_lens') == [1, 2]

    with pytest.raises(ValueError, match='output_lens'):
        expand_request_values([1, 2], 3, 'output_lens')


@pytest.mark.parametrize('probability', [0.0, -0.1, 1.1])
def test_percentile_rejects_invalid_probability(probability):
    with pytest.raises(ValueError, match='probability'):
        percentile([1.0], probability)


def test_summarize_samples_reports_latency_and_correctness():
    rows = []
    for value, greedy_match in [(10.0, True), (20.0, True), (30.0, False)]:
        rows.append({
            'ttft_mean_ms': value,
            'ttft_p50_ms': value,
            'ttft_p95_ms': value + 0.5,
            'ttft_p99_ms': value + 0.75,
            'queue_mean_ms': value / 2,
            'queue_p95_ms': value / 2 + 0.5,
            'tpot_mean_ms': value + 1,
            'itl_p95_ms': value + 1.5,
            'itl_p99_ms': value + 1.75,
            'e2e_max_ms': value + 2,
            'output_tokens_per_second': 1000 / value,
            'request_throughput_per_second': 100 / value,
            'incremental_peak_allocated_mib': value + 3,
            'peak_kv_utilization': value / 100,
            'kv_pool_allocated_mib': value + 4,
            'kv_pool_saving_vs_dense_static': value / 200,
            'peak_paged_committed_mib': value + 5,
            'peak_block_packing_efficiency': value / 100,
            'peak_saving_vs_dense_padded': value / 200,
            'greedy_match': greedy_match,
        })

    summary = summarize_samples(rows)

    assert summary['num_samples'] == 3
    assert summary['ttft_mean_ms']['p50'] == 20.0
    assert summary['ttft_mean_ms']['p95'] == 30.0
    assert not summary['all_greedy_match']
    assert not summary['all_runtime_match']


def test_top2_snapshot_reports_margin_and_token_ids():
    snapshot = top2_snapshot(torch.tensor([1.0, 4.5, 4.0]))

    assert snapshot == {
        'top1_token_id': 1,
        'top1_logit': 4.5,
        'top2_token_id': 2,
        'top2_logit': 4.0,
        'top2_margin': 0.5,
    }


def test_top2_snapshot_uses_argmax_tie_breaking():
    snapshot = top2_snapshot(torch.tensor([1.0, 4.5, 4.5]))

    assert snapshot['top1_token_id'] == 1
    assert snapshot['top2_token_id'] == 2
    assert snapshot['top2_margin'] == 0.0


def test_top2_snapshot_rejects_non_finite_logits():
    with pytest.raises(ValueError, match='finite'):
        top2_snapshot(torch.tensor([1.0, float('nan')]))


def test_first_divergence_stops_before_histories_differ():
    dense = [[10, 20, 30]]
    isolated = [[10, 21, 40]]
    batched = [[10, 20, 50]]
    snapshots = [[[
        {'top1_token_id': token, 'top1_logit': 1.0,
         'top2_token_id': 0, 'top2_logit': 0.0, 'top2_margin': 1.0}
        for token in tokens[0]
    ]] for tokens in (dense, isolated, batched)]

    reports = build_first_divergences(
        dense, isolated, batched, snapshots[0], snapshots[1], snapshots[2]
    )

    assert len(reports) == 1
    assert reports[0]['output_index'] == 1
    assert reports[0]['shared_prefix_token_ids'] == [10]
    assert reports[0]['paths']['dense']['chosen_token_id'] == 20
    assert reports[0]['paths']['isolated_paged']['chosen_token_id'] == 21


def test_staggered_ragged_workload_reports_per_request_latencies():
    torch.manual_seed(7)
    config = TinyQwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    model = TinyQwen2ForCausalLM(config).eval()
    prompts = [[1, 2, 3], [4, 5]]
    output_lens = [2, 1]
    dense = dense_greedy_tokens(model, prompts, output_lens)
    isolated = isolated_paged_greedy_tokens(
        model, config, prompts, output_lens, block_size=8
    )

    result = run_workload(
        model,
        config,
        prompts,
        output_lens,
        8,
        dense,
        isolated,
        arrival_times_ms=[0.0, 0.5],
        max_num_seqs=1,
        max_num_batched_tokens=3,
        kv_cache_blocks=1,
    )

    assert result['runtime_match']
    assert result['all_kv_blocks_released']
    assert [item['output_len'] for item in result['request_metrics']] == [2, 1]
    assert [item['arrival_ms'] for item in result['request_metrics']] == [0.0, 0.5]
    assert result['request_metrics'][1]['queue_ms'] > 0
    assert result['ttft_p95_ms'] >= result['ttft_p50_ms']
    assert result['itl_p95_ms'] >= 0
    assert result['kv_bytes_per_token'] == 64
    assert result['kv_pool_allocated_mib'] == pytest.approx(8 * 64 / 2**20)
    assert result['peak_useful_kv_mib'] <= result['peak_paged_committed_mib']
    assert 0 < result['peak_block_packing_efficiency'] <= 1
