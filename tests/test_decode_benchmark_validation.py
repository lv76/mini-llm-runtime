import runpy
import sys
from pathlib import Path

import pytest
import torch


benchmark = runpy.run_path(str(
    Path(__file__).resolve().parents[1] / 'benchmarks/benchmark_decode_runtime.py'
))
compare_logits = benchmark['compare_logits']
validation_status = benchmark['validation_status']
parse_args = benchmark['parse_args']
select_replay_case = benchmark['select_replay_case']


def test_parse_args_supports_bf16_reduction_ab_flag(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['benchmark_decode_runtime.py'])
    assert not parse_args().disable_bf16_reduced_precision_reduction

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'benchmark_decode_runtime.py',
            '--disable-bf16-reduced-precision-reduction',
        ],
    )
    assert parse_args().disable_bf16_reduced_precision_reduction


def test_select_replay_case_requires_exact_saved_inputs():
    case = {
        'batch_size': 2,
        'context_lens': [4, 4],
        'prompt_token_ids': [[1, 2, 3], [4, 5, 6]],
        'decode_token_ids': [7, 8],
    }

    assert select_replay_case([case], 2, [4, 4]) is case

    with pytest.raises(ValueError, match='exactly one case'):
        select_replay_case([case], 2, [8, 8])


def test_select_replay_case_rejects_wrong_prompt_length():
    case = {
        'batch_size': 1,
        'context_lens': [4],
        'prompt_token_ids': [[1, 2]],
        'decode_token_ids': [3],
    }

    with pytest.raises(ValueError, match='context_len - 1'):
        select_replay_case([case], 1, [4])


def test_sparse_outlier_stays_failed_despite_small_average_error():
    reference = torch.zeros(2, 10000)
    actual = reference.clone()
    actual[1, 123] = 0.546875
    report = compare_logits(actual, reference, atol=0.5, rtol=0.1)
    assert not report['allclose']
    assert report['violating_elements'] == 1
    assert report['per_request'][0]['violating_elements'] == 0
    assert report['per_request'][1]['violating_elements'] == 1
    assert report['per_request'][1]['mean_abs_error'] < 0.0001
    assert validation_status({'batch': report}, [1.0]) == 'failed_logit_tolerance'


def test_greedy_flip_is_reported_even_if_logits_pass():
    reference = torch.tensor([[1., 1.0625]])
    actual = torch.tensor([[1.0625, 1.]])
    report = compare_logits(actual, reference, atol=0.5, rtol=0.1)
    assert report['allclose']
    assert report['per_request'][0]['reference_top2_margin'] == 0.0625
    assert validation_status({'batch': report}, [0.0]) == 'logits_pass_greedy_mismatch'


def test_identical_logits_pass():
    logits = torch.tensor([[1., 2., 3.]])
    report = compare_logits(logits, logits, atol=1e-4, rtol=1e-4)
    assert report['per_request'][0]['normalized_rmse'] == 0
    assert validation_status({'batch': report}, [1.0]) == 'logits_and_greedy_pass'


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_values_always_fail(value):
    with pytest.raises(AssertionError, match='non-finite'):
        compare_logits(torch.tensor([[0., value]]), torch.zeros(1, 2), atol=.5, rtol=.1)
