import runpy
from pathlib import Path

import pytest
import torch


benchmark = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / 'benchmarks'
        / 'benchmark_kv_cache_memory.py'
    )
)
compute_layout = benchmark['compute_layout']
expand_values = benchmark['expand_values']
tensor_storage_bytes = benchmark['tensor_storage_bytes']


def test_compute_layout_compares_equal_service_capacity():
    layout = compute_layout(
        prompt_lens=[32, 128, 512, 1024],
        output_lens=[8],
        block_size=16,
        max_num_seqs=4,
    )

    assert layout.request_capacity_tokens == [39, 135, 519, 1031]
    assert layout.useful_token_slots == 1724
    assert layout.paged_blocks == 110
    assert layout.paged_token_slots == 1760
    assert layout.dense_token_slots == 4124
    assert layout.paged_packing_efficiency == pytest.approx(1724 / 1760)
    assert layout.dense_packing_efficiency == pytest.approx(1724 / 4124)
    assert layout.paged_saving_vs_dense == pytest.approx(1 - 1760 / 4124)


def test_compute_layout_selects_largest_concurrent_requests():
    layout = compute_layout(
        prompt_lens=[8, 64, 128],
        output_lens=[2, 4, 8],
        block_size=16,
        max_num_seqs=2,
    )

    assert layout.selected_capacity_tokens == [135, 67]
    assert layout.paged_blocks == 14
    assert layout.dense_token_slots == 270


def test_expand_values_broadcasts_or_rejects_wrong_length():
    assert expand_values([8], 3, 'output_lens') == [8, 8, 8]
    assert expand_values([1, 2], 2, 'output_lens') == [1, 2]
    with pytest.raises(ValueError, match='output_lens'):
        expand_values([1, 2], 3, 'output_lens')


def test_tensor_storage_bytes_uses_dtype_size():
    assert tensor_storage_bytes((2, 3, 4), torch.bfloat16) == 48
    assert tensor_storage_bytes((2, 3, 4), torch.float32) == 96
