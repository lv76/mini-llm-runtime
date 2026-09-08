import pytest
import torch

from mini_llm_runtime.runtime.paged_kv_cache import PagedKVCache


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))]


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('slots', [[], [11], [11, 0, 7, 4, 2]])
def test_batched_slot_write_matches_copy_reference(device, dtype, slots):
    torch.manual_seed(42)
    cache = PagedKVCache(2, 3, 4, 2, 8, dtype=dtype, device=device)
    cache.k_cache.normal_()
    cache.v_cache.normal_()
    expected_k, expected_v = cache.k_cache.clone(), cache.v_cache.clone()
    indices = torch.tensor(slots, dtype=torch.int32, device=device)
    # Non-contiguous sources with conversion from FP32, like the old copy_ path.
    k = torch.randn(len(slots), 8, 2, device=device).transpose(1, 2)
    v = torch.randn_like(k)
    for i, slot in enumerate(slots):
        expected_k[1, slot // 4, slot % 4].copy_(k[i])
        expected_v[1, slot // 4, slot % 4].copy_(v[i])
    cache.write_slots(indices, k, v, layer_idx=1)
    assert torch.equal(cache.k_cache, expected_k)
    assert torch.equal(cache.v_cache, expected_v)
    # Reusing previously written slots in a later call is allowed.
    cache.write_slots(indices, k + 1, v + 1, layer_idx=1)
    for i, slot in enumerate(slots):
        expected_k[1, slot // 4, slot % 4].copy_(k[i] + 1)
        expected_v[1, slot // 4, slot % 4].copy_(v[i] + 1)
    assert torch.equal(cache.k_cache, expected_k)
    assert torch.equal(cache.v_cache, expected_v)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('slots,error,message', [
    ([0, 0], ValueError, 'distinct'),
    ([0, -1], IndexError, 'out-of-range'),
    ([0, 12], IndexError, 'out-of-range'),
    ([0., 1.], TypeError, 'integer'),
])
def test_invalid_slot_write_does_not_modify_cache(device, slots, error, message):
    cache = PagedKVCache(2, 3, 4, 2, 8, dtype=torch.float32, device=device)
    cache.k_cache.fill_(17)
    cache.v_cache.fill_(23)
    with pytest.raises(error, match=message):
        cache.write_slots(torch.tensor(slots, device=device),
                          torch.randn(2, 2, 8, device=device),
                          torch.randn(2, 2, 8, device=device), layer_idx=1)
    assert torch.all(cache.k_cache == 17)
    assert torch.all(cache.v_cache == 23)
