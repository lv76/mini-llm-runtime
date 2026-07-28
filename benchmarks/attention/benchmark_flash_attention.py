import torch
import torch.nn.functional as F
import statistics

from mini_llm_runtime.kernels.triton.flash_attention import flash_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.profiler import profile, ProfilerActivity

def make_inputs(batch_size, num_heads, seq_len, head_dim):
    device = "cuda"
    dtype = torch.float16

    shape = (batch_size, num_heads, seq_len, head_dim)
    Q = torch.randn(shape, device = device, dtype = dtype)
    K = torch.randn(shape, device = device, dtype = dtype)
    V = torch.randn(shape, device = device, dtype = dtype)

    return Q, K, V


def benchmark_one(fn, warmup_iters = 2, iterations = 100):
    for _ in range(warmup_iters):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(iterations):
        fn()

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations

def benchmark_median(fn, repeats=5):
    results = []

    for _ in range(repeats):
        latency_ms = benchmark_one(fn)
        results.append(latency_ms)

    return statistics.median(results), results

def main():

    batch_size = 1
    num_heads = 8
    head_dim = 64
    seq_lens = [2048]

    for seq_len in seq_lens:
        q, k, v = make_inputs(
            batch_size,
            num_heads,
            seq_len,
            head_dim
        )

        B, H, S, D = q.shape
        q_triton = q.reshape(B*H, S, D)
        k_triton = k.reshape(B*H, S, D)
        v_triton = v.reshape(B*H, S, D)

        fn_triton = lambda : flash_attention(
            q_triton,
            k_triton,
            v_triton,
            is_causal=False
        )

        fn_sdpa = lambda : F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        ) 

        def run_flash_sdpa(q, k, v):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=False,
        )

        fn_sdpa_flash = lambda: run_flash_sdpa(q, k, v)
        
        triton_ms, triton_runs = benchmark_median(fn_triton, repeats=5)
        sdpa_ms, sdpa_runs = benchmark_median(fn_sdpa, repeats=5)
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            cudnn_ms, cudnn_runs = benchmark_median(
                fn_sdpa,
                repeats=5,
            )
        

        print(f"S={seq_len:4d}")
        print(f"  Triton runs: {[round(x, 4) for x in triton_runs]}")
        print(f"  SDPA runs:   {[round(x, 4) for x in sdpa_runs]}")
        print(f"  cuDNN run: {[round(x, 4) for x in cudnn_runs]}")
        print(
            f"  Median | "
            f"Triton={triton_ms:.4f} ms | "
            f"SDPA={sdpa_ms:.4f} ms | "
            f"cuDNN={cudnn_ms:.4f} ms"
        )



if __name__ == "__main__":
    main()