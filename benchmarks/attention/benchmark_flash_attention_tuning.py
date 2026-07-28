import csv
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from mini_llm_runtime.kernels.triton.flash_attention import flash_fwd_triton


WARMUP_ITERS = 10
ITERATIONS = 100
REPEATS = 5

CONFIGS = [
    {
        "name": "bm16_bn64_w4_s2",
        "block_m": 16,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
    {
        "name": "bm16_bn64_w4_s1",
        "block_m": 16,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 1,
    },
    {
        "name": "bm16_bn32_w4_s2",
        "block_m": 16,
        "block_n": 32,
        "num_warps": 4,
        "num_stages": 2,
    },
    {
        "name": "bm16_bn64_w8_s2",
        "block_m": 16,
        "block_n": 64,
        "num_warps": 8,
        "num_stages": 2,
    },
    {
        "name": "bm32_bn64_w4_s2",
        "block_m": 32,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
    {
        "name": "bm64_bn64_w4_s2",
        "block_m": 64,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
    {
        "name": "bm64_bn64_w4_s1",
        "block_m": 64,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 1,
    },
    {
        "name": "bm64_bn64_w4_s3",
        "block_m": 64,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 3,
    },
    {
        "name": "bm64_bn64_w8_s2",
        "block_m": 64,
        "block_n": 64,
        "num_warps": 8,
        "num_stages": 2,
    },
    {
        "name": "bm64_bn128_w4_s2",
        "block_m": 64,
        "block_n": 128,
        "num_warps": 4,
        "num_stages": 2,
    },
    {
        "name": "bm128_bn64_w4_s2",
        "block_m": 128,
        "block_n": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
]


def run_flash_fwd(q, k, v, config):
    output, _ = flash_fwd_triton(
        q,
        k,
        v,
        is_causal=False,
        block_m=config["block_m"],
        block_n=config["block_n"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return output


def benchmark_repeat(q, k, v, config):
    for _ in range(WARMUP_ITERS):
        run_flash_fwd(q, k, v, config)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERATIONS):
        run_flash_fwd(q, k, v, config)
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / ITERATIONS


def benchmark_callable(fn):
    for _ in range(WARMUP_ITERS):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERATIONS):
        fn()
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / ITERATIONS


def save_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "block_m",
        "block_n",
        "num_warps",
        "num_stages",
        "status",
        "latency_run_1_ms",
        "latency_run_2_ms",
        "latency_run_3_ms",
        "latency_run_4_ms",
        "latency_run_5_ms",
        "median_latency_ms",
        "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "name": result["name"],
                "block_m": result["block_m"],
                "block_n": result["block_n"],
                "num_warps": result["num_warps"],
                "num_stages": result["num_stages"],
                "status": result["status"],
                "median_latency_ms": result["median_latency_ms"],
                "error": result["error"],
            }

            for index in range(REPEATS):
                key = f"latency_run_{index + 1}_ms"
                if index < len(result["latencies_ms"]):
                    row[key] = result["latencies_ms"][index]
                else:
                    row[key] = ""

            writer.writerow(row)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    batch_size = 1
    num_heads = 8
    seq_len = 2048
    head_dim = 64
    dtype = torch.float16
    device = "cuda"
    is_causal = False

    torch.manual_seed(0)

    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    q_triton = q.reshape(batch_size * num_heads, seq_len, head_dim)
    k_triton = k.reshape(batch_size * num_heads, seq_len, head_dim)
    v_triton = v.reshape(batch_size * num_heads, seq_len, head_dim)

    with torch.inference_mode():
        def run_sdpa():
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=is_causal,
            )

        reference = run_sdpa()

        print("\nBenchmarking PyTorch SDPA Auto...")
        sdpa_latencies = [
            benchmark_callable(run_sdpa)
            for _ in range(REPEATS)
        ]
        sdpa_median = statistics.median(sdpa_latencies)
        print(
            "  raw runs: "
            f"{[round(value, 6) for value in sdpa_latencies]} ms"
        )
        print(f"  median latency: {sdpa_median:.6f} ms")

        print("\nBenchmarking cuDNN SDPA...")
        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            cudnn_latencies = [
                benchmark_callable(run_sdpa)
                for _ in range(REPEATS)
            ]
        cudnn_median = statistics.median(cudnn_latencies)
        print(
            "  raw runs: "
            f"{[round(value, 6) for value in cudnn_latencies]} ms"
        )
        print(f"  median latency: {cudnn_median:.6f} ms")

        results = []

        for config in CONFIGS:
            print(f"\nRunning {config['name']}...")

            latencies_ms = []
            status = "passed"
            error = ""
            median_latency_ms = ""

            try:
                actual = run_flash_fwd(
                    q_triton,
                    k_triton,
                    v_triton,
                    config,
                ).reshape(shape)

                torch.testing.assert_close(
                    actual,
                    reference,
                    rtol=1e-2,
                    atol=1e-2,
                )

                for _ in range(REPEATS):
                    latencies_ms.append(
                        benchmark_repeat(
                            q_triton,
                            k_triton,
                            v_triton,
                            config,
                        )
                    )

                median_latency_ms = statistics.median(latencies_ms)

                print(
                    "  raw runs: "
                    f"{[round(value, 6) for value in latencies_ms]} ms"
                )
                print(f"  median latency: {median_latency_ms:.6f} ms")
            except Exception as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
                print(
                    "  raw runs: "
                    f"{[round(value, 6) for value in latencies_ms]} ms"
                )
                print("  median latency: failed")
                print(f"  failed: {error}")

            results.append(
                {
                    **config,
                    "status": status,
                    "latencies_ms": latencies_ms,
                    "median_latency_ms": median_latency_ms,
                    "error": error,
                }
            )

    successful_results = [
        result
        for result in results
        if result["status"] == "passed"
    ]
    successful_results.sort(
        key=lambda result: result["median_latency_ms"]
    )

    print("\nReferences:")
    print(f"  PyTorch SDPA Auto: {sdpa_median:.6f} ms")
    print(f"  cuDNN SDPA:        {cudnn_median:.6f} ms")

    print("\nRanking (fastest to slowest):")
    if successful_results:
        for rank, result in enumerate(successful_results, start=1):
            print(
                f"  {rank}. {result['name']}: "
                f"{result['median_latency_ms']:.6f} ms"
            )
    else:
        print("  No configuration completed successfully.")

    output_path = (
        Path(__file__).resolve().parent
        / "results"
        / "flash_attention_tuning_s2048.csv"
    )
    save_results(results, output_path)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
