import argparse
import csv
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from mini_llm_runtime.kernels.triton.flash_attention import flash_fwd_triton


WARMUP_ITERS = 200
ITERATIONS = 100
REPEATS = 5

SEQ_LENS = [128, 256, 512, 1024, 2048]

TRITON_CONFIG = {
    "block_m": 64,
    "block_n": 64,
    "num_warps": 4,
    "num_stages": 2,
}


def warmup_callable(fn):
    for _ in range(WARMUP_ITERS):
        fn()

    torch.cuda.synchronize()


def benchmark_callable(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERATIONS):
        fn()
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / ITERATIONS


def print_latencies(name, latencies, median_latency):
    print(f"\n{name}")
    print(
        "  raw runs: "
        f"{[round(value, 6) for value in latencies]} ms"
    )
    print(f"  median latency: {median_latency:.6f} ms")


def save_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seq_len",
        "sdpa_auto_median_ms",
        "cudnn_median_ms",
        "triton_median_ms",
        "triton_over_sdpa",
    ]

    for implementation in ("sdpa_auto", "cudnn", "triton"):
        for repeat in range(1, REPEATS + 1):
            fieldnames.append(
                f"{implementation}_run_{repeat}_ms"
            )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "seq_len": result["seq_len"],
                "sdpa_auto_median_ms": result["sdpa_auto_median_ms"],
                "cudnn_median_ms": result["cudnn_median_ms"],
                "triton_median_ms": result["triton_median_ms"],
                "triton_over_sdpa": result["triton_over_sdpa"],
            }

            for implementation in ("sdpa_auto", "cudnn", "triton"):
                latencies = result[f"{implementation}_latencies_ms"]
                for repeat, latency in enumerate(latencies, start=1):
                    row[f"{implementation}_run_{repeat}_ms"] = latency

            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FlashAttention across sequence lengths."
        )
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        help=(
            "Benchmark only this sequence length. "
            "By default, run the full sequence-length sweep."
        ),
    )
    args = parser.parse_args()

    if args.seq_len is not None and args.seq_len <= 0:
        parser.error("--seq-len must be a positive integer.")

    return args


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    batch_size = 1
    num_heads = 8
    head_dim = 64
    dtype = torch.float16
    device = "cuda"
    is_causal = False

    torch.manual_seed(0)

    seq_lens = (
        [args.seq_len]
        if args.seq_len is not None
        else SEQ_LENS
    )
    results = []

    with torch.inference_mode():
        for seq_len in seq_lens:
            shape = (
                batch_size,
                num_heads,
                seq_len,
                head_dim,
            )
            q = torch.randn(shape, device=device, dtype=dtype)
            k = torch.randn(shape, device=device, dtype=dtype)
            v = torch.randn(shape, device=device, dtype=dtype)

            q_triton = q.reshape(
                batch_size * num_heads,
                seq_len,
                head_dim,
            )
            k_triton = k.reshape(
                batch_size * num_heads,
                seq_len,
                head_dim,
            )
            v_triton = v.reshape(
                batch_size * num_heads,
                seq_len,
                head_dim,
            )

            def run_sdpa():
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )

            def run_triton():
                output, _ = flash_fwd_triton(
                    q_triton,
                    k_triton,
                    v_triton,
                    is_causal=is_causal,
                    block_m=TRITON_CONFIG["block_m"],
                    block_n=TRITON_CONFIG["block_n"],
                    num_warps=TRITON_CONFIG["num_warps"],
                    num_stages=TRITON_CONFIG["num_stages"],
                )
                return output

            out_sdpa = run_sdpa()
            out_triton = run_triton().reshape(shape)
            torch.testing.assert_close(
                out_triton,
                out_sdpa,
                rtol=1e-2,
                atol=1e-2,
            )
            del out_sdpa, out_triton

            print(f"\nSequence length: {seq_len}")

            warmup_callable(run_sdpa)
            sdpa_auto_latencies = [
                benchmark_callable(run_sdpa)
                for _ in range(REPEATS)
            ]
            sdpa_auto_median = statistics.median(
                sdpa_auto_latencies
            )
            print_latencies(
                "PyTorch SDPA Auto",
                sdpa_auto_latencies,
                sdpa_auto_median,
            )

            with sdpa_kernel(
                backends=[SDPBackend.CUDNN_ATTENTION]
            ):
                warmup_callable(run_sdpa)
                cudnn_latencies = [
                    benchmark_callable(run_sdpa)
                    for _ in range(REPEATS)
                ]
            cudnn_median = statistics.median(cudnn_latencies)
            print_latencies(
                "cuDNN SDPA",
                cudnn_latencies,
                cudnn_median,
            )

            warmup_callable(run_triton)
            triton_latencies = [
                benchmark_callable(run_triton)
                for _ in range(REPEATS)
            ]
            triton_median = statistics.median(triton_latencies)
            print_latencies(
                "Triton FlashAttention",
                triton_latencies,
                triton_median,
            )

            results.append(
                {
                    "seq_len": seq_len,
                    "sdpa_auto_latencies_ms": sdpa_auto_latencies,
                    "sdpa_auto_median_ms": sdpa_auto_median,
                    "cudnn_latencies_ms": cudnn_latencies,
                    "cudnn_median_ms": cudnn_median,
                    "triton_latencies_ms": triton_latencies,
                    "triton_median_ms": triton_median,
                    "triton_over_sdpa": (
                        triton_median / sdpa_auto_median
                    ),
                }
            )

    print("\nSummary:")
    print(
        f"{'Seq Len':>7} | "
        f"{'SDPA Auto':>10} | "
        f"{'cuDNN':>10} | "
        f"{'Triton':>10} | "
        f"{'Triton / SDPA':>13}"
    )
    print("-" * 65)
    for result in results:
        print(
            f"{result['seq_len']:7d} | "
            f"{result['sdpa_auto_median_ms']:10.6f} | "
            f"{result['cudnn_median_ms']:10.6f} | "
            f"{result['triton_median_ms']:10.6f} | "
            f"{result['triton_over_sdpa']:13.6f}"
        )

    if args.seq_len is None:
        output_filename = "flash_attention_sweep.csv"
    else:
        output_filename = (
            f"flash_attention_sweep_s{args.seq_len}.csv"
        )

    output_path = (
        Path(__file__).resolve().parent
        / "results"
        / output_filename
    )
    save_results(results, output_path)
    print(f"\nSaved results to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
