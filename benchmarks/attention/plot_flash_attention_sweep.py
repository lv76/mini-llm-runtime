import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "results"
INPUT_PATH = RESULTS_DIR / "flash_attention_sweep.csv"
LATENCY_OUTPUT_PATH = RESULTS_DIR / "flash_attention_latency.png"
RATIO_OUTPUT_PATH = RESULTS_DIR / "flash_attention_ratio.png"


def load_results(csv_path):
    required_columns = {
        "seq_len",
        "sdpa_auto_median_ms",
        "cudnn_median_ms",
        "triton_median_ms",
    }

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = required_columns.difference(
            reader.fieldnames or []
        )
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"CSV is missing required columns: {missing}"
            )

        results = []
        for row in reader:
            sdpa_auto = float(row["sdpa_auto_median_ms"])
            triton = float(row["triton_median_ms"])
            results.append(
                {
                    "seq_len": int(row["seq_len"]),
                    "sdpa_auto_median_ms": sdpa_auto,
                    "cudnn_median_ms": float(
                        row["cudnn_median_ms"]
                    ),
                    "triton_median_ms": triton,
                    "triton_over_sdpa": triton / sdpa_auto,
                }
            )

    if not results:
        raise ValueError(f"CSV contains no benchmark rows: {csv_path}")

    results.sort(key=lambda result: result["seq_len"])
    return results


def print_results_table(results):
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


def plot_latency(results, output_path):
    seq_lens = [result["seq_len"] for result in results]
    sdpa_auto = [
        result["sdpa_auto_median_ms"]
        for result in results
    ]
    cudnn = [
        result["cudnn_median_ms"]
        for result in results
    ]
    triton = [
        result["triton_median_ms"]
        for result in results
    ]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        seq_lens,
        sdpa_auto,
        marker="o",
        linewidth=2.2,
        label="PyTorch SDPA Auto",
    )
    axis.plot(
        seq_lens,
        cudnn,
        marker="s",
        linewidth=2.2,
        label="cuDNN SDPA",
    )
    axis.plot(
        seq_lens,
        triton,
        marker="^",
        linewidth=2.2,
        label="Triton FlashAttention",
    )

    axis.set_title(
        "FlashAttention Forward Latency by Sequence Length"
    )
    axis.set_xlabel("Sequence Length")
    axis.set_ylabel("Median Latency (ms)")
    axis.set_xticks(seq_lens)
    axis.grid(True, linestyle="--", alpha=0.45)
    axis.legend(frameon=True)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def plot_ratio(results, output_path):
    seq_lens = [result["seq_len"] for result in results]
    ratios = [
        result["triton_over_sdpa"]
        for result in results
    ]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        seq_lens,
        ratios,
        color="tab:purple",
        marker="o",
        linewidth=2.2,
        label="Triton / PyTorch SDPA Auto",
    )
    axis.axhline(
        1.0,
        color="tab:red",
        linestyle="--",
        linewidth=1.6,
        label="Equal latency",
    )

    axis.set_title(
        "Triton FlashAttention Relative Latency"
    )
    axis.set_xlabel("Sequence Length")
    axis.set_ylabel("Triton / SDPA Latency Ratio")
    axis.set_xticks(seq_lens)
    axis.grid(True, linestyle="--", alpha=0.45)
    axis.legend(frameon=True)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main():
    results = load_results(INPUT_PATH)
    print_results_table(results)

    plot_latency(results, LATENCY_OUTPUT_PATH)
    plot_ratio(results, RATIO_OUTPUT_PATH)

    print(f"\nSaved latency plot to: {LATENCY_OUTPUT_PATH}")
    print(f"Saved ratio plot to:   {RATIO_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
