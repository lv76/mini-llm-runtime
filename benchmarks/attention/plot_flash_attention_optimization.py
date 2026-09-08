import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "results"
INPUT_PATH = RESULTS_DIR / "flash_attention_optimization.csv"
OUTPUT_PATH = RESULTS_DIR / "flash_attention_optimization.png"
BLOCK_M_PROGRESSION = [16, 32, 64]


def load_results(csv_path):
    required_columns = {
        "config",
        "block_m",
        "block_n",
        "num_warps",
        "num_stages",
        "median_latency_ms",
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

        results_by_block_m = {}
        for row in reader:
            block_m = int(row["block_m"])
            if block_m not in BLOCK_M_PROGRESSION:
                continue

            latency = float(row["median_latency_ms"])
            if latency <= 0:
                raise ValueError(
                    f"Latency must be positive for BLOCK_M={block_m}."
                )

            results_by_block_m[block_m] = {
                "config": row["config"],
                "block_m": block_m,
                "median_latency_ms": latency,
            }

    missing_blocks = [
        block_m
        for block_m in BLOCK_M_PROGRESSION
        if block_m not in results_by_block_m
    ]
    if missing_blocks:
        missing = ", ".join(
            f"BM{block_m}" for block_m in missing_blocks
        )
        raise ValueError(
            f"CSV is missing optimization results for: {missing}"
        )

    return [
        results_by_block_m[block_m]
        for block_m in BLOCK_M_PROGRESSION
    ]


def latency_reduction(before, after):
    return (before - after) / before * 100.0


def print_reductions(results):
    bm16_latency = results[0]["median_latency_ms"]
    bm32_latency = results[1]["median_latency_ms"]
    bm64_latency = results[2]["median_latency_ms"]

    print("Relative latency reduction:")
    print(
        "  BM16 -> BM32: "
        f"{latency_reduction(bm16_latency, bm32_latency):.2f}%"
    )
    print(
        "  BM32 -> BM64: "
        f"{latency_reduction(bm32_latency, bm64_latency):.2f}%"
    )
    print(
        "  BM16 -> BM64: "
        f"{latency_reduction(bm16_latency, bm64_latency):.2f}%"
    )


def plot_optimization(results, output_path):
    labels = [
        f"BM{result['block_m']}"
        for result in results
    ]
    latencies = [
        result["median_latency_ms"]
        for result in results
    ]

    figure, axis = plt.subplots(figsize=(8, 5.5))
    bars = axis.bar(
        labels,
        latencies,
        color=["#8AB6D6", "#4C86B7", "#175D98"],
        width=0.62,
    )

    axis.set_title(
        "Triton FlashAttention Forward Optimization"
    )
    axis.set_xlabel("Q Tile Size (BLOCK_M)")
    axis.set_ylabel("Median Latency (ms)")
    axis.set_ylim(0, max(latencies) * 1.18)
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )
    axis.set_axisbelow(True)

    for bar, latency in zip(bars, latencies, strict=True):
        axis.annotate(
            f"{latency:.6f} ms",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

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
    print_reductions(results)
    plot_optimization(results, OUTPUT_PATH)
    print(f"\nSaved optimization plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
