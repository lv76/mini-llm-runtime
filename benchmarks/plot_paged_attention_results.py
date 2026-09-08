"""Plot publication-friendly PagedAttention benchmark figures from CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


SEQ_REQUIRED_FIELDS = {
    "seq_len",
    "partition_size",
    "num_partitions",
    "block_n",
    "num_warps",
    "split_kernel_ms",
    "single_pass_ms",
    "split_vs_single",
}
PARTITION_REQUIRED_FIELDS = {
    "partition_size",
    "num_partitions",
    "split_kernel_ms",
    "single_pass_ms",
    "split_vs_single",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PagedAttention performance figures from benchmark CSVs."
    )
    parser.add_argument(
        "--seq-csv",
        type=Path,
        required=True,
        help="sequence-length scaling CSV from --seq-split-configs",
    )
    parser.add_argument(
        "--partition-csv",
        type=Path,
        required=True,
        help="partition granularity CSV from --partition-sizes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for generated PNG figures",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = set(reader.fieldnames or ())
        missing = required_fields - fields
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"benchmark CSV is empty: {path}")
    return rows


def positive_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if value <= 0:
        raise ValueError(f"{field} must be positive, got {value}")
    return value


def positive_int(row: dict[str, str], field: str) -> int:
    value = int(row[field])
    if value <= 0:
        raise ValueError(f"{field} must be positive, got {value}")
    return value


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "font.size": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def configure_log2_sequence_axis(ax: plt.Axes, seq_lens: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lens)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()


def plot_sequence_latency(
    rows: list[dict[str, str]], output_path: Path, dpi: int
) -> None:
    data = sorted(
        (
            positive_int(row, "seq_len"),
            positive_float(row, "single_pass_ms"),
            positive_float(row, "split_kernel_ms"),
        )
        for row in rows
    )
    seq_lens = [item[0] for item in data]
    if len(seq_lens) != len(set(seq_lens)):
        raise ValueError("sequence scaling CSV must contain one row per seq_len")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(
        seq_lens,
        [item[1] for item in data],
        marker="o",
        label="Tuned Single-pass",
    )
    ax.plot(
        seq_lens,
        [item[2] for item in data],
        marker="s",
        label="Split-KV",
    )
    configure_log2_sequence_axis(ax, seq_lens)
    ax.set_title("PagedAttention Decode Latency vs. Sequence Length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_sequence_speedup(
    rows: list[dict[str, str]], output_path: Path, dpi: int
) -> None:
    data = sorted(
        (
            positive_int(row, "seq_len"),
            positive_float(row, "single_pass_ms")
            / positive_float(row, "split_kernel_ms"),
        )
        for row in rows
    )
    seq_lens = [item[0] for item in data]
    if len(seq_lens) != len(set(seq_lens)):
        raise ValueError("sequence scaling CSV must contain one row per seq_len")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(
        seq_lens,
        [item[1] for item in data],
        marker="o",
        color="#2a9d8f",
        label="Split-KV",
    )
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.4, label="Break-even")
    configure_log2_sequence_axis(ax, seq_lens)
    ax.set_title("Split-KV Speedup over Tuned Single-pass")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Speedup over Tuned Single-pass")
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_partition_sweep(
    rows: list[dict[str, str]], output_path: Path, dpi: int
) -> None:
    data = sorted(
        (
            positive_int(row, "num_partitions"),
            positive_float(row, "split_kernel_ms"),
        )
        for row in rows
    )
    num_partitions = [item[0] for item in data]
    if len(num_partitions) != len(set(num_partitions)):
        raise ValueError(
            "partition sweep CSV must contain one row per num_partitions"
        )
    latencies = [item[1] for item in data]
    best_index = min(range(len(data)), key=lambda index: latencies[index])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(num_partitions, latencies, marker="o", color="#e76f51")
    ax.scatter(
        [num_partitions[best_index]],
        [latencies[best_index]],
        color="#264653",
        s=55,
        zorder=3,
        label="Lowest latency",
    )
    ax.annotate(
        f"{latencies[best_index]:.4f} ms",
        (num_partitions[best_index], latencies[best_index]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(num_partitions)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_title("Split-KV Partition Granularity at S=4096")
    ax.set_xlabel("Number of KV Partitions")
    ax.set_ylabel("Split-KV Latency (ms)")
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.dpi < 200:
        raise ValueError("--dpi must be at least 200")

    seq_rows = read_csv(args.seq_csv, SEQ_REQUIRED_FIELDS)
    partition_rows = read_csv(args.partition_csv, PARTITION_REQUIRED_FIELDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    plot_sequence_latency(
        seq_rows,
        args.output_dir / "paged_attention_seq_scaling.png",
        args.dpi,
    )
    plot_sequence_speedup(
        seq_rows,
        args.output_dir / "paged_attention_speedup.png",
        args.dpi,
    )
    plot_partition_sweep(
        partition_rows,
        args.output_dir / "paged_attention_partition_sweep.png",
        args.dpi,
    )


if __name__ == "__main__":
    main()
