import runpy
from pathlib import Path

import torch


diagnostic = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "diagnose_decode_numerics.py"
    )
)
metrics = diagnostic["metrics"]


def test_metrics_keeps_strict_and_bf16_acceptance_counts_distinct():
    actual = torch.tensor([0.6, 0.1, 0.0])
    reference = torch.zeros(3)

    strict = metrics(actual, reference)
    bfloat16_acceptance = metrics(actual, reference, atol=0.5, rtol=0.1)

    assert strict["different_elements"] == 2
    assert strict["violating_elements"] == 2
    assert bfloat16_acceptance["different_elements"] == 2
    assert bfloat16_acceptance["violating_elements"] == 1


def test_metrics_applies_relative_tolerance_per_element():
    actual = torch.tensor([11.4, 11.6])
    reference = torch.tensor([10.0, 10.0])

    report = metrics(actual, reference, atol=0.5, rtol=0.1)

    assert report["violating_elements"] == 1
