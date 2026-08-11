import math

import torch

from src.model.lead_confidence import (
    compute_lead_confidence,
    compute_lead_coverage,
    compute_lead_smoothness,
    lead_confidence_by_name,
)


def _sine_lead(num_samples: int = 500) -> torch.Tensor:
    t = torch.linspace(0, 8 * math.pi, num_samples)
    return torch.sin(t)


def test_coverage_full_valid_lead_is_one() -> None:
    lines = _sine_lead().unsqueeze(0)
    coverage = compute_lead_coverage(lines)
    assert coverage.shape == (1,)
    assert coverage.item() == 1.0


def test_coverage_all_nan_lead_is_zero() -> None:
    lines = torch.full((1, 100), float("nan"))
    coverage = compute_lead_coverage(lines)
    assert coverage.item() == 0.0


def test_coverage_partial_nan_lead_matches_fraction() -> None:
    lines = _sine_lead(100).unsqueeze(0)
    lines[0, :40] = float("nan")
    coverage = compute_lead_coverage(lines)
    assert abs(coverage.item() - 0.6) < 1e-6


def test_smoothness_clean_signal_is_near_one() -> None:
    lines = _sine_lead().unsqueeze(0)
    smoothness = compute_lead_smoothness(lines)
    assert smoothness.item() > 0.99


def test_smoothness_penalizes_single_spike() -> None:
    clean = _sine_lead().unsqueeze(0)
    spiky = clean.clone()
    spiky[0, 250] += 50.0  # A single huge, isolated discontinuity amid otherwise smooth samples.

    clean_score = compute_lead_smoothness(clean).item()
    spiky_score = compute_lead_smoothness(spiky).item()

    assert spiky_score < clean_score
    assert spiky_score < 1.0


def test_smoothness_too_few_samples_defaults_to_one() -> None:
    lines = torch.tensor([[1.0, float("nan"), float("nan")]])
    smoothness = compute_lead_smoothness(lines)
    assert smoothness.item() == 1.0


def test_compute_lead_confidence_is_product_of_components() -> None:
    lines = torch.stack([_sine_lead(100), torch.full((100,), float("nan"))])
    scores = compute_lead_confidence(lines)

    assert set(scores.keys()) == {"coverage", "smoothness", "confidence"}
    assert torch.allclose(scores["confidence"], scores["coverage"] * scores["smoothness"])
    assert scores["confidence"][1].item() == 0.0  # Fully-NaN lead must score zero confidence.


def test_lead_confidence_by_name_keys_and_types() -> None:
    lines = torch.stack([_sine_lead(50), _sine_lead(50)])
    result = lead_confidence_by_name(lines, lead_names=["I", "II"])

    assert set(result.keys()) == {"I", "II"}
    for lead_scores in result.values():
        assert set(lead_scores.keys()) == {"coverage", "smoothness", "confidence"}
        for value in lead_scores.values():
            assert isinstance(value, float)
