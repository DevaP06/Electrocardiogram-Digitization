from typing import Dict

import torch

DEFAULT_LEAD_NAMES: list[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def compute_lead_coverage(canonical_lines: torch.Tensor) -> torch.Tensor:
    """
    Fraction of valid (non-NaN) samples per lead in a canonicalized signal tensor.

    A lead that failed to digitize -- because it was never detected, cropped away, or dropped during
    line merging -- shows up as an (partially) NaN row rather than a loud error. Coverage turns that
    silent failure into a number: a lead with coverage near 0 was effectively not recovered at all.

    Args:
        canonical_lines: Tensor of shape (num_leads, num_samples), NaN where no signal was recovered.

    Returns:
        Tensor of shape (num_leads,) with values in [0, 1].
    """
    valid = ~torch.isnan(canonical_lines)
    coverage: torch.Tensor = valid.float().mean(dim=1)
    return coverage


def compute_lead_smoothness(canonical_lines: torch.Tensor, jump_z_threshold: float = 8.0) -> torch.Tensor:
    """
    Penalizes leads whose *valid* samples contain abrupt, isolated discontinuities -- a symptom of
    mis-stitched trace fragments (see SignalExtractor.match_and_merge_lines) that coverage alone
    would not catch, since the samples are technically non-NaN.

    For each lead, sample-to-sample jumps are compared against a robust (median-absolute-deviation
    based) estimate of the lead's own typical jump size, so the threshold adapts to each lead's
    amplitude/noise level rather than using a fixed physical unit cutoff.

    Args:
        canonical_lines: Tensor of shape (num_leads, num_samples), NaN where no signal was recovered.
        jump_z_threshold: Robust z-score above which a sample-to-sample jump counts as a discontinuity.

    Returns:
        Tensor of shape (num_leads,) with values in [0, 1]. 1.0 means no anomalous jumps (or too few
        valid samples to judge); lower values mean a larger fraction of jumps were anomalous.
    """
    num_leads = canonical_lines.shape[0]
    scores = torch.ones(num_leads, dtype=torch.float32)

    for lead_idx in range(num_leads):
        valid = canonical_lines[lead_idx][~torch.isnan(canonical_lines[lead_idx])]
        if valid.numel() < 3:
            continue  # Not enough samples to judge smoothness; leave the neutral default of 1.0.

        abs_diffs = valid.diff().abs()
        median_abs_diff = abs_diffs.median()
        mad = (abs_diffs - median_abs_diff).abs().median()
        robust_scale = mad * 1.4826 + 1e-6  # 1.4826 makes MAD a consistent estimator of std for Gaussian data.

        z_scores = abs_diffs / robust_scale
        jump_fraction = (z_scores > jump_z_threshold).float().mean()
        scores[lead_idx] = 1.0 - jump_fraction

    return scores


def compute_lead_confidence(canonical_lines: torch.Tensor, jump_z_threshold: float = 8.0) -> Dict[str, torch.Tensor]:
    """
    Per-lead confidence for a canonicalized, digitized ECG signal, combining coverage (was the lead
    reconstructed at all) and smoothness (is the reconstruction free of stitching artifacts).

    Args:
        canonical_lines: Tensor of shape (num_leads, num_samples), NaN where no signal was recovered.
        jump_z_threshold: Forwarded to compute_lead_smoothness.

    Returns:
        Dict with "coverage", "smoothness", and "confidence" (their product) tensors, each of shape
        (num_leads,) with values in [0, 1].
    """
    coverage = compute_lead_coverage(canonical_lines)
    smoothness = compute_lead_smoothness(canonical_lines, jump_z_threshold)
    return {"coverage": coverage, "smoothness": smoothness, "confidence": coverage * smoothness}


def lead_confidence_by_name(
    canonical_lines: torch.Tensor, lead_names: list[str] = DEFAULT_LEAD_NAMES, jump_z_threshold: float = 8.0
) -> Dict[str, Dict[str, float]]:
    """
    Same as compute_lead_confidence, but keyed by lead name with plain Python floats -- convenient for
    logging, CSV export, or JSON serialization at the edges of the pipeline.

    Args:
        canonical_lines: Tensor of shape (num_leads, num_samples), NaN where no signal was recovered.
        lead_names: Names for each row of canonical_lines, in order.
        jump_z_threshold: Forwarded to compute_lead_smoothness.

    Returns:
        Dict mapping lead name to a dict of {"coverage", "smoothness", "confidence"} floats.
    """
    scores = compute_lead_confidence(canonical_lines, jump_z_threshold)
    return {name: {key: float(scores[key][i].item()) for key in scores} for i, name in enumerate(lead_names)}
