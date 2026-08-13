import argparse
import os
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import resample
from scipy.stats import pearsonr
from tqdm import tqdm
from yacs.config import CfgNode as CN

from src.config.default import get_cfg
from src.utils import find_config_path


def crop_ahus_raw_timeseries(gt: npt.NDArray[Any], target_length: int = 5000) -> npt.NDArray[Any]:
    """Truncates ground truth that runs longer than the digitized output it will be compared against.

    target_length defaults to 5000 for backwards compatibility, but callers should pass the prediction's
    own length: the digitized sample count is set by LAYOUT_IDENTIFIER.target_num_samples, which is 5000
    for the AHUS configs this originally assumed but 10000 for
    src/config/inference_wrapper_george-moody-2024.yml. Hardcoding 5000 silently cropped ground truth to
    half the prediction's length there, which surfaced as a broadcast error rather than a wrong score --
    the lucky case; the same mismatch in the other direction would have scored silently.
    """
    if gt.shape[0] > target_length:
        return gt[:target_length]
    return gt


def find_ground_truth_csvs(gt_dir: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for entry in os.scandir(gt_dir):
        if entry.is_dir():
            expected = os.path.join(entry.path, f"digital_signal_{entry.name}.csv")
            if os.path.isfile(expected):
                results.append((entry.name, expected))
    return results


def find_digitized_csv(gt_folder: str) -> dict[str, str]:
    results: dict[str, str] = {}
    if not os.path.isdir(gt_folder):
        return results
    for fname in os.listdir(gt_folder):
        if fname.endswith("_timeseries_canonical.csv"):
            stem = fname[: -len("_timeseries_canonical.csv")]
            results[stem] = os.path.join(gt_folder, fname)
    return results


def load_signal_csv(path: str) -> npt.NDArray[Any]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return data.T if data.shape[0] < data.shape[1] else data  # (leads, samples)


def resample_to_length(signal: npt.NDArray[Any], target_length: int) -> npt.NDArray[Any]:
    leads, _ = signal.shape
    return np.stack([resample(signal[lead], target_length) for lead in range(leads)])


def shift_and_crop(arr: npt.NDArray[Any], shift: int) -> npt.NDArray[Any]:
    if shift == 0:
        return arr
    elif shift > 0:
        return arr[..., shift:]
    else:
        return arr[..., :shift]


def rescale_lead_time_axis(lead_signal: npt.NDArray[Any], scale: float) -> npt.NDArray[Any]:
    """Resamples one lead onto a time axis stretched by `scale`, anchored at its first valid sample.

    NaN gaps are preserved rather than interpolated across: in a multi-column layout each lead is only
    drawn over part of the record, and those gaps are meaningful (they mark where the lead simply is not
    on the paper), so filling them would invent signal and inflate the score.

    Args:
        lead_signal: (samples,) one lead, NaN where no signal was recovered.
        scale: Multiplier on the time axis. <1 compresses, >1 stretches.

    Returns:
        Array of the same length, resampled, still NaN outside the (rescaled) valid region.
    """
    if scale == 1.0:
        return lead_signal
    valid = np.where(~np.isnan(lead_signal))[0]
    if valid.size < 2:
        return lead_signal

    length = lead_signal.shape[0]
    source_positions = valid[0] + (valid - valid[0]) * scale
    low = int(np.ceil(source_positions[0]))
    high = min(int(np.floor(source_positions[-1])), length - 1)
    if high <= low:
        return lead_signal

    out: npt.NDArray[Any] = np.full(length, np.nan)
    grid = np.arange(low, high + 1)
    out[grid] = np.interp(grid, source_positions, lead_signal[valid])
    return out


def build_scale_grid(scale_min: float, scale_max: float, scale_step: float) -> list[float]:
    """Scales to search. Defaults collapse to [1.0], i.e. the original shift-only behavior."""
    if scale_step <= 0 or scale_max <= scale_min:
        return [1.0]
    grid = np.arange(scale_min, scale_max + scale_step / 2, scale_step)
    return [float(s) for s in grid]


def compute_metrics(
    gt: npt.NDArray[Any],
    pred: npt.NDArray[Any],
    n_translations: int = 0,
    scales: Optional[list[float]] = None,
) -> dict[str, list[float]]:
    """Scores digitized leads against ground truth, per lead, keeping the best-SNR alignment.

    A rigid shift alone is often not enough to align these signals. The digitized time axis is not
    guaranteed to map 1:1 onto ground-truth sample indices -- LeadIdentifier.normalize() crops to the
    detected signal extent and stretches that to target_num_samples, so if the detected extent differs
    from the true paper width the result is a *scale* error, not just an offset. Measured on real output
    this was a consistent ~3.5% compression, and because no shift can correct a scale error it dragged
    reported correlation from ~0.95 down to ~0.5 on waveforms that actually matched well. Searching
    scale alongside shift separates "the digitizer got the waveform wrong" from "the time axis is
    slightly rescaled", which are very different failures.

    Args:
        gt: (samples, leads) ground truth in microvolts.
        pred: (samples, leads) digitized output in microvolts.
        n_translations: Width of the symmetric shift search window, in samples.
        scales: Time-axis scales to search. None or [1.0] reproduces the original shift-only scoring.

    Returns:
        Per-lead lists of pearson, rms, snr_db, shift, scale and nans_fraction.
    """
    gt = crop_ahus_raw_timeseries(gt, pred.shape[0])
    gt, pred = gt.T, pred.T  # (leads, samples) in units of mV
    scale_grid = scales if scales else [1.0]

    metrics: dict[str, list[float]] = {
        "pearson": [],
        "rms": [],
        "snr_db": [],
        "shift": [],
        "scale": [],
        "nans_fraction": [],
    }
    max_shift = n_translations // 2
    min_shift = -n_translations // 2

    if n_translations == 1:
        shifts = [0]
    else:
        shifts = np.arange(min_shift, max_shift + 1).tolist()

    for lead in range(gt.shape[0]):
        best_snr = -np.inf
        best_metrics = {
            "pearson": np.nan,
            "rms": np.nan,
            "snr_db": np.nan,
            "shift": 0,
            "scale": 1.0,
            "nans_fraction": 0.0,
        }
        for scale in scale_grid:
            pred_scaled = rescale_lead_time_axis(pred[lead], scale)
            for shift in shifts:
                if shift == 0:
                    gt_aligned = gt[lead]
                    pred_shifted = pred_scaled
                elif shift > 0:
                    gt_aligned = gt[lead][shift:]
                    pred_shifted = pred_scaled[:-shift]
                else:  # shift < 0
                    gt_aligned = gt[lead][:shift]
                    pred_shifted = pred_scaled[-shift:]

                mask = ~(np.isnan(gt_aligned) | np.isnan(pred_shifted))
                if not np.any(mask):
                    continue
                gt_valid = gt_aligned[mask].copy()
                pred_valid = pred_shifted[mask].copy()
                gt_valid -= np.mean(gt_valid)
                pred_valid -= np.mean(pred_valid)

                try:
                    pearson = pearsonr(gt_valid, pred_valid)[0]
                except Exception:
                    pearson = np.nan
                rms = float(np.sqrt(np.mean((gt_valid - pred_valid) ** 2)))
                power_signal = float(np.mean(gt_valid**2))
                power_noise = float(np.mean((gt_valid - pred_valid) ** 2))
                snr_db = 10 * np.log10(power_signal / power_noise) if power_noise > 0 else np.nan

                if best_snr < snr_db:
                    best_snr = snr_db
                    best_metrics = {
                        "pearson": pearson,
                        "rms": rms,
                        "snr_db": snr_db,
                        "scale": scale,
                        "shift": shift,
                        "nans_fraction": np.mean(np.isnan(pred_shifted)),
                    }

        metrics["pearson"].append(best_metrics["pearson"])
        metrics["rms"].append(best_metrics["rms"])
        metrics["snr_db"].append(best_metrics["snr_db"])
        metrics["shift"].append(best_metrics["shift"])
        metrics["scale"].append(best_metrics["scale"])
        metrics["nans_fraction"].append(best_metrics["nans_fraction"])

    return metrics


def main(cfg: CN) -> None:
    digitized_dir: str = cfg["digitized_dir"]
    ground_truth_dir: str = cfg["ground_truth_dir"]
    results_csv: str = cfg["results_csv"]
    n_translations: int = cfg.get("shift_steps", 0)
    # Defaults collapse the grid to [1.0], preserving the original shift-only behavior for existing
    # configs. Set scale_max > scale_min to enable the search -- see compute_metrics for why.
    scales = build_scale_grid(
        float(cfg.get("scale_min", 1.0)), float(cfg.get("scale_max", 1.0)), float(cfg.get("scale_step", 0.0))
    )
    if len(scales) > 1:
        print(f"Searching {len(scales)} time scales from {scales[0]:.4f} to {scales[-1]:.4f}")

    gt_files = find_ground_truth_csvs(ground_truth_dir)
    if not gt_files:
        print("No ground truth files found.")
        return

    results: list[dict[str, Any]] = []

    for folder, gt_csv in tqdm(gt_files):
        digitized_folder = os.path.join(digitized_dir, folder)
        digitized_csvs = find_digitized_csv(digitized_folder)

        gt_signal = load_signal_csv(gt_csv)
        for stem, digitized_csv in digitized_csvs.items():
            digitized_signal = load_signal_csv(digitized_csv)

            metrics = compute_metrics(gt_signal, digitized_signal, n_translations, scales)

            row: dict[str, Any] = {
                "folder": folder,
                "gt_csv": os.path.basename(gt_csv),
                "digitized_csv": os.path.basename(digitized_csv),
                **{f"pearson_{i+1}": v for i, v in enumerate(metrics["pearson"])},
                **{f"rms_{i+1}": v for i, v in enumerate(metrics["rms"])},
                **{f"snr_db_{i+1}": v for i, v in enumerate(metrics["snr_db"])},
                **{f"shift_{i+1}": v for i, v in enumerate(metrics["shift"])},
                **{f"scale_{i+1}": v for i, v in enumerate(metrics["scale"])},
                **{f"nans_fraction_{i+1}": v for i, v in enumerate(metrics["nans_fraction"])},
            }
            results.append(row)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    df.to_csv(results_csv, index=False)
    print(f"Saved metrics to {results_csv}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Evaluate digitized ECGs against ground truth.")
    argparser.add_argument(
        "--config",
        type=str,
        default="evaluate.yml",
        help="Config file name or path (searched in . and src/config/). Default: evaluate.yml",
    )

    args = argparser.parse_args()
    config_path = find_config_path(args.config)
    cfg = get_cfg(config_path)
    main(cfg)
