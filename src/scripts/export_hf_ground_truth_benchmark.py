"""
Exports a real end-to-end accuracy benchmark set from the raw Ahus-AIM/Open-ECG-Digitizer-Development-Dataset
(the same dataset download_hf_dataset.py uses) -- but where that script only extracts the `img`/`mask` fields
for segmentation training, this script also extracts `dat`/`hea` (a standard WFDB signal record: binary
samples + text header) and converts it into the ground-truth format src.evaluate expects.

Each dataset row contributes up to two independent examples: the base ("<id>") image+signal pair, and a
second, independent one stored under the "_T0" suffixed fields (img_T0/dat_T0/hea_T0) -- so a 217-row val
split can yield up to 434 benchmark examples.

Output layout:
    <images_out>/<id>.png                            -- feed this to `python -m src.digitize`
    <ground_truth_out>/<id>/digital_signal_<id>.csv   -- point `evaluate.yml`'s ground_truth_dir at this

Usage:
    python -m src.scripts.export_hf_ground_truth_benchmark --max_examples 20   # quick first pass
    python -m src.scripts.export_hf_ground_truth_benchmark                     # full val split
"""

import argparse
import os
import tempfile
from typing import Any, Optional

import numpy as np
import wfdb
from datasets import load_dataset

from src.evaluate import resample_to_length

CANONICAL_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# WFDB signal units, lowercased, -> multiplier to convert to microvolts. digitize.py's own output is always
# in microvolts (see save_timeseries_csv in src/digitize.py) -- evaluate.py's compute_metrics does NOT unit-
# convert or resample on its own, so a mismatch here would silently wreck RMS/SNR while leaving Pearson
# looking fine (it's scale/shift invariant), which is exactly what would make the bug easy to miss.
UNIT_TO_MICROVOLT = {"uv": 1.0, "µv": 1.0, "mv": 1000.0, "v": 1_000_000.0}


def _unit_scale(unit: str) -> float:
    key = unit.strip().lower()
    if key not in UNIT_TO_MICROVOLT:
        raise ValueError(f"Unrecognized WFDB signal unit {unit!r}; add it to UNIT_TO_MICROVOLT.")
    return UNIT_TO_MICROVOLT[key]


def _wfdb_record_name_from_header(hea_text: str) -> str:
    """The .hea file's first line starts with the record name the file's own internal references (e.g. the
    .dat filename each signal line points to) are built around -- which may or may not match this dataset
    row's `id`. Extracting it directly, rather than assuming id == record name, is what makes writing the
    .hea/.dat pair back out under matching filenames reliable regardless of the source dataset's naming
    convention."""
    first_line = hea_text.strip().splitlines()[0]
    return first_line.split()[0]


def _reorder_to_canonical(signal: np.ndarray, sig_name: list, units: list) -> np.ndarray:
    """(n_samples, n_channels) WFDB signal, in whatever lead order/units the record has, -> (n_samples, 12)
    in CANONICAL_LEAD_ORDER (matching LeadIdentifier.LEAD_CHANNEL_ORDER / digitize.py's output column order
    -- evaluate.py's load_signal_csv matches gt/pred purely by column position, not by header name, so this
    order must match exactly). Leads absent from the record are left as NaN."""
    name_to_idx = {name.strip().upper(): i for i, name in enumerate(sig_name)}
    n_samples = signal.shape[0]
    out = np.full((n_samples, len(CANONICAL_LEAD_ORDER)), np.nan, dtype=np.float64)
    for col, lead in enumerate(CANONICAL_LEAD_ORDER):
        idx = name_to_idx.get(lead.upper())
        if idx is None:
            continue
        out[:, col] = signal[:, idx] * _unit_scale(units[idx])
    return out


def _read_wfdb_record(dat_bytes: bytes, hea_text: str, tmpdir: str) -> "wfdb.Record":
    record_name = _wfdb_record_name_from_header(hea_text)
    with open(os.path.join(tmpdir, f"{record_name}.hea"), "w", newline="\n") as f:
        f.write(hea_text)
    with open(os.path.join(tmpdir, f"{record_name}.dat"), "wb") as f:
        f.write(dat_bytes)
    return wfdb.rdrecord(os.path.join(tmpdir, record_name))


def export_example(
    example_id: str,
    img: Any,
    dat_bytes: bytes,
    hea_text: str,
    images_out: str,
    ground_truth_out: str,
    target_num_samples: int,
    tmpdir: str,
) -> None:
    img.convert("RGB").save(os.path.join(images_out, f"{example_id}.png"))

    record = _read_wfdb_record(dat_bytes, hea_text, tmpdir)
    signal_uv = _reorder_to_canonical(record.p_signal, record.sig_name, record.units)  # (n_samples, 12), uV

    resampled = resample_to_length(signal_uv.T, target_num_samples).T  # resample_to_length wants (leads, samples)

    out_dir = os.path.join(ground_truth_out, example_id)
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(
        os.path.join(out_dir, f"digital_signal_{example_id}.csv"),
        resampled,
        delimiter=",",
        header=",".join(CANONICAL_LEAD_ORDER),
        comments="",
    )


def main(
    images_out: str,
    ground_truth_out: str,
    target_num_samples: int,
    max_examples: Optional[int],
    split: str,
    include_t0: bool,
) -> None:
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(ground_truth_out, exist_ok=True)

    dataset = load_dataset(
        "Ahus-AIM/Open-ECG-Digitizer-Development-Dataset",
        data_files={split: f"data/{split}-*.parquet"},
    )[split]

    n = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    ok, failed = 0, 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n):
            example = dataset[i]
            base_id = example["id"]

            variants = [(base_id, "img", "dat", "hea")]
            if include_t0:
                variants.append((f"{base_id}_T0", "img_T0", "dat_T0", "hea_T0"))

            for example_id, img_key, dat_key, hea_key in variants:
                if example.get(img_key) is None or example.get(dat_key) is None or example.get(hea_key) is None:
                    print(f"[{i + 1}/{n}] {example_id}: SKIPPED (missing {img_key}/{dat_key}/{hea_key})")
                    continue
                try:
                    export_example(
                        example_id,
                        example[img_key],
                        example[dat_key],
                        example[hea_key],
                        images_out,
                        ground_truth_out,
                        target_num_samples,
                        tmpdir,
                    )
                except Exception as e:
                    print(f"[{i + 1}/{n}] {example_id}: FAILED ({e})")
                    failed += 1
                else:
                    print(f"[{i + 1}/{n}] {example_id}: OK")
                    ok += 1

    print(f"\nDone. {ok} exported, {failed} failed.")
    print(f"Images:       {images_out}")
    print(f"Ground truth: {ground_truth_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export img+WFDB ground truth from the Ahus-AIM HF dataset into src.digitize/src.evaluate's expected layouts."
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--images_out", default="data/ahus_benchmark/images")
    parser.add_argument("--ground_truth_out", default="data/ahus_benchmark/ground_truth")
    parser.add_argument(
        "--target_num_samples",
        type=int,
        default=5000,
        help="Must match the target_num_samples the LAYOUT_IDENTIFIER config uses when you run src.digitize "
        "on these images (LeadIdentifier's default is 5000 -- see src/config/inference_wrapper_ahus_testset.yml).",
    )
    parser.add_argument("--max_examples", type=int, default=None, help="Limit rows processed -- do a small run first.")
    parser.add_argument("--no_t0", action="store_true", help="Skip the _T0 variant, export only the base pair per row.")
    args = parser.parse_args()

    main(
        images_out=args.images_out,
        ground_truth_out=args.ground_truth_out,
        target_num_samples=args.target_num_samples,
        max_examples=args.max_examples,
        split=args.split,
        include_t0=not args.no_t0,
    )
