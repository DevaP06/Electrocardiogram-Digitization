"""
Exports ground truth for an end-to-end accuracy benchmark from WFDB records already on local disk
(e.g. PTB-XL), rather than from the HuggingFace development dataset that
export_hf_ground_truth_benchmark.py reads.

Why a separate entry point: the development dataset's validation split is not held out with respect to
the released weights -- the authors trained on that dataset, so numbers measured there describe fit to
development data, not generalization. PTB-XL is a different cohort entirely, so it supports the claim
the other benchmark cannot. Everything downstream (lead ordering, microvolt units, per-<id> directory
layout, src.evaluate scoring) is deliberately identical so the two are directly comparable.

Output layout (matching export_hf_ground_truth_benchmark.py -- see that module for why the per-<id>
image subdirectory is load-bearing rather than cosmetic):
    <ground_truth_out>/<id>/digital_signal_<id>.csv

Usage:
    python -m src.scripts.export_local_wfdb_ground_truth \
        --records_dir data/ptbxl/records500 --max_records 20        # small first pass
    python -m src.scripts.export_local_wfdb_ground_truth \
        --records_dir data/ptbxl/records500 --record_list data/ptbxl/selected.txt
"""

import argparse
import os
from typing import List, Optional

import numpy as np
import wfdb

from src.evaluate import resample_to_length
from src.scripts.export_hf_ground_truth_benchmark import CANONICAL_LEAD_ORDER, _reorder_to_canonical


def find_wfdb_records(records_dir: str, limit: Optional[int] = None) -> List[str]:
    """Every WFDB record under records_dir, as paths without the .hea/.dat extension.

    PTB-XL nests records in numbered subdirectories (records500/00000/00001_hr.hea), so this walks
    recursively rather than listing one level. Sorted for reproducibility -- an unsorted walk would make
    --max_records select a different subset per run and quietly break comparability between runs.
    """
    found: List[str] = []
    for root, _, files in os.walk(records_dir):
        for name in sorted(files):
            if name.endswith(".hea"):
                found.append(os.path.join(root, name[: -len(".hea")]))
    found.sort()
    return found[:limit] if limit else found


def export_record(
    record_path: str,
    ground_truth_out: str,
    target_num_samples: int,
    depicted_seconds: Optional[float],
) -> str:
    """Writes one record's ground-truth CSV. Returns the id used for its directory."""
    record = wfdb.rdrecord(record_path)
    record_id = os.path.basename(record_path)

    signal_uv = _reorder_to_canonical(record.p_signal, record.sig_name, record.units)

    # Crop to the window the rendered image depicts, before resampling -- ground truth covering a
    # different span than the picture makes every sample index disagree progressively, which reads as a
    # growing time lag rather than an obvious offset. PTB-XL records are exactly 10 s, so with the
    # default this is a no-op; it matters if a source record runs longer than its rendering.
    if depicted_seconds is not None:
        n_depicted = int(round(record.fs * depicted_seconds))
        if 0 < n_depicted < signal_uv.shape[0]:
            signal_uv = signal_uv[:n_depicted]

    resampled = resample_to_length(signal_uv.T, target_num_samples).T

    out_dir = os.path.join(ground_truth_out, record_id)
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(
        os.path.join(out_dir, f"digital_signal_{record_id}.csv"),
        resampled,
        delimiter=",",
        header=",".join(CANONICAL_LEAD_ORDER),
        comments="",
    )
    return record_id


def main(
    records_dir: str,
    ground_truth_out: str,
    target_num_samples: int,
    max_records: Optional[int],
    depicted_seconds: Optional[float],
    record_list: Optional[str],
) -> None:
    os.makedirs(ground_truth_out, exist_ok=True)

    if record_list:
        with open(record_list) as f:
            records = [line.strip() for line in f if line.strip()]
    else:
        records = find_wfdb_records(records_dir, max_records)

    if not records:
        raise SystemExit(f"No .hea records found under {records_dir!r}.")

    ok, failed = 0, 0
    for i, record_path in enumerate(records):
        try:
            record_id = export_record(record_path, ground_truth_out, target_num_samples, depicted_seconds)
        except Exception as e:
            print(f"[{i + 1}/{len(records)}] {os.path.basename(record_path)}: FAILED ({e})")
            failed += 1
        else:
            ok += 1
            if (i + 1) % 25 == 0 or (i + 1) == len(records):
                print(f"[{i + 1}/{len(records)}] {record_id}: OK")

    print(f"\nDone. {ok} exported, {failed} failed.")
    print(f"Ground truth: {ground_truth_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ground truth from local WFDB records (e.g. PTB-XL).")
    parser.add_argument("--records_dir", default="data/ptbxl/records500", help="Directory walked for .hea/.dat pairs.")
    parser.add_argument("--ground_truth_out", default="data/ptbxl_benchmark/ground_truth")
    parser.add_argument(
        "--target_num_samples",
        type=int,
        default=10000,
        help="Must match the LAYOUT_IDENTIFIER target_num_samples used when digitizing these images "
        "(src/config/inference_wrapper_george-moody-2024.yml uses 10000).",
    )
    parser.add_argument(
        "--depicted_seconds",
        type=float,
        default=10.0,
        help="Seconds of the record the rendered image shows. PTB-XL records are exactly 10 s, so the "
        "default is a no-op; pass 0 to disable cropping entirely.",
    )
    parser.add_argument("--max_records", type=int, default=None, help="Limit records processed -- small run first.")
    parser.add_argument("--record_list", default=None, help="File of record paths (no extension), one per line.")
    args = parser.parse_args()

    main(
        records_dir=args.records_dir,
        ground_truth_out=args.ground_truth_out,
        target_num_samples=args.target_num_samples,
        max_records=args.max_records,
        depicted_seconds=args.depicted_seconds if args.depicted_seconds > 0 else None,
        record_list=args.record_list,
    )
