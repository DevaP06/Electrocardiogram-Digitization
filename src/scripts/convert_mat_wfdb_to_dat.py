"""
Converts PhysioNet-Challenge-style WFDB records (.hea + .mat) into standard WFDB (.hea + .dat).

Needed because ECG-Image-Kit's image generator reads the two container formats down different code
paths, and only the .dat one applies the ADC gain:

    .dat -> wfdb.rdrecord(...).p_signal   -> physical units (mV)          correct
    .mat -> scipy.io.loadmat(...)['val']  -> raw ADC integers, no gain    1000x too large

For a record with gain 1000 ADC/mV that makes every amplitude 1000x too big, which both crashes the
generator's reference-WFDB writer (values outside int16) and would silently mis-scale the plotted trace.
Converting the container up front keeps ECG-Image-Kit unmodified -- worth doing rather than patching it,
since "we used the official generator as published" is a claim worth being able to make.

The conversion is lossless in physical units: samples are read as p_signal (mV) and written back with
the same gain/baseline/units, so round-tripping changes the container, not the signal.

Usage:
    python -m src.scripts.convert_mat_wfdb_to_dat --input_dir data/cpsc_2018/g1 --output_dir data/cpsc_dat
"""

import argparse
import os
from typing import Optional

import numpy as np
import wfdb


def convert_record(header_path: str, output_dir: str, max_seconds: Optional[float] = None) -> str:
    """Reads one .hea/.mat record and writes a .hea/.dat pair. Returns the record name."""
    record_base = header_path[: -len(".hea")]
    record_name = os.path.basename(record_base)
    record = wfdb.rdrecord(record_base)

    signal = record.p_signal  # (n_samples, n_channels) in the header's declared units
    if max_seconds is not None:
        n_keep = int(round(record.fs * max_seconds))
        if 0 < n_keep < signal.shape[0]:
            signal = signal[:n_keep]

    # fmt from the source header can carry container-specific decoration (e.g. '16x1+24', where +24
    # skips the .mat file header). That is meaningless for a plain .dat, so write plain format 16.
    wfdb.wrsamp(
        record_name=record_name,
        fs=record.fs,
        units=record.units,
        sig_name=record.sig_name,
        p_signal=np.asarray(signal, dtype=np.float64),
        fmt=["16"] * signal.shape[1],
        adc_gain=record.adc_gain,
        baseline=record.baseline,
        comments=record.comments,
        write_dir=output_dir,
    )
    return record_name


def main(input_dir: str, output_dir: str, max_seconds: Optional[float], limit: Optional[int]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    headers = sorted(
        os.path.join(root, name) for root, _, files in os.walk(input_dir) for name in files if name.endswith(".hea")
    )
    if limit:
        headers = headers[:limit]
    if not headers:
        raise SystemExit(f"No .hea files found under {input_dir!r}.")

    ok, failed = 0, 0
    for i, header_path in enumerate(headers):
        try:
            name = convert_record(header_path, output_dir, max_seconds)
        except Exception as e:
            print(f"[{i + 1}/{len(headers)}] {os.path.basename(header_path)}: FAILED ({e})")
            failed += 1
        else:
            ok += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(headers):
                print(f"[{i + 1}/{len(headers)}] {name}: OK")

    print(f"\nDone. {ok} converted, {failed} failed -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .hea/.mat WFDB records to .hea/.dat.")
    parser.add_argument("--input_dir", default="data/cpsc_2018/g1")
    parser.add_argument("--output_dir", default="data/cpsc_dat")
    parser.add_argument(
        "--max_seconds",
        type=float,
        default=10.0,
        help="Truncate each record to this many seconds. CPSC records vary from 10 s to 60 s while a "
        "rendered page shows a fixed window, so truncating here keeps the generated image, the written "
        "record and the exported ground truth all covering the same span. Pass 0 to keep full length.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_seconds=args.max_seconds if args.max_seconds > 0 else None,
        limit=args.limit,
    )
