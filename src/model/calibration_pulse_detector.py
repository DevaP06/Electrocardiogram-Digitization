"""
Detects the calibration pulse -- the rectangular gain marker most 12-lead ECG printouts draw at the start
(and often also the end) of each trace -- in extracted signal lines, and uses its measured pixel height to
derive the image's actual mV/mm scale instead of assuming the standard-gain default.

Background: by convention, a calibration pulse's electrical amplitude is always exactly 1 mV, regardless of
the amplitude gain the machine was set to when it printed the strip (10 mm/mV is standard, but 5 mm/mV and
20 mm/mV printouts exist and are otherwise indistinguishable from a standard-gain strip after digitization).
The pulse's *physical height in mm* is what actually varies with gain -- a taller pulse means a lower
mV/mm. So measuring the pulse's height directly recovers the true scale for that specific image, instead of
LeadIdentifier.normalize() silently assuming DEFAULT_MV_PER_MM for every image regardless of its real gain.
"""

from typing import List, NamedTuple, Optional

import torch

DEFAULT_MV_PER_MM: float = 0.1  # Standard 10 mm/mV gain -- the pipeline-wide fallback when no pulse is found.


class CalibrationPulseResult(NamedTuple):
    """Outcome of calibration-pulse detection across a set of extracted signal lines.

    Attributes:
        mv_per_mm: The scale to use for this image -- the consensus of per-lead estimates if enough of
            them agreed, otherwise DEFAULT_MV_PER_MM.
        detected: Whether a confident, cross-lead-consistent calibration pulse was found. False means
            mv_per_mm is just the fallback default, not a measurement.
        num_leads_checked: How many lines were long enough to search at all.
        num_leads_detected: How many of those yielded a pulse candidate passing every threshold, from
            either end of the line.
        per_lead_mv_per_mm: The individual estimate from each detection (a line can contribute up to two,
            one per end) -- useful for logging/debugging the spread behind the consensus value.
    """

    mv_per_mm: float
    detected: bool
    num_leads_checked: int
    num_leads_detected: int
    per_lead_mv_per_mm: List[float]


class CalibrationPulseDetector:
    def __init__(
        self,
        search_width_mm: float = 15.0,
        baseline_width_mm: float = 1.5,
        min_pulse_width_mm: float = 2.0,
        max_pulse_width_mm: float = 12.0,
        num_width_candidates: int = 10,
        min_jump_mm: float = 3.0,
        flatness_ratio_threshold: float = 0.25,
        return_tolerance_ratio: float = 0.35,
        min_leads_for_consensus: int = 2,
        consensus_relative_tolerance: float = 0.2,
        check_both_ends: bool = True,
    ) -> None:
        """
        Args:
            search_width_mm: How far from the start of a line's valid range to search for a pulse, in mm.
                Calibration pulses are conventionally printed at the very edge of a lead's trace, so this
                only needs to be a few times the expected pulse width.
            baseline_width_mm: Width of the flat "isoelectric" segment checked immediately before and
                after a candidate plateau, in mm.
            min_pulse_width_mm / max_pulse_width_mm: Range of plateau widths tried, in mm. Real
                calibration pulses are typically ~5mm wide (200ms at 25mm/s, or 100ms at 50mm/s); this
                range is intentionally generous since paper speed isn't measured anywhere else in this
                pipeline.
            num_width_candidates: Number of plateau widths sampled (linearly) from the range above.
            min_jump_mm: Minimum step height, in mm (using the image's own detected mm/pixel scale), for a
                candidate to be considered at all -- rules out amplifying ordinary trace noise into a
                spurious "pulse".
            flatness_ratio_threshold: A candidate's baseline and plateau segments must each have a
                standard deviation below this fraction of the jump height. Real biological complexes
                (P/QRS/T) are curved, not flat-topped, so this is the main thing that distinguishes a
                calibration pulse from a tall QRS spike that happens to sit near the edge of the trace.
            return_tolerance_ratio: After the plateau, the signal must return to within this fraction of
                the jump height of the original baseline -- rules out a monotonic drift or a real ST-like
                elevation being mistaken for a pulse.
            min_leads_for_consensus: Minimum number of independent detections (across lines and ends)
                required before the aggregate estimate is trusted. A single detection could be a false
                positive from an unusually flat-topped beat.
            consensus_relative_tolerance: Individual estimates must agree with the median within this
                relative tolerance to count toward consensus; detections that disagree wildly (e.g. a
                genuine false positive) are dropped rather than allowed to skew the result.
            check_both_ends: If True, also search the tail of each line (some layouts print the
                calibration mark at the end of a strip rather than the start) by running the same
                detector on the reversed line.
        """
        self.search_width_mm = search_width_mm
        self.baseline_width_mm = baseline_width_mm
        self.min_pulse_width_mm = min_pulse_width_mm
        self.max_pulse_width_mm = max_pulse_width_mm
        self.num_width_candidates = num_width_candidates
        self.min_jump_mm = min_jump_mm
        self.flatness_ratio_threshold = flatness_ratio_threshold
        self.return_tolerance_ratio = return_tolerance_ratio
        self.min_leads_for_consensus = min_leads_for_consensus
        self.consensus_relative_tolerance = consensus_relative_tolerance
        self.check_both_ends = check_both_ends

    def _segment_stats(self, line: torch.Tensor, start: int, end: int) -> Optional[tuple[float, float]]:
        """Mean and std of line[start:end], ignoring NaNs. None if the (clamped) range has no valid samples."""
        start = max(start, 0)
        end = min(end, line.shape[0])
        if end <= start:
            return None
        segment = line[start:end]
        valid = segment[~torch.isnan(segment)]
        if valid.numel() == 0:
            return None
        std = float(valid.std().item()) if valid.numel() > 1 else 0.0
        return float(valid.mean().item()), std

    def detect_pulse_in_line(self, line: torch.Tensor, px_per_mm: float) -> Optional[float]:
        """
        Searches near the start of a single extracted line for a rectangular step-plateau-step pattern
        (baseline -> sharp jump -> flat plateau -> sharp jump back to baseline) and returns its height in
        pixels, or None if no candidate passes every threshold.

        Args:
            line: 1D tensor of pixel y-positions, NaN outside the line's valid column range.
            px_per_mm: Pixels per mm for this image, as computed by PixelSizeFinder.

        Returns:
            The best-scoring pulse height in pixels, or None.
        """
        valid_idx = (~torch.isnan(line)).nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return None
        line_start = int(valid_idx[0].item())

        baseline_width_px = max(1, round(self.baseline_width_mm * px_per_mm))
        search_width_px = round(self.search_width_mm * px_per_mm)
        min_jump_px = self.min_jump_mm * px_per_mm

        best_score = 0.0
        best_height: Optional[float] = None

        for width_mm in torch.linspace(self.min_pulse_width_mm, self.max_pulse_width_mm, self.num_width_candidates).tolist():
            width_px = max(1, round(width_mm * px_per_mm))
            plateau_start = line_start + baseline_width_px
            plateau_end = plateau_start + width_px
            if plateau_end + baseline_width_px > line_start + search_width_px:
                continue  # This candidate would run past the search window.

            baseline_stats = self._segment_stats(line, line_start, plateau_start)
            plateau_stats = self._segment_stats(line, plateau_start, plateau_end)
            after_stats = self._segment_stats(line, plateau_end, plateau_end + baseline_width_px)
            if baseline_stats is None or plateau_stats is None or after_stats is None:
                continue

            baseline_mean, baseline_std = baseline_stats
            plateau_mean, plateau_std = plateau_stats
            after_mean, _ = after_stats

            jump = abs(plateau_mean - baseline_mean)
            if jump < min_jump_px:
                continue
            if baseline_std > self.flatness_ratio_threshold * jump:
                continue
            if plateau_std > self.flatness_ratio_threshold * jump:
                continue
            if abs(after_mean - baseline_mean) > self.return_tolerance_ratio * jump:
                continue

            # Prefer tall, flat candidates: a large jump with low noise on both flat segments.
            score = jump / (baseline_std + plateau_std + 1.0)
            if score > best_score:
                best_score = score
                best_height = jump

        return best_height

    def __call__(self, lines: torch.Tensor, avg_pixel_per_mm: float) -> CalibrationPulseResult:
        """
        Args:
            lines: Raw extracted lines (e.g. SignalExtractor's output, before lead-name matching), shape
                (num_lines, width); pixel y-positions with NaN outside each line's valid range.
            avg_pixel_per_mm: Pixels per mm, as computed by PixelSizeFinder.

        Returns:
            CalibrationPulseResult with the mv_per_mm to use for this image.
        """
        if avg_pixel_per_mm <= 0 or lines.numel() == 0:
            return CalibrationPulseResult(DEFAULT_MV_PER_MM, False, 0, 0, [])

        estimates: List[float] = []
        num_checked = 0
        min_valid_samples = round(self.search_width_mm * avg_pixel_per_mm)

        for line in lines:
            valid_count = int((~torch.isnan(line)).sum().item())
            if valid_count < min_valid_samples:
                continue  # Too short to reliably search for a pulse.
            num_checked += 1

            candidates = [line]
            if self.check_both_ends:
                candidates.append(line.flip(0))

            for candidate_line in candidates:
                pulse_height_px = self.detect_pulse_in_line(candidate_line, avg_pixel_per_mm)
                if pulse_height_px is not None and pulse_height_px > 0:
                    estimates.append(avg_pixel_per_mm / pulse_height_px)

        if len(estimates) < self.min_leads_for_consensus:
            return CalibrationPulseResult(DEFAULT_MV_PER_MM, False, num_checked, len(estimates), estimates)

        median = float(torch.tensor(estimates).median().item())
        agreeing = [e for e in estimates if abs(e - median) <= self.consensus_relative_tolerance * median]

        if len(agreeing) < self.min_leads_for_consensus:
            return CalibrationPulseResult(DEFAULT_MV_PER_MM, False, num_checked, len(estimates), estimates)

        consensus_mv_per_mm = float(torch.tensor(agreeing).median().item())
        return CalibrationPulseResult(consensus_mv_per_mm, True, num_checked, len(estimates), estimates)
