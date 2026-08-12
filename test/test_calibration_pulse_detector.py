import math

import torch

from src.model.calibration_pulse_detector import (
    DEFAULT_MV_PER_MM,
    CalibrationPulseDetector,
)

PX_PER_MM = 10.0  # Chosen so a 100px pulse height maps to exactly 0.1 mV/mm (standard gain), for clean asserts.


def _make_line(
    total_width: int = 600,
    baseline_px: int = 15,
    pulse_width_px: int = 50,
    pulse_height_px: float = 100.0,
    after_baseline_px: int = 15,
    noise_std: float = 0.3,
    seed: int = 0,
) -> torch.Tensor:
    """Builds a synthetic extracted line: flat baseline -> rectangular pulse -> flat baseline -> a smooth
    sine wave standing in for ordinary ECG trace content, with small noise throughout."""
    torch.manual_seed(seed)
    line = torch.zeros(total_width)

    pulse_start = baseline_px
    pulse_end = pulse_start + pulse_width_px
    after_end = pulse_end + after_baseline_px

    line[pulse_start:pulse_end] = pulse_height_px

    t = torch.linspace(0, 8 * math.pi, total_width - after_end)
    line[after_end:] = 20 * torch.sin(t)

    line += torch.randn(total_width) * noise_std
    return line


def _make_flat_lead(total_width: int = 600, noise_std: float = 0.3, seed: int = 0) -> torch.Tensor:
    """A line with no pulse at all -- pure smooth sine content from the very start, as a lead without a
    calibration mark (or a mid-strip lead in a multi-row layout) would look."""
    torch.manual_seed(seed)
    t = torch.linspace(0, 8 * math.pi, total_width)
    return 20 * torch.sin(t) + torch.randn(total_width) * noise_std


def test_detect_pulse_in_line_finds_known_height() -> None:
    detector = CalibrationPulseDetector()
    line = _make_line(pulse_height_px=100.0)

    height = detector.detect_pulse_in_line(line, PX_PER_MM)

    assert height is not None
    # A discretized set of candidate widths/offsets plus synthetic noise means this won't land exactly on
    # the true 100px height -- 10% tolerance is generous for a heuristic detector, tight enough to catch a
    # real regression.
    assert abs(height - 100.0) < 10.0


def test_detect_pulse_in_line_none_for_pure_sine() -> None:
    detector = CalibrationPulseDetector()
    line = _make_flat_lead()

    height = detector.detect_pulse_in_line(line, PX_PER_MM)

    assert height is None


def test_detect_pulse_in_line_respects_min_jump() -> None:
    detector = CalibrationPulseDetector(min_jump_mm=3.0)
    tiny_pulse = _make_line(pulse_height_px=2.0)  # 0.2mm at PX_PER_MM=10, well under the 3mm floor.

    height = detector.detect_pulse_in_line(tiny_pulse, PX_PER_MM)

    assert height is None


def test_call_reaches_consensus_across_leads() -> None:
    detector = CalibrationPulseDetector(min_leads_for_consensus=2)
    lines = torch.stack([_make_line(pulse_height_px=100.0, seed=i) for i in range(4)])

    result = detector(lines, PX_PER_MM)

    assert result.detected
    assert abs(result.mv_per_mm - DEFAULT_MV_PER_MM) < 0.02
    assert result.num_leads_detected >= 2


def test_call_falls_back_to_default_with_no_pulses() -> None:
    detector = CalibrationPulseDetector(min_leads_for_consensus=2)
    lines = torch.stack([_make_flat_lead(seed=i) for i in range(4)])

    result = detector(lines, PX_PER_MM)

    assert not result.detected
    assert result.mv_per_mm == DEFAULT_MV_PER_MM
    assert result.num_leads_detected == 0


def test_call_falls_back_when_too_few_leads_agree() -> None:
    """One lead has a real pulse, the rest have none -- shouldn't be enough for consensus even though
    one detection technically succeeded."""
    detector = CalibrationPulseDetector(min_leads_for_consensus=2)
    lines = torch.stack(
        [_make_line(pulse_height_px=100.0, seed=0)] + [_make_flat_lead(seed=i) for i in range(1, 4)]
    )

    result = detector(lines, PX_PER_MM)

    assert not result.detected
    assert result.mv_per_mm == DEFAULT_MV_PER_MM


def test_call_detects_pulse_at_end_of_line_when_check_both_ends() -> None:
    detector = CalibrationPulseDetector(min_leads_for_consensus=2, check_both_ends=True)

    def _end_pulse_line(seed: int) -> torch.Tensor:
        return _make_line(pulse_height_px=100.0, seed=seed).flip(0)

    lines = torch.stack([_end_pulse_line(i) for i in range(3)])

    result = detector(lines, PX_PER_MM)

    assert result.detected
    assert abs(result.mv_per_mm - DEFAULT_MV_PER_MM) < 0.02


def test_call_ignores_lines_shorter_than_search_width() -> None:
    detector = CalibrationPulseDetector()
    short_line = torch.full((50,), float("nan"))
    short_line[:20] = 0.0  # Far shorter than search_width_mm(15) * PX_PER_MM(10) = 150px.

    result = detector(short_line.unsqueeze(0), PX_PER_MM)

    assert result.num_leads_checked == 0
    assert not result.detected


def test_call_handles_empty_input() -> None:
    detector = CalibrationPulseDetector()
    result = detector(torch.empty(0, 0), PX_PER_MM)

    assert not result.detected
    assert result.mv_per_mm == DEFAULT_MV_PER_MM


def test_outlier_detection_dropped_from_consensus() -> None:
    """A wildly different estimate (e.g. a false-positive flat-topped beat elsewhere) shouldn't drag the
    consensus away from the agreeing majority."""
    detector = CalibrationPulseDetector(min_leads_for_consensus=2, consensus_relative_tolerance=0.2)
    normal_lines = [_make_line(pulse_height_px=100.0, seed=i) for i in range(3)]
    outlier_line = _make_line(pulse_height_px=25.0, seed=99)  # mv_per_mm ~= 0.4, far from ~0.1.

    lines = torch.stack(normal_lines + [outlier_line])
    result = detector(lines, PX_PER_MM)

    assert result.detected
    assert abs(result.mv_per_mm - DEFAULT_MV_PER_MM) < 0.02
