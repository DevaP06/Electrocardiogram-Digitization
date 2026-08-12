from typing import Any

import torch

from src.model.calibration_pulse_detector import DEFAULT_MV_PER_MM, CalibrationPulseResult
from src.model.inference_wrapper import InferenceWrapper


class _IdentityCropper:
    """Stub standing in for Cropper: perspective-warping is irrelevant to this test, so just pass tensors
    through unchanged."""

    def __call__(self, signal_prob: torch.Tensor, alignment_params: Any) -> torch.Tensor:
        return torch.zeros(4, 2)

    def apply_perspective(self, tensor: torch.Tensor, source_points: torch.Tensor, fill_value: float = 0) -> torch.Tensor:
        return tensor


class _StubIdentifier:
    """Records the mv_per_mm it was called with, so the test can assert it came from the calibration
    detector rather than always being DEFAULT_MV_PER_MM."""

    def __init__(self) -> None:
        self.received_mv_per_mm: float | None = None

    def __call__(
        self,
        signals: torch.Tensor,
        text_prob: torch.Tensor,
        avg_pixel_per_mm: float,
        mv_per_mm: float = DEFAULT_MV_PER_MM,
        layout_should_include_substring: str | None = None,
    ) -> dict[str, Any]:
        self.received_mv_per_mm = mv_per_mm
        return {"layout": "test_layout", "flip": False, "cost": 0.0, "canonical_lines": signals, "lines": signals}


def _make_wrapper(calibration_detected: bool, detected_mv_per_mm: float) -> tuple[InferenceWrapper, _StubIdentifier]:
    """Builds an InferenceWrapper with every submodule stubbed except the calibration wiring itself, so
    _postprocess_after_segmentation can run without real weights/config, per the same bypass-__new__
    pattern test_inference_wrapper_batching.py uses for _segment_batch."""
    wrapper = InferenceWrapper.__new__(InferenceWrapper)
    torch.nn.Module.__init__(wrapper)

    wrapper.times = {}
    wrapper._timing_enabled = False
    wrapper.rotate_on_resample = False
    wrapper.apply_dewarping = False
    wrapper.apply_calibration_detection = True

    wrapper.perspective_detector = lambda grid_prob: None
    wrapper.cropper = _IdentityCropper()
    wrapper.pixel_size_finder = lambda grid_prob: (0.1, 0.1)  # mm/pixel in x, y -> avg_pixel_per_mm = 10.0
    wrapper.signal_extractor = lambda signal_prob: torch.rand(3, 40)
    wrapper.calibration_detector = lambda signals, avg_pixel_per_mm: CalibrationPulseResult(
        mv_per_mm=detected_mv_per_mm,
        detected=calibration_detected,
        num_leads_checked=3,
        num_leads_detected=3 if calibration_detected else 0,
        per_lead_mv_per_mm=[detected_mv_per_mm] * 3 if calibration_detected else [],
    )
    identifier = _StubIdentifier()
    wrapper.identifier = identifier

    return wrapper, identifier


def _dummy_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image = torch.rand(1, 3, 20, 20)
    signal_prob = torch.rand(1, 1, 20, 20)
    grid_prob = torch.rand(1, 1, 20, 20)
    text_prob = torch.rand(1, 1, 20, 20)
    return image, signal_prob, grid_prob, text_prob


def test_detected_calibration_scale_is_forwarded_to_identifier() -> None:
    wrapper, identifier = _make_wrapper(calibration_detected=True, detected_mv_per_mm=0.05)
    image, signal_prob, grid_prob, text_prob = _dummy_inputs()

    result = wrapper._postprocess_after_segmentation(image, signal_prob, grid_prob, text_prob, None, wrapper.times)

    assert identifier.received_mv_per_mm == 0.05
    assert result["calibration"]["mv_per_mm"] == 0.05
    assert result["calibration"]["detected"] is True
    assert result["calibration"]["num_leads_detected"] == 3


def test_undetected_calibration_falls_back_to_default() -> None:
    wrapper, identifier = _make_wrapper(calibration_detected=False, detected_mv_per_mm=DEFAULT_MV_PER_MM)
    image, signal_prob, grid_prob, text_prob = _dummy_inputs()

    result = wrapper._postprocess_after_segmentation(image, signal_prob, grid_prob, text_prob, None, wrapper.times)

    assert identifier.received_mv_per_mm == DEFAULT_MV_PER_MM
    assert result["calibration"]["detected"] is False


def test_disabling_calibration_detection_skips_it_and_uses_default() -> None:
    wrapper, identifier = _make_wrapper(calibration_detected=True, detected_mv_per_mm=0.05)
    wrapper.apply_calibration_detection = False
    image, signal_prob, grid_prob, text_prob = _dummy_inputs()

    result = wrapper._postprocess_after_segmentation(image, signal_prob, grid_prob, text_prob, None, wrapper.times)

    # calibration_detector must not even be consulted -- mv_per_mm stays the untouched default.
    assert identifier.received_mv_per_mm == DEFAULT_MV_PER_MM
    assert result["calibration"]["detected"] is False
    assert result["calibration"]["mv_per_mm"] == DEFAULT_MV_PER_MM
