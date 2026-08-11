import torch

from src.model.inference_wrapper import InferenceWrapper
from src.model.unet import UNet


def _make_wrapper_for_segmentation_test() -> InferenceWrapper:
    """Builds an InferenceWrapper with just enough state for _segment_batch/_split_feature_maps to run,
    bypassing __init__ (which requires a full pipeline config and real weight files on disk for the
    dewarper/cropper/lead-identifier sub-models that _segment_batch doesn't touch at all)."""
    wrapper = InferenceWrapper.__new__(InferenceWrapper)
    torch.nn.Module.__init__(wrapper)  # Set up _modules/_parameters/_buffers without the full config-driven __init__.
    torch.manual_seed(0)
    wrapper.segmentation_model = UNet(num_in_channels=3, num_out_channels=4, dims=[2, 4], depth=1).eval()
    wrapper.signal_class = 2
    wrapper.grid_class = 0
    wrapper.text_background_class = 1
    wrapper.times = {}
    return wrapper


def test_segment_batch_matches_per_image_segmentation() -> None:
    """The core correctness claim: batching same-shaped images through the segmentation model must give
    the same results (within float32 tolerance) as running each image through it alone. This uses a
    tiny, shallow dummy UNet specifically so float32 rounding stays within atol=1e-6 -- the real, much
    deeper production UNet can show larger (~1e-3) batch-size-dependent floating-point drift from conv
    kernels choosing a different execution order per batch size; see forward_batch()'s docstring. That's
    expected numerical behavior, not something a unit test at this scale is meant to catch -- this test
    instead verifies the batching *mechanism* itself (grouping, splitting, no cross-contamination) is
    correct."""
    wrapper = _make_wrapper_for_segmentation_test()
    torch.manual_seed(1)
    images = [torch.rand(1, 3, 32, 40) for _ in range(3)]

    with torch.no_grad():
        batched_results = wrapper._segment_batch(images)
        individual_results = [wrapper._get_feature_maps(img) for img in images]

    for (batched_signal, batched_grid, batched_text), (single_signal, single_grid, single_text) in zip(
        batched_results, individual_results
    ):
        assert torch.allclose(batched_signal, single_signal, atol=1e-6)
        assert torch.allclose(batched_grid, single_grid, atol=1e-6)
        assert torch.allclose(batched_text, single_text, atol=1e-6)


def test_segment_batch_handles_mixed_shapes_without_cross_contamination() -> None:
    """Images of different shapes must not be padded/stacked together (that would change results near
    the border); each shape-group's output should still exactly match single-image inference."""
    wrapper = _make_wrapper_for_segmentation_test()
    torch.manual_seed(2)
    images = [
        torch.rand(1, 3, 32, 40),
        torch.rand(1, 3, 24, 24),
        torch.rand(1, 3, 32, 40),  # Same shape as the first, different content.
    ]

    with torch.no_grad():
        batched_results = wrapper._segment_batch(images)
        individual_results = [wrapper._get_feature_maps(img) for img in images]

    for batched, single in zip(batched_results, individual_results):
        for batched_map, single_map in zip(batched, single):
            assert batched_map.shape == single_map.shape
            assert torch.allclose(batched_map, single_map, atol=1e-6)

    # The two same-shaped-but-different-content images must not have been confused with each other.
    assert not torch.allclose(batched_results[0][0], batched_results[2][0])


def test_segment_batch_preserves_input_order() -> None:
    wrapper = _make_wrapper_for_segmentation_test()
    torch.manual_seed(3)
    images = [torch.rand(1, 3, 20, 20) for _ in range(4)]

    with torch.no_grad():
        results = wrapper._segment_batch(images)

        assert len(results) == len(images)
        for i, image in enumerate(images):
            expected = wrapper._get_feature_maps(image)
            for got_map, expected_map in zip(results[i], expected):
                assert torch.allclose(got_map, expected_map, atol=1e-6)
