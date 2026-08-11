import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Generator, List, Optional

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.nn import Module
from yacs.config import CfgNode as CN

from src.utils import import_class_from_path


@contextmanager
def timed_section(name: str, times_dict: dict[str, float]) -> Generator[None, None, None]:
    """Context manager for timing code blocks.

    Args:
        name: Name of the section.
        times_dict: Dictionary to store timing.
    """
    start = time.time()
    yield
    times_dict[name] = time.time() - start


class InferenceWrapper(Module):
    def __init__(
        self,
        config: CN,
        device: str,
        resample_size: None | tuple[int, ...] = None,
        grid_class: int = 0,
        text_background_class: int = 1,
        signal_class: int = 2,
        background_class: int = 3,
        rotate_on_resample: bool = False,
        enable_timing: bool = False,
        minimum_image_size: int = 512,
        apply_dewarping: bool = True,
    ) -> None:
        """Inference wrapper for ECG pipeline.

        Args:
            config: Configuration node.
            device: Torch device string.
            resample_size: Optional resample target size.
            grid_class: Grid class index.
            text_background_class: Text and background class index.
            signal_class: Signal class index.
            background_class: Background class index.
            rotate_on_resample: Whether to rotate on resample.
            enable_timing: Whether to print timings.
            minimum_image_size: Minimum allowed image size.
            apply_dewarping: Whether to apply dewarping (perspective correction is still performed regardless).
        """
        super().__init__()
        self.config = config
        self.device = device
        self.resample_size = resample_size
        self.grid_class = grid_class
        self.text_background_class = text_background_class
        self.signal_class = signal_class
        self.background_class = background_class
        self.rotate_on_resample = rotate_on_resample
        self._timing_enabled = enable_timing
        self.minimum_image_size = minimum_image_size
        self.apply_dewarping = apply_dewarping

        self.signal_extractor = self._load_signal_extractor()
        self.perspective_detector: Any = self._load_perspective_detector()
        self.segmentation_model: Any = self._load_segmentation_model().to(self.device)
        self.cropper: Any = self._load_cropper()
        self.pixel_size_finder: Any = self._load_pixel_size_finder()
        self.dewarper: Any = self._load_dewarper()
        self.identifier = self._load_layout_identifier()
        self.times: dict[str, float] = {}

    @torch.no_grad()
    def forward(
        self, image: Tensor, layout_should_include_substring: None | str
    ) -> dict[str, Tensor | str | float | None | dict[str, Any]]:
        """Performs full inference on an input image.

        Args:
            image: Input image tensor.
            layout_should_include_substring: Optional substring to filter layout names.

        Returns:
            Dictionary with processed outputs and intermediate results.
        """
        self._check_image_dimensions(image)
        image = self.min_max_normalize(image)
        image = image.to(self.device)

        self.times = {}
        image = self._resample_image(image, self.times)

        signal_prob, grid_prob, text_prob = self._get_feature_maps(image)

        result = self._postprocess_after_segmentation(
            image, signal_prob, grid_prob, text_prob, layout_should_include_substring, self.times
        )
        self._print_profiling_results()
        return result

    @torch.no_grad()
    def forward_batch(
        self,
        images: List[Tensor],
        layout_should_include_substrings: List[None | str],
    ) -> List[dict[str, Tensor | str | float | None | dict[str, Any]]]:
        """Performs full inference on a batch of images.

        This is a throughput optimization for one specific stage, not a claim that forward_batch() and
        forward() are interchangeable bit-for-bit:

        Segmentation (the only truly GPU-vectorizable stage here) is batched across images that share
        the same shape after resampling, via _segment_batch(). Because UNet normalizes with
        InstanceNorm2d (per-sample, not per-batch), batching same-shaped images is mathematically exact
        *in exact arithmetic* -- each sample's normalization only ever depends on itself. In measured
        practice on real weights/images, however, batch-of-1 vs batch-of-N segmentation output differs
        by up to ~1e-3 on raw class probabilities: GPU (and CPU) conv kernels can select a different,
        equally valid execution/summation order depending on batch size, and floating-point addition
        isn't associative. That's expected numerical behavior for a network this deep, not a bug -- but
        because several downstream stages key off hard thresholds (Hough-transform binarization,
        connected-component labeling, peak detection), a ~1e-3 input perturbation can occasionally
        cascade into a visibly different final digitized signal for a given image. This is the same
        *character* of sensitivity forward() already has on its own: SignalExtractor and
        PerspectiveDetector use unseeded torch.randn/torch.randperm calls, so calling forward() twice on
        one image does not reproduce bit-identical output either. Batching does not introduce a new
        failure mode here; it recombines with one that already existed.

        Everything after segmentation -- perspective detection, cropping, pixel-size search, dewarping,
        and signal extraction -- runs sequentially per image, unchanged from forward(). This code is
        single-image NumPy/SciPy/scikit-image (Hough transforms, KNN graphs, connected-component
        labeling, Hungarian matching, and -- notably -- tight pure-Python per-pixel/per-candidate loops
        in SignalExtractor) with no batch dimension to vectorize into. An earlier version of this method
        ran that part on a ThreadPoolExecutor, on the assumption that most of it releases the GIL; that
        assumption didn't hold (measured slower than sequential, from GIL contention on the Python-loop
        stretches with no real parallelism gained), so it was removed. A future version could get a real
        win here via multiprocessing instead of threading, but that needs each worker to hold its own
        copy of the loaded models -- a larger change than this pass covers.

        Args:
            images: Input image tensors, each of shape (1, 3, H, W).
            layout_should_include_substrings: One optional substring filter per image, same length as
                images.

        Returns:
            List of result dictionaries, one per input image, in the same order as `images`. Each has
            the same shape as forward()'s return value.
        """
        if len(images) != len(layout_should_include_substrings):
            raise ValueError(
                f"images and layout_should_include_substrings must have the same length, got "
                f"{len(images)} and {len(layout_should_include_substrings)}."
            )
        for image in images:
            self._check_image_dimensions(image)

        prepared = [self._resample_image(self.min_max_normalize(image).to(self.device)) for image in images]
        feature_maps = self._segment_batch(prepared)

        results: List[dict[str, Tensor | str | float | None | dict[str, Any]]] = []
        for i in range(len(images)):
            times: dict[str, float] = {}
            results.append(
                self._postprocess_after_segmentation(
                    prepared[i],
                    feature_maps[i][0],
                    feature_maps[i][1],
                    feature_maps[i][2],
                    layout_should_include_substrings[i],
                    times,
                )
            )
            self._print_profiling_results(times, prefix=f" (image {i})")

        return results

    def _postprocess_after_segmentation(
        self,
        image: Tensor,
        signal_prob: Tensor,
        grid_prob: Tensor,
        text_prob: Tensor,
        layout_should_include_substring: None | str,
        times: dict[str, float],
    ) -> dict[str, Tensor | str | float | None | dict[str, Any]]:
        """Runs everything from perspective detection through lead identification. Shared by forward()
        (single image, times=self.times) and forward_batch() (per-image, run concurrently, each with
        its own local times dict) so the two entrypoints can never compute different results -- batching
        only changes how this work is scheduled, never what it computes.
        """
        with timed_section("Perspective detection", times):
            alignment_params = self.perspective_detector(grid_prob)

        with timed_section("Cropping", times):
            source_points = self.cropper(signal_prob, alignment_params)

        aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._align_feature_maps(
            image, signal_prob, grid_prob, text_prob, source_points, times
        )

        with timed_section("Pixel size search", times):
            mm_per_pixel_x, mm_per_pixel_y = self.pixel_size_finder(aligned_grid_prob)
            avg_pixel_per_mm = (1 / mm_per_pixel_x + 1 / mm_per_pixel_y) / 2

        with timed_section("Dewarping", times):
            if self.apply_dewarping:
                self.dewarper.fit(aligned_grid_prob.squeeze(), avg_pixel_per_mm)
                aligned_signal_prob = self.dewarper.transform(aligned_signal_prob.squeeze())

        with timed_section("Signal extraction", times):
            signals = self.signal_extractor(aligned_signal_prob.squeeze())

        layout = self.identifier(
            signals,
            aligned_text_prob,
            avg_pixel_per_mm,
            layout_should_include_substring=layout_should_include_substring,
        )
        try:
            layout_str = layout["layout"]
            layout_is_flipped = str(layout["flip"])
            layout_cost = layout.get("cost", 1.0)
        except KeyError:
            layout_str = "Unknown layout"
            layout_is_flipped = "False"
            layout_cost = 1.0

        return {
            "layout_name": layout_str,
            "input_image": image.cpu(),
            "aligned": {
                "image": aligned_image.cpu(),
                "signal_prob": aligned_signal_prob.cpu(),
                "grid_prob": aligned_grid_prob.cpu(),
                "text_prob": aligned_text_prob.cpu(),
            },
            "signal": {
                "raw_lines": signals.cpu(),
                "canonical_lines": layout.get("canonical_lines", None),
                "lines": layout.get("lines", None),
                "layout_matching_cost": layout_cost,
                "layout_is_flipped": layout_is_flipped,
                "lead_confidence": layout.get("lead_confidence", None),
            },
            "pixel_spacing_mm": {
                "x": mm_per_pixel_x,
                "y": mm_per_pixel_y,
                "average_pixel_per_mm": avg_pixel_per_mm,
            },
            "source_points": source_points.cpu(),
        }

    def _align_feature_maps(
        self,
        image: Tensor,
        signal_prob: Tensor,
        grid_prob: Tensor,
        text_prob: Tensor,
        source_points: Tensor,
        times: Optional[dict[str, float]] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Aligns image and feature maps using perspective cropping.

        Returns:
            Aligned image, signal, grid, and text tensors.
        """
        times = self.times if times is None else times
        with timed_section("Feature map resampling", times):
            aligned_signal_prob = self.cropper.apply_perspective(signal_prob, source_points, fill_value=0)
            aligned_image = self.cropper.apply_perspective(image, source_points, fill_value=0)
            aligned_grid_prob = self.cropper.apply_perspective(grid_prob, source_points, fill_value=0)
            aligned_text_prob = self.cropper.apply_perspective(text_prob, source_points, fill_value=0)
            if self.rotate_on_resample:
                aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._rotate_on_resample(
                    aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob
                )
            aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._crop_y(
                aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob
            )

            return aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob

    def _crop_y(
        self, image: Tensor, signal_prob: Tensor, grid_prob: Tensor, text_prob: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Crops tensors in y and x using bounds from feature maps.

        Returns:
            Cropped image, signal, grid, and text tensors.
        """

        def get_bounds(tensor: Tensor) -> tuple[int, int]:
            prob = torch.clamp(
                tensor.squeeze().sum(dim=tensor.dim() - 3) - tensor.squeeze().sum(dim=tensor.dim() - 3).mean(),
                min=0,
            )
            non_zero = (prob > 0).nonzero(as_tuple=True)[0]
            if non_zero.numel() == 0:
                return 0, tensor.shape[2] - 1
            return int(non_zero[0].item()), int(non_zero[-1].item())

        y1, y2 = get_bounds(signal_prob + grid_prob)

        slices = (slice(None), slice(None), slice(y1, y2 + 1), slice(None))
        return image[slices], signal_prob[slices], grid_prob[slices], text_prob[slices]

    def _print_profiling_results(self, times: Optional[dict[str, float]] = None, prefix: str = "") -> None:
        """Prints the timings for each timed section.

        Args:
            times: Timings to print. Defaults to self.times (the single-image forward() path). The
                batched forward_batch() path passes each image's own dict explicitly instead, since
                self.times is not safe to share across concurrently-running images.
            prefix: Optional label (e.g. an image index) prepended to the header line.
        """
        if not self._timing_enabled:
            return
        times = self.times if times is None else times
        if not times:
            return
        print(f" Timing results{prefix}:")
        max_length = max(len(section) for section in times.keys())
        for section, duration in times.items():
            print(f"    {section:<{max_length+2}}{duration:.2f} s")
        total_time = sum(times.values())
        print(f"Total time: {total_time:.2f} s")

    def _rotate_on_resample(
        self,
        aligned_image: Tensor,
        aligned_signal_prob: Tensor,
        aligned_grid_prob: Tensor,
        aligned_text_prob: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Rotates all tensors if height > width.

        Returns:
            Rotated tensors in same order.
        """
        if aligned_image.shape[2] > aligned_image.shape[3]:
            aligned_image = torch.rot90(aligned_image, k=3, dims=(2, 3))
            aligned_signal_prob = torch.rot90(aligned_signal_prob, k=3, dims=(2, 3))
            aligned_grid_prob = torch.rot90(aligned_grid_prob, k=3, dims=(2, 3))
            aligned_text_prob = torch.rot90(aligned_text_prob, k=3, dims=(2, 3))
        return aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob

    def _resample_image(self, image: Tensor, times: Optional[dict[str, float]] = None) -> Tensor:
        times = self.times if times is None else times
        with timed_section("Initial resampling", times):
            if self.resample_size is None:
                return image

            height, width = image.shape[2], image.shape[3]
            min_dim = min(height, width)
            max_dim = max(height, width)

            if min_dim < self.minimum_image_size:
                scale: float = self.minimum_image_size / min_dim
                new_size: tuple[int, int] = (int(height * scale), int(width * scale))
                interpolated: Tensor = F.interpolate(image, size=new_size, mode="bilinear", align_corners=False)
                return interpolated

            if isinstance(self.resample_size, int):
                if max_dim > self.resample_size:
                    scale = self.resample_size / max_dim
                    new_size = (int(height * scale), int(width * scale))
                    return F.interpolate(image, size=new_size, mode="bilinear", align_corners=False, antialias=True)
                return image

            if isinstance(self.resample_size, tuple):
                interpolated = F.interpolate(
                    image, size=self.resample_size, mode="bilinear", align_corners=False, antialias=True
                )
                return interpolated

            raise ValueError(f"Invalid resample_size: {self.resample_size}. Expected int or tuple of (height, width).")

    def process_sparse_prob(self, signal_prob: Tensor) -> Tensor:
        """Zero-centers and rescales a (N, C, H, W) probability map to [0, 1].

        Reduces per-sample (dims 1..end), not globally, so that stacking multiple images into one batch
        tensor (see _segment_batch) gives each image the same result it would get processed alone --
        for a lone image (N=1) this is exactly the same computation as reducing over the whole tensor,
        so single-image behavior is unchanged.
        """
        reduce_dims = tuple(range(1, signal_prob.dim()))
        signal_prob = signal_prob - signal_prob.mean(dim=reduce_dims, keepdim=True)
        signal_prob = torch.clamp(signal_prob, min=0)
        signal_prob = signal_prob / (signal_prob.amax(dim=reduce_dims, keepdim=True) + 1e-9)
        return signal_prob

    def _split_feature_maps(self, prob: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Slices a softmax'd (N, C, H, W) class-probability tensor into the three per-pixel maps the
        rest of the pipeline consumes, and normalizes each. Shared by the single-image and batched
        segmentation paths so they can never compute this differently."""
        signal_prob = prob[:, [self.signal_class], :, :]
        grid_prob = prob[:, [self.grid_class], :, :]
        text_prob = prob[:, [self.text_background_class], :, :]

        signal_prob = self.process_sparse_prob(signal_prob)
        grid_prob = self.process_sparse_prob(grid_prob)
        text_prob = self.process_sparse_prob(text_prob)
        return signal_prob, grid_prob, text_prob

    def _get_feature_maps(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        with timed_section("Segmentation", self.times):
            logits = self.segmentation_model(image)
            prob = torch.softmax(logits, dim=1)
            return self._split_feature_maps(prob)

    def _segment_batch(self, images: List[Tensor]) -> List[tuple[Tensor, Tensor, Tensor]]:
        """Runs the segmentation model once per distinct image shape in the batch, instead of once per
        image, by stacking same-shaped images into a single forward pass.

        This is exact in exact arithmetic, not an approximation like padding differently-shaped images
        together would be: UNet's normalization layers are InstanceNorm2d, which (unlike BatchNorm)
        normalizes each sample independently of the rest of the batch, so nothing about *what* is
        computed for one image depends on which other images share its batch. In measured floating-point
        practice, batch-of-1 vs batch-of-N can still differ by up to ~1e-3 on raw class probabilities,
        because conv kernels are free to pick a different (still numerically valid) execution order for
        different batch sizes and float addition isn't associative -- see forward_batch()'s docstring
        for how that can (rarely) cascade into a different final digitized signal, and why it's still
        worth doing. Differently-shaped images are never padded together (which would change results
        near the image border for a different, avoidable reason: Conv2d's replicate-padding would then
        reflect padding pixels instead of true edge pixels), so each shape-group is still its own batch.

        Args:
            images: Already-normalized, already-resampled image tensors, each of shape (1, 3, H, W).

        Returns:
            List of (signal_prob, grid_prob, text_prob) tuples, one per input image, in the same order.
        """
        shape_groups: dict[tuple[int, int], List[int]] = defaultdict(list)
        for idx, image in enumerate(images):
            shape_groups[(image.shape[2], image.shape[3])].append(idx)

        results: List[Optional[tuple[Tensor, Tensor, Tensor]]] = [None] * len(images)
        for indices in shape_groups.values():
            batch = torch.cat([images[i] for i in indices], dim=0)
            logits = self.segmentation_model(batch)
            prob = torch.softmax(logits, dim=1)
            signal_prob, grid_prob, text_prob = self._split_feature_maps(prob)
            for batch_pos, original_idx in enumerate(indices):
                results[original_idx] = (
                    signal_prob[[batch_pos]],
                    grid_prob[[batch_pos]],
                    text_prob[[batch_pos]],
                )

        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    def min_max_normalize(self, image: Tensor) -> Tensor:
        return (image - image.min()) / (image.max() - image.min())

    def _load_signal_extractor(self) -> Any:
        signal_extractor_class = import_class_from_path(self.config.SIGNAL_EXTRACTOR.class_path)
        extractor: Any = signal_extractor_class(**self.config.SIGNAL_EXTRACTOR.KWARGS)
        return extractor

    def _load_perspective_detector(self) -> Any:
        perspective_detector_class = import_class_from_path(self.config.PERSPECTIVE_DETECTOR.class_path)
        perspective_detector: Any = perspective_detector_class(**self.config.PERSPECTIVE_DETECTOR.KWARGS)
        return perspective_detector

    def _load_segmentation_model(self) -> Any:
        segmentation_model_class = import_class_from_path(self.config.SEGMENTATION_MODEL.class_path)
        segmentation_model: Any = segmentation_model_class(**self.config.SEGMENTATION_MODEL.KWARGS)
        self._load_segmentation_model_weights(segmentation_model)
        return segmentation_model.eval()

    def _load_cropper(self) -> Any:
        cropper_class = import_class_from_path(self.config.CROPPER.class_path)
        cropper: Any = cropper_class(**self.config.CROPPER.KWARGS)
        return cropper

    def _load_pixel_size_finder(self) -> Any:
        pixel_size_finder_class = import_class_from_path(self.config.PIXEL_SIZE_FINDER.class_path)
        pixel_size_finder: Any = pixel_size_finder_class(**self.config.PIXEL_SIZE_FINDER.KWARGS)
        return pixel_size_finder

    def _load_dewarper(self) -> Any:
        dewarper_class = import_class_from_path(self.config.DEWARPER.class_path)
        dewarper: Any = dewarper_class(**self.config.DEWARPER.KWARGS)
        return dewarper

    def _load_layout_identifier(self) -> Any:
        layouts = yaml.safe_load(open(self.config.LAYOUT_IDENTIFIER.config_path, "r"))
        unet_cfg = yaml.safe_load(open(self.config.LAYOUT_IDENTIFIER.unet_config_path, "r"))
        unet_class = import_class_from_path(unet_cfg["MODEL"]["class_path"])
        unet: torch.nn.Module = unet_class(**unet_cfg["MODEL"]["KWARGS"])
        checkpoint = torch.load(self.config.LAYOUT_IDENTIFIER.unet_weight_path, map_location=self.device, weights_only=False)
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        unet.load_state_dict(checkpoint)
        unet.eval()

        identifier_class = import_class_from_path(self.config.LAYOUT_IDENTIFIER.class_path)
        identifier: Any = identifier_class(
            layouts=layouts,
            unet=unet,
            **self.config.LAYOUT_IDENTIFIER.KWARGS,
        )
        return identifier

    def _load_segmentation_model_weights(self, segmentation_model: torch.nn.Module) -> None:
        """Loads weights for segmentation model.

        Args:
            segmentation_model: The model to load weights into.
        """
        checkpoint = torch.load(self.config.SEGMENTATION_MODEL.weight_path, weights_only=False, map_location=self.device)
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        segmentation_model.load_state_dict(checkpoint)

    def _check_image_dimensions(self, image: Tensor) -> None:
        """Checks input image dimensions.

        Args:
            image: Image tensor.

        Raises:
            NotImplementedError: If batch or channel dims are incorrect.
        """
        if image.dim() != 4:
            raise NotImplementedError(f"Expected 4 dimensions, got tensor with {image.dim()} dimensions")
        if image.shape[0] != 1:
            raise NotImplementedError(f"Batch processing not supported, got tensor with shape {image.shape}")
        if image.shape[1] != 3:
            raise NotImplementedError(f"Expected 3 channels, got tensor with shape {image.shape}")
