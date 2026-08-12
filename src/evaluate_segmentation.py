"""
Segmentation-only benchmark for a trained U-Net checkpoint.

Scores predicted (grid / text_background / signal / background) masks against ground-truth (image.png,
mask.png) pairs -- the same format ECGScanDataset trains on -- independent of every downstream heuristic
stage (perspective correction, dewarping, signal extraction, lead identification). This is the right tool
when what you have is segmentation training/val data rather than the full pipeline's digitized-signal ground
truth (which src/evaluate.py needs instead): it isolates whether a change to the model or its loss (e.g. the
clDice term in src/loss/loss.py) actually improves segmentation quality, without every downstream stage's
heuristics acting as a confound.

Usage:
    python -m src.evaluate_segmentation --weight_path weights/unet_weights_07072025.pt
    python -m src.evaluate_segmentation --weight_path weights/unet_weights_cldice.pt --output_csv results/segmentation_metrics_cldice.csv

Then diff two runs' output_csv (e.g. with pandas) to see whether clDice moved cl_dice_signal / dice_signal.
"""

import argparse
import csv
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.config.default import get_cfg
from src.dataset.ecg_scan import ECGScanDataset
from src.loss.loss import rgb_to_one_hot, soft_cl_dice
from src.transform.vision import RefineMask
from src.utils import find_config_path, import_class_from_path

CLASS_NAMES = ["grid", "text_background", "signal", "background"]


def resample_to_max_dim(tensor: torch.Tensor, max_dim: int, min_dim_floor: int = 512) -> torch.Tensor:
    """Resizes a (C, H, W) tensor to mirror InferenceWrapper._resample_image's behavior: upscales if the
    smaller side is below min_dim_floor, else downscales if the larger side exceeds max_dim, else leaves it
    untouched.

    This matters for two reasons, not just one: it keeps the benchmark evaluating the model at the same
    scale InferenceWrapper actually runs it at during real inference (it never feeds the model a raw scan
    at native dataset resolution), and separately, running this U-Net at certain raw/unresampled resolutions
    was observed to crash the CPU backend outright (access violation) during smoke-testing on Windows --
    resampling first avoids that class of problem too.
    """
    _, height, width = tensor.shape
    min_dim, max_dim_actual = min(height, width), max(height, width)
    x = tensor.unsqueeze(0)

    if min_dim < min_dim_floor:
        scale = min_dim_floor / min_dim
        new_size = (round(height * scale), round(width * scale))
        return F.interpolate(x, size=new_size, mode="bilinear", align_corners=False).squeeze(0)
    if max_dim_actual > max_dim:
        scale = max_dim / max_dim_actual
        new_size = (round(height * scale), round(width * scale))
        return F.interpolate(x, size=new_size, mode="bilinear", align_corners=False, antialias=True).squeeze(0)
    return tensor


def dice_score(
    probs: torch.Tensor, target_one_hot: torch.Tensor, union_exponent: int = 2, smooth: float = 1e-3
) -> torch.Tensor:
    """Per-class soft Dice *similarity* (not loss -- 1.0 is a perfect match), using the exact formula
    DiceFocalLoss optimizes during training (src/loss/loss.py) so these numbers are directly comparable to
    what training actually saw.

    Args:
        probs: (C, H, W) predicted class probabilities.
        target_one_hot: (C, H, W) one-hot(-ish) ground truth, same layout as rgb_to_one_hot's output.

    Returns:
        (C,) Dice score per class.
    """
    dims = (1, 2)
    intersection = (probs * target_one_hot).sum(dims)
    if union_exponent == 1:
        union = (probs + target_one_hot).sum(dims)
    else:
        union = (probs**2 + target_one_hot**2).sum(dims)
    return (2 * intersection + smooth) / (union + smooth)


def load_checkpoint(weight_path: str, model_cfg: Any, device: str) -> torch.nn.Module:
    model_class = import_class_from_path(model_cfg.class_path)
    model: torch.nn.Module = model_class(**model_cfg.KWARGS)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, tuple):
        checkpoint = checkpoint[0]
    checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    return model.eval().to(device)


@torch.no_grad()
def evaluate(
    weight_path: str,
    data_path: str,
    model_cfg: Any,
    signal_class: int = 2,
    device: str = "cpu",
    cl_dice_iterations: int = 5,
    output_csv: str = "results/segmentation_metrics.csv",
    max_images: Optional[int] = None,
    resample_size: int = 3000,
    min_dim_floor: int = 512,
) -> List[Dict[str, Any]]:
    model = load_checkpoint(weight_path, model_cfg, device)
    dataset = ECGScanDataset(data_path=data_path)
    refine_mask = RefineMask()

    n = len(dataset) if max_images is None else min(max_images, len(dataset))
    if n == 0:
        raise ValueError(f"No (image, mask) pairs found under {data_path!r}.")

    rows: List[Dict[str, Any]] = []
    for idx in range(n):
        scan, mask = dataset[idx]
        # Resize scan and mask identically so they stay pixel-aligned -- see resample_to_max_dim's
        # docstring for why this step exists at all.
        scan = resample_to_max_dim(scan, resample_size, min_dim_floor)
        mask = resample_to_max_dim(mask, resample_size, min_dim_floor)
        # Same ground-truth cleanup applied during training's val loop: normalizes grid/signal channels
        # and resolves overlap by letting the signal class take priority (RefineMask).
        scan, mask = refine_mask.forward(scan, mask)

        image = scan.unsqueeze(0).to(device)
        target = rgb_to_one_hot(mask.unsqueeze(0)).to(device)  # (1, 4, H, W)

        logits = model(image)
        probs = torch.softmax(logits, dim=1)

        dice = dice_score(probs[0], target[0])  # (4,)
        cl_dice_loss = soft_cl_dice(
            probs[:, [signal_class]], target[:, [signal_class]], iterations=cl_dice_iterations
        ).mean()

        row: Dict[str, Any] = {
            "index": idx,
            "file": os.path.basename(dataset.ecg_scan_files[idx]),
        }
        for c, name in enumerate(CLASS_NAMES):
            row[f"dice_{name}"] = float(dice[c].item())
        row["cl_dice_signal"] = float(1 - cl_dice_loss.item())  # score, not loss -- 1.0 is a perfect match
        rows.append(row)
        print(f"[{idx + 1}/{n}] {row}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-image metrics to {output_csv}")

    print("=== Summary (mean over dataset) ===")
    for key in rows[0]:
        if key in ("index", "file"):
            continue
        values = [r[key] for r in rows]
        print(f"{key}: {sum(values) / len(values):.4f}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation-only benchmark for a U-Net checkpoint.")
    parser.add_argument("--weight_path", required=True, help="Path to the .pt checkpoint to evaluate.")
    parser.add_argument(
        "--data_path",
        default="../../data/ecg_dataset/val",
        help="Dataset dir with <id>.png/<id>_mask.png pairs. NOTE: ECGScanDataset resolves relative paths "
        "against src/dataset/, not the repo root -- hence the '../../' default. Pass an absolute path to "
        "avoid this if it's confusing.",
    )
    parser.add_argument(
        "--model_config",
        default="unet.yml",
        help="Config file providing MODEL.class_path/KWARGS (searched in . and src/config/).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cl_dice_iterations", type=int, default=5)
    parser.add_argument("--output_csv", default="results/segmentation_metrics.csv")
    parser.add_argument("--max_images", type=int, default=None, help="Limit to the first N images (quick smoke test).")
    parser.add_argument(
        "--resample_size",
        type=int,
        default=1200,
        help="Max side length. NOTE: InferenceWrapper's own default is 3000, but full-resolution CPU "
        "inference through this U-Net was observed to crash outright (native access violation, not a "
        "catchable Python exception -- consistent with a CPU memory-allocator issue, not a code bug) on "
        "images around ~1650x2339 on at least one Windows machine. 1200 was confirmed to run cleanly there; "
        "raise it if your machine has more headroom, and lower it further if you still see a crash with no "
        "Python traceback.",
    )
    parser.add_argument("--min_dim_floor", type=int, default=512, help="Min side length, matching InferenceWrapper's default.")
    args = parser.parse_args()

    config_path = find_config_path(args.model_config)
    cfg = get_cfg(config_path)

    evaluate(
        weight_path=args.weight_path,
        data_path=args.data_path,
        model_cfg=cfg.MODEL,
        device=args.device,
        cl_dice_iterations=args.cl_dice_iterations,
        output_csv=args.output_csv,
        max_images=args.max_images,
        resample_size=args.resample_size,
        min_dim_floor=args.min_dim_floor,
    )
