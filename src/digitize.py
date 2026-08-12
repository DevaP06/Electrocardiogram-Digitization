import argparse
import os
import random
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torchvision.io import decode_image
from tqdm import tqdm
from yacs.config import CfgNode as CN

from src.config.default import get_cfg
from src.utils import find_config_path, import_class_from_path

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def get_candidate_file_paths(config: CN) -> list[str]:
    candidate_filenames: list[str] = []
    for root, _, files in os.walk(config.DATA.images_path):
        for file in files:
            if file.endswith(tuple(config.DATA.image_extensions)):
                candidate_filenames.append(os.path.join(root, file))
    random.seed(42)
    random.shuffle(candidate_filenames)
    return candidate_filenames


def clear_and_prepare_output_dir(config: CN) -> None:
    output_path: str = config.DATA.output_path
    if os.path.exists(output_path):
        for root, dirs, files in os.walk(output_path, topdown=False):
            if config.DATA.get("clear_output_dir_if_exists", False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copy_file_structure(config.DATA.images_path, output_path)


def copy_file_structure(src: str, dst: str) -> None:
    for root, dirs, _ in os.walk(src):
        rel_path: str = os.path.relpath(root, src)
        target_dir: str = os.path.join(dst, rel_path)
        os.makedirs(target_dir, exist_ok=True)


def decode_and_prepare_image(file_path: str) -> torch.Tensor:
    image: torch.Tensor = decode_image(file_path, mode="RGB")
    C, H, W = image.shape
    if C == 1:
        image = image.expand(3, H, W)
    elif C == 4:
        image = image[:3]
    return image.unsqueeze(0)


def canonical_from_got_values(got_values: dict[str, Any]) -> torch.Tensor | None:
    canonical: torch.Tensor | None = None
    if "signal" in got_values and isinstance(got_values["signal"], dict):
        canonical = got_values["signal"].get("canonical_lines")
    elif "canonical_lines" in got_values:
        canonical = got_values["canonical_lines"]
    return canonical


def save_lead_confidence_csv(got_values: dict[str, Any], output_basepath: str) -> None:
    """Writes per-lead coverage/smoothness/confidence scores, so a low-confidence lead is a reportable
    metric a caller can check and flag, instead of a digitized-but-wrong signal that looks fine at a
    glance."""
    lead_confidence: dict[str, dict[str, float]] | None = got_values.get("signal", {}).get("lead_confidence")
    if not lead_confidence:
        return
    with open(output_basepath + "_lead_confidence.csv", "w") as f:
        f.write("lead,coverage,smoothness,confidence\n")
        for lead_name, scores in lead_confidence.items():
            f.write(f"{lead_name},{scores['coverage']:.4f},{scores['smoothness']:.4f},{scores['confidence']:.4f}\n")


def save_timeseries_csv(canonical: torch.Tensor | None, output_basepath: str) -> None:
    if canonical is None:
        return
    data: npt.NDArray[Any] = canonical.squeeze().cpu().numpy()
    # Assume shape (n_leads, n_points)
    if data.ndim == 1:
        data = data[None, :]
    n_leads = data.shape[0]
    col_names = LEAD_NAMES[:n_leads]
    # Transpose to shape (n_points, n_leads)
    data = data.T
    header = ",".join(col_names)
    np.savetxt(output_basepath + "_timeseries_canonical.csv", data, delimiter=",", header=header, comments="")


def save_png_plot(got_values: dict[str, Any], canonical: torch.Tensor | None, output_basepath: str) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    axs[0, 0].imshow(got_values["input_image"].squeeze().permute(1, 2, 0).cpu().numpy() * 0.999)
    source_points = got_values["source_points"]
    axs[0, 0].scatter(source_points[:, 0].cpu().numpy(), source_points[:, 1].cpu().numpy(), s=20, c="red")
    axs[0, 1].imshow(got_values["aligned"]["image"].squeeze().permute(1, 2, 0).cpu().numpy() * 0.999)
    axs[1, 0].imshow(got_values["aligned"]["signal_prob"].squeeze().cpu().numpy(), interpolation="none", vmin=0, vmax=1)
    for i in range(0, 15, 2):
        for j in range(0, 15, 2):
            xval = i * 5 / got_values["pixel_spacing_mm"]["x"]
            yval = j * 5 / got_values["pixel_spacing_mm"]["y"]
            axs[0, 1].add_patch(
                plt.Rectangle(  # type: ignore
                    (xval, yval),
                    width=5 / got_values["pixel_spacing_mm"]["x"],
                    height=5 / got_values["pixel_spacing_mm"]["y"],
                    edgecolor="red",
                    facecolor="none",
                )
            )
    if canonical is not None:
        lines: npt.NDArray[Any] = canonical.squeeze().cpu().numpy()
        lines -= np.linspace(0, 24_000, num=lines.shape[0])[:, None]  # 2 uV offset per lead
        axs[1, 1].plot(lines.T, linewidth=0.5)
    plt.tight_layout()
    plt.suptitle(
        got_values.get("layout_name", "") + " Layout cost: " + f'{got_values["signal"]["layout_matching_cost"]:.2f}',
        fontsize=16,
    )
    plt.savefig(output_basepath + ".png", dpi=200)
    plt.close()


def save_matching_cost(got_values: dict[str, Any], output_basepath: str) -> None:
    # if no csv called digization_metadata.csv exists, create it with header "matching_cost, is_flipped, lead_layout"
    metadata_file = os.path.join(os.path.dirname(output_basepath), "digitization_metadata.csv")
    if not os.path.exists(metadata_file):
        with open(metadata_file, "w") as f:
            f.write("file_path,matching_cost,is_flipped,lead_layout,mv_per_mm,calibration_detected,calibration_num_leads\n")
    # then append the values
    with open(metadata_file, "a") as f:
        file_name = os.path.basename(output_basepath)
        matching_cost = got_values.get("signal", {}).get("layout_matching_cost", float("nan"))
        is_flipped = got_values.get("is_flipped", False)
        lead_layout = got_values.get("layout_name", "")
        calibration = got_values.get("calibration", {})
        mv_per_mm = calibration.get("mv_per_mm", float("nan"))
        calibration_detected = calibration.get("detected", False)
        calibration_num_leads = calibration.get("num_leads_detected", 0)
        f.write(
            f"{file_name},{matching_cost},{is_flipped},{lead_layout},"
            f"{mv_per_mm},{calibration_detected},{calibration_num_leads}\n"
        )


def save_outputs(got_values: dict[str, Any], output_basepath: str, save_mode: str = "all") -> None:
    canonical = canonical_from_got_values(got_values)
    if save_mode in ["all", "timeseries_only"]:
        save_timeseries_csv(canonical, output_basepath)
        save_lead_confidence_csv(got_values, output_basepath)
    if save_mode in ["all", "png_only"]:
        save_png_plot(got_values, canonical, output_basepath)
    save_matching_cost(got_values, output_basepath)


def get_layout_should_include_substring(file_path: str, config: CN) -> str | None:
    if config.DATA.get("layout_should_include_substring") is not None:
        if "limb" in str(file_path):
            return "limb"
        elif "precordial" in str(file_path):
            return "precordial"
    return None


def save_result_for_file(file_path: str, config: CN, got_values: dict[str, Any], save_mode: str) -> None:
    if config.DATA.get("output_path") is not None:
        rel_path = os.path.relpath(file_path, config.DATA.images_path)
        output_file_path = os.path.join(config.DATA.output_path, rel_path)
        output_basepath = os.path.splitext(output_file_path)[0]
        os.makedirs(os.path.dirname(output_basepath), exist_ok=True)
        save_outputs(got_values, output_basepath, save_mode)


def process_one_file(file_path: str, config: CN, inference_wrapper: Any, save_mode: str) -> None:
    image = decode_and_prepare_image(file_path)
    layout_should_include_substring = get_layout_should_include_substring(file_path, config)
    got_values = inference_wrapper(image, layout_should_include_substring=layout_should_include_substring)
    save_result_for_file(file_path, config, got_values, save_mode)


def process_file_batch(file_paths: list[str], config: CN, inference_wrapper: Any, save_mode: str) -> None:
    """Digitizes a batch of files in one call to forward_batch() instead of one file at a time -- see
    InferenceWrapper.forward_batch for what batching does and does not speed up. Falls back to
    processing files one by one (via the existing per-file error handling in main()) if the batch as a
    whole fails, so one bad file can't take an entire batch of otherwise-fine files down with it."""
    images = [decode_and_prepare_image(file_path) for file_path in file_paths]
    substrings = [get_layout_should_include_substring(file_path, config) for file_path in file_paths]

    try:
        results = inference_wrapper.forward_batch(images, substrings)
    except Exception as e:
        print(f"Batch of {len(file_paths)} files failed ({e}); retrying one file at a time.")
        for file_path in file_paths:
            try:
                process_one_file(file_path, config, inference_wrapper, save_mode)
            except Exception as file_error:
                print(f"Error processing {file_path}: {file_error}")
        return

    for file_path, got_values in zip(file_paths, results):
        save_result_for_file(file_path, config, got_values, save_mode)


def main(config: CN) -> None:
    inference_wrapper_class = import_class_from_path(config.MODEL.class_path)
    inference_wrapper = inference_wrapper_class(**config.MODEL.KWARGS)
    save_mode: str = getattr(config.DATA, "save_mode", "all")
    batch_size: int = config.DATA.get("batch_size", 1)
    file_paths: list[str] = get_candidate_file_paths(config)

    include_list = config.DATA.get("path_should_include", [])

    if config.DATA.get("output_path") is not None:
        clear_and_prepare_output_dir(config)

    included_paths = []
    for file_path in file_paths:
        should_be_included = not bool(include_list)
        for include in include_list:
            if include in file_path:
                should_be_included = True
        if should_be_included:
            included_paths.append(file_path)

    if batch_size <= 1:
        for file_path in tqdm(included_paths):
            print(f"Processing file: {file_path}")
            try:
                process_one_file(file_path, config, inference_wrapper, save_mode)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    else:
        batches = [included_paths[i : i + batch_size] for i in range(0, len(included_paths), batch_size)]
        for batch in tqdm(batches):
            print(f"Processing batch: {batch}")
            process_file_batch(batch, config, inference_wrapper, save_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digitize ECGs and save output.")
    parser.add_argument(
        "--config",
        type=str,
        help="Config file name or path (searched in . and src/config/)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override config options like A.B.C=123 or DATA.save_mode='png_only' (spaces require quotes).",
    )
    args = parser.parse_args()

    config_path = find_config_path(args.config)
    cfg = get_cfg(config_path)

    if args.overrides:
        kv_list: list[str] = []
        for ov in args.overrides:
            if "=" not in ov:
                raise ValueError(f"Malformed override '{ov}'. Use KEY=VALUE (e.g., MODEL.device='cuda:0').")
            k, v = ov.split("=", 1)

            # Check if key exists in config before merging
            node = cfg
            key_parts = k.split(".")
            exists = True
            for part in key_parts[:-1]:
                if part in node:
                    node = node[part]
                else:
                    exists = False
                    break
            if not exists or key_parts[-1] not in node:
                warnings.warn(f"Config key '{k}' not found in loaded config. Override skipped.")
                continue

            kv_list.extend([k, v])

        if kv_list:
            cfg.merge_from_list(kv_list)

    main(cfg)
