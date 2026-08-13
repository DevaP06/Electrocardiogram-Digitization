"""
Downloads the public Open-ECG-Digitizer-Development-Dataset from Hugging Face and
exports it into the <id>.png / <id>_mask.png layout expected by
src.dataset.ecg_scan.ECGScanDataset, under data/ecg_dataset/{train,val}/.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Route the HF cache off the default C:\Users\<user>\.cache\huggingface (the dataset is ~75GB), and
# deliberately keep it OUT of the repo directory: datasets builds intermediate arrow files at paths
# like <HF_HOME>/datasets/Ahus-AIM___open-ecg-digitizer-development-dataset/default-<hash>/0.0.0/
# <40-char-hash>.incomplete/open-ecg-digitizer-development-dataset-train-00000-00000-of-NNNNN.arrow.
# That suffix alone is ~200 chars, so nesting it under a repo path blows past Windows' 260-char
# MAX_PATH (unless LongPathsEnabled is set) and fails with a confusing FileNotFoundError mid-generation
# -- after the whole multi-hour download has already succeeded. Override with HF_HOME if needed.
DEFAULT_CACHE = "D:/hf_cache_ecg" if os.name == "nt" else os.path.join(REPO_ROOT, "hf_cache")
os.environ.setdefault("HF_HOME", DEFAULT_CACHE)

from datasets import load_dataset  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "data", "ecg_dataset")


def export_split(ds_split: "object", split_name: str) -> None:
    out_dir = os.path.join(OUT_DIR, split_name)
    os.makedirs(out_dir, exist_ok=True)
    n = len(ds_split)  # type: ignore[arg-type]
    for i, example in enumerate(ds_split):  # type: ignore[arg-type]
        id_ = example["id"]
        img_path = os.path.join(out_dir, f"{id_}.png")
        mask_path = os.path.join(out_dir, f"{id_}_mask.png")
        if os.path.exists(img_path) and os.path.exists(mask_path):
            continue
        example["img"].convert("RGB").save(img_path)
        example["mask"].convert("RGB").save(mask_path)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"[{split_name}] {i + 1}/{n}", flush=True)


def main() -> None:
    print("Loading dataset (this triggers the ~75GB download on first run)...", flush=True)
    # The repo's README data_files glob (`data/train-*`) isn't picked up by auto
    # config detection in this datasets version, so pass it explicitly.
    ds = load_dataset(
        "Ahus-AIM/Open-ECG-Digitizer-Development-Dataset",
        data_files={"train": "data/train-*.parquet", "val": "data/val-*.parquet"},
    )
    print("Download complete. Exporting PNG pairs...", flush=True)
    export_split(ds["train"], "train")
    export_split(ds["val"], "val")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
