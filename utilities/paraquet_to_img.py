from huggingface_hub import hf_hub_download
import pandas as pd
from PIL import Image
import io
import os

# Create save folder
os.makedirs("ecg_samples", exist_ok=True)

# Download only one parquet shard
file_path = hf_hub_download(
    repo_id="Ahus-AIM/Open-ECG-Digitizer-Development-Dataset",
    repo_type="dataset",
    filename="data/train-00000-of-00068.parquet"
)

print("Downloaded:", file_path)

# Read parquet
df = pd.read_parquet(file_path)

print("Columns:", df.columns)

# Save first 10 ECG images
for i in range(10):

    sample = df.iloc[i]

    # image bytes from img column
    img_bytes = sample["img"]["bytes"]

    # convert bytes to image
    image = Image.open(io.BytesIO(img_bytes))

    # save image
    save_path = f"ecg_samples/ecg_{i}.png"
    image.save(save_path)

    print("Saved:", save_path)