# app/main.py

import os
import uuid

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

from app.pipeline import digitizer


app = FastAPI()


@app.get("/")
def health():

    return {
        "status": "healthy"
    }


@app.post("/digitize")
async def digitize_ecg(
    file: UploadFile = File(...)
):

    temp_dir = "temp"

    os.makedirs(
        temp_dir,
        exist_ok=True
    )

    extension = file.filename.split(".")[-1]

    temp_file = os.path.join(
        temp_dir,
        f"{uuid.uuid4()}.{extension}"
    )

    with open(temp_file, "wb") as f:
        f.write(
            await file.read()
        )

    try:

        result = digitizer.digitize(
            temp_file
        )

        return result

    finally:

        if os.path.exists(temp_file):
            os.remove(temp_file)





@app.post("/extract-signal")
async def extract_signal(
    file: UploadFile = File(...)
):

    temp_dir = "temp"

    os.makedirs(
        temp_dir,
        exist_ok=True
    )

    extension = file.filename.split(".")[-1]

    temp_file = os.path.join(
        temp_dir,
        f"{uuid.uuid4()}.{extension}"
    )

    with open(temp_file, "wb") as f:
        f.write(
            await file.read()
        )

    try:

        signal, meta = digitizer.extract_signal(
            temp_file
        )

        return {
            "layout": meta.get(
                "layout_name"
            ),
            "signal_shape": list(
                signal.shape
            ),
            "signal": signal.tolist()
        }

    finally:

        if os.path.exists(temp_file):
            os.remove(temp_file)