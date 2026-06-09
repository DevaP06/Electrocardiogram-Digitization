# app/pipeline.py

import os
from pathlib import Path

import numpy as np
import torch
from torchvision.io import decode_image

from src.config.default import get_cfg
from src.utils import import_class_from_path
from src.digitize import canonical_from_got_values


BASE_DIR = Path(__file__).resolve().parent.parent


class ECGDigitizer:

    def __init__(self):

        # Make all relative paths inside the repo resolve correctly
        os.chdir(BASE_DIR)

        cfg = get_cfg(
            str(
                BASE_DIR /
                "src" /
                "config" /
                "inference_wrapper_ahus_testset.yml"
            )
        )

        wrapper_class = import_class_from_path(
            cfg.MODEL.class_path
        )

        self.wrapper = wrapper_class(
            **cfg.MODEL.KWARGS
        )

    def preprocess(self, image_path):

        image = decode_image(
            image_path,
            mode="RGB"
        )

        C, H, W = image.shape

        if C == 1:
            image = image.expand(3, H, W)

        elif C == 4:
            image = image[:3]

        return image.unsqueeze(0)

    def extract_signal(self, image_path):

        image = self.preprocess(image_path)

        got_values = self.wrapper(
            image,
            layout_should_include_substring=None
        )

        canonical = canonical_from_got_values(
            got_values
        )

        if canonical is None:
            raise Exception(
                "Failed to generate ECG signal"
            )

        signal = (
            canonical
            .squeeze()
            .cpu()
            .numpy()
        )

        signal = np.nan_to_num(
            signal,
            nan=0.0
        )

        return signal, got_values

    def digitize(self, image_path):

        signal, got_values = self.extract_signal(
            image_path
        )

        return {
            "layout": got_values.get("layout_name"),
            "signal_shape": list(signal.shape)
        }


digitizer = ECGDigitizer()