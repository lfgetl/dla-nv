import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from nemo.collections.tts.models import FastPitchModel
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.model.mel_spectrogram import MelSpectrogramConfig
from src.utils.io_utils import ROOT_PATH


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir, is_audio=False, *args, **kwargs):
        self._data_dir = Path(data_dir)
        self.is_audio = is_audio
        index = self._create_index()

        super().__init__(index, *args, **kwargs)

    def _create_index(self):  #
        index = []
        if not self.is_audio:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.spec = (
                FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
                .eval()
                .to(device)
            )

        for path in Path(self._data_dir).iterdir():
            if self.is_audio:
                index.append(
                    {
                        "file_id": str(path)
                        .split("/")[-1]
                        .split(".")[
                            0
                        ],  # честно не знаю в каком формате выдаст, лучше перестраховаться
                        "audio_path": str(path),
                    }
                )
            else:
                f = open(path)
                text = f.read()
                f.close()
                with torch.no_grad():
                    parsed = self.spec.parse(text, normalize=True)
                    spectrogram = self.spec.generate_spectrogram(parsed)
                    index.append(
                        {
                            "file_id": str(path).split("/")[-1].split(".")[0],
                            "audio_path": "",
                            "spectrogram": spectrogram.to("cpu"),
                        }
                    )

        return index
