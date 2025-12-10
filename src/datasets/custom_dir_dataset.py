import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
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
            fastpitch, _ = torch.hub.load(
                "NVIDIA/DeepLearningExamples:torchhub", "nvidia_fastpitch"
            )
            self.tp = torch.hub.load(
                "NVIDIA/DeepLearningExamples:torchhub",
                "nvidia_textprocessing_utils",
                cmudict_path="cmudict-0.7b",
                heteronyms_path="heteronyms",
            )
            self.spec = fastpitch.eval().to(device)

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
                    batch = self.tp.prepare_input_sequence([text], batch_size=1)[0]
                    mel, mel_lens, *_ = self.spec(batch["text"].to(device))
                    index.append(
                        {
                            "file_id": str(path).split("/")[-1].split(".")[0],
                            "audio_path": "",
                            "spectrogram": mel[0].detach().to("cpu"),
                        }
                    )

        return index
