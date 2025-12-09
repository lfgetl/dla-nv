import csv
import json
import os

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class NVDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "LJSpeech-1.1"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir

        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):  #
        index = []
        dir_part = self._data_dir
        read_path = dir_part / "metadata.csv"
        dir_audio = dir_part / "wavs"

        with open(read_path, "r", newline="") as file:
            reader = csv.reader(file, delimiter="|", quoting=csv.QUOTE_NONE)
            for i, row in enumerate(tqdm(reader)):
                if part == "train" and i % 100 == 0:
                    continue
                if part == "val" and i % 100 != 0:
                    continue
                id, text, normalized_text = row
                audio_path = str(dir_audio / (id + ".wav"))

                index.append(
                    {
                        "file_id": id,
                        "audio_path": audio_path,
                    }
                )
        return index
