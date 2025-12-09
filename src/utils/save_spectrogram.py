import os
import sys
from pathlib import Path

import torch
import torchaudio
from torchvision.utils import save_image

from src.logger.utils import plot_spectrogram
from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


def save_spectrogram(path_orig, save_path):
    sp = Path(save_path)
    sp.mkdir(exist_ok=True)
    target_sr = 22050
    mel = MelSpectrogram(MelSpectrogramConfig())
    for path in Path(path_orig).iterdir():
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        spectrogram = mel(audio_tensor).squeeze(0)
        filename = str(path).split("/")[-1].split(".")[0]
        save_image(
            plot_spectrogram(spectrogram, filename),
            str(sp / (filename + ".png")),
        )


if __name__ == "__main__":
    save_spectrogram(sys.argv[1], sys.argv[2])
