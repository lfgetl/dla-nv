import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential
from torch.nn.utils import spectral_norm, weight_norm

from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


class SubMSD(nn.Module):
    def __init__(self, relu_slope, norm):
        super().__init__()
        self.relu = nn.LeakyReLU(relu_slope)
        self.convs = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, padding=7)
                ),
                norm(
                    nn.Conv1d(
                        in_channels=16,
                        out_channels=64,
                        kernel_size=41,
                        stride=4,
                        groups=4,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        in_channels=64,
                        out_channels=256,
                        kernel_size=41,
                        stride=4,
                        groups=16,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        in_channels=256,
                        out_channels=1024,
                        kernel_size=41,
                        stride=4,
                        groups=64,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        in_channels=1024,
                        out_channels=1024,
                        kernel_size=41,
                        stride=4,
                        groups=256,
                        padding=20,
                    )
                ),
                norm(
                    nn.Conv1d(
                        in_channels=1024, out_channels=1024, kernel_size=5, padding=2
                    )
                ),
            ]
        )
        self.final_conv = norm(
            nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=3, padding=1)
        )

    def forward(self, audio):
        x = audio.unsqueeze(1)
        features = []
        for layer in self.convs:
            x = self.relu(layer(x))
            features.append(x)
        x = self.final_conv(x)
        features.append(x)

        return torch.flatten(x), features


class MSD(nn.Module):
    def __init__(self, relu_slope):
        super().__init__()
        self.relu = nn.LeakyReLU(relu_slope)

        self.discs = nn.ModuleList(
            [
                SubMSD(relu_slope, spectral_norm),
                SubMSD(relu_slope, weight_norm),
                SubMSD(relu_slope, weight_norm),
            ]
        )

        self.pooling = nn.ModuleList(
            [
                None,
                nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
                nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            ]
        )

    def forward(self, generated_audio, target_audio, **batch):
        res = {
            "msd_outputs_gen": [],
            "msd_outputs_real": [],
            "features_msd_gen": [],
            "features_msd_real": [],
        }
        for i, disc in enumerate(self.discs):
            if self.pooling[i] is not None:
                generated_audio = self.pooling[i](generated_audio)
                target_audio = self.pooling[i](target_audio)
            output_gen, feature_gen = disc(generated_audio)
            res["msd_outputs_gen"].append(output_gen)
            res["features_msd_gen"].append(feature_gen)
            output_real, feature_real = disc(target_audio)
            res["msd_outputs_real"].append(output_real)
            res["features_msd_real"].append(feature_real)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
