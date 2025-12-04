import torch
from torch import nn
from torch.nn import Sequential

from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


def pad_size(kernel_size, dilation):
    return dilation * (kernel_size - 1) // 2


class MRF(nn.Module):
    def __init__(self, hidden_size, kernel_size, dilations, relu):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilations[0],
                    padding=pad_size(kernel_size, dilations[0]),
                ),
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilations[1],
                    padding=pad_size(kernel_size, dilations[1]),
                ),
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilations[2],
                    padding=pad_size(kernel_size, dilations[2]),
                ),
            ]
        )
        self.relu = nn.LeakyReLU(relu)
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=1,
                    padding=pad_size(kernel_size, 1),
                ),
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=1,
                    padding=pad_size(kernel_size, 1),
                ),
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=1,
                    padding=pad_size(kernel_size, 1),
                ),
            ]
        )

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            dx = self.relu(x)
            dx = conv1(x)
            dx = self.relu(x)
            dx = conv2(x)
            x += dx
        return x


class Generator(nn.Module):
    """
    Generator
    """

    def __init__(self, in_channels, hidden_size, k_u, k_r, dilations, relu_slope):
        """
        Args:

        """
        super().__init__()
        self.k_u = k_u
        self.k_r = k_r
        self.dilations = dilations
        self.relu = nn.LeakyReLU(relu_slope)
        self.mel = MelSpectrogram(MelSpectrogramConfig())

        self.first_conv = nn.Conv1d(
            in_channels=in_channels, out_channels=hidden_size, kernel_size=7, padding=3
        )

        self.convT = nn.ModuleList([])
        for i, l in enumerate(k_u):
            self.convT.append(
                nn.ConvTranspose1d(
                    in_channels=hidden_size // (2**i),
                    out_channels=hidden_size // (2 ** (i + 1)),
                    kernel_size=l,
                    stride=l // 2,
                    padding=l // 4,
                )
            )

        self.mrfs = []
        for i in range(len(k_u)):
            ml = nn.ModuleList([])
            channels = hidden_size // (2 ** (i + 1))
            for kernel_size in k_r:
                ml.append(MRF(channels, kernel_size, dilations, relu_slope))
            self.mrfs.append(ml)

        self.conv_end = nn.Conv1d(
            in_channels=hidden_size // (2 ** len(k_u)),
            out_channels=1,
            kernel_size=7,
            padding=3,
        )

    def forward(self, spectrogram, **batch):
        # returns generated audio and generated spectrogram
        x = self.first_conv(spectrogram)
        for i in range(len(self.k_u)):
            x = self.relu(x)
            x = self.convT[i](x)
            res = self.mrfs[i][0](x)
            for j, mrf in enumerate(self.mrfs[i]):
                if j == 0:
                    continue
                res += mrf(x)
        x /= len(self.k_u)
        x = torch.tanh(self.conv_end(self.relu(x)))
        return {"generated_audio": x.squeeze(1), "generated_spectrogram": self.mel(x)}

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
