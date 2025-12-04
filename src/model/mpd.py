import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential
from torch.nn.utils import weight_norm


class SubMPD(nn.Module):
    """
    sub-discriminator for mpd
    """

    def __init__(self, p, relu_slope):
        super().__init__()

        self.p = p
        self.relu = nn.LeakyReLU(relu_slope)
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=128,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=1024,
                        kernel_size=(5, 1),
                        stride=(1, 1),
                        padding=(2, 0),
                    )
                ),
            ]
        )
        self.final_conv = weight_norm(
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
            )
        )

    def forward(self, audio):
        features = []
        x = audio.unsqueeze(1)
        b, c, t = x.shape
        if t % self.p != 0:
            n_pad = self.p - (t % self.p)
            t += n_pad
            x = F.pad(x, (0, n_pad), "replicate")
        x = x.view(b, c, t // self.p, self.p)
        for layer in self.convs:
            x = self.relu(layer(x))
            features.append(x)
        x = self.final_conv(x)
        features.append(x)

        return torch.flatten(x), features


class MPD(nn.Module):
    def __init__(self, relu_slope, ps):
        super().__init__()
        self.discs = nn.ModuleList([SubMPD(p, relu_slope) for p in ps])

    def forward(self, generated_audio, target_audio, **batch):
        res = {
            "mpd_outputs_gen": [],
            "mpd_outputs_real": [],
            "features_mpd_gen": [],
            "features_mpd_real": [],
        }
        for disc in self.discs:
            output_gen, feature_gen = disc(generated_audio)
            res["mpd_outputs_gen"].append(output_gen)
            res["features_mpd_gen"].append(feature_gen)
            output_real, feature_real = disc(target_audio)
            res["mpd_outputs_real"].append(output_real)
            res["features_mpd_real"].append(feature_real)

        return res

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
