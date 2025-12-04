import torch
import torch.nn.functional as F
from torch import nn


class TotalGenLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.fl = FeatureLoss()
        self.gl = GeneratorLoss()

    def forward(
        self,
        features_mpd_real,
        features_mpd_gen,
        features_msd_real,
        features_msd_gen,
        msd_outputs_gen,
        mpd_outputs_gen,
        spectrogram,
        generated_spectrogram,
        **batch
    ):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        fl_mpd = 2 * self.fl(features_mpd_real, features_mpd_gen)
        fl_msd = 2 * self.fl(features_msd_real, features_msd_gen)
        gl_mpd = self.gl(mpd_outputs_gen)
        gl_msd = self.gl(msd_outputs_gen)
        mel_loss = 45 * F.l1_loss(spectrogram, generated_spectrogram)
        loss = fl_mpd + fl_msd + gl_mpd + gl_msd + mel_loss
        return {
            "gen_loss": loss,
            "mpd_feature_loss": fl_mpd,
            "msd_feature_loss": fl_msd,
            "mpd_gan_loss": gl_mpd,
            "msd_gan_loss": gl_msd,
            "mel_loss": mel_loss,
        }


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features_real, features_gen):
        loss = 0.0
        for (
            fr,
            fg,
        ) in zip(features_real, features_gen):
            for r, g in zip(fr, fg):
                loss += F.l1_loss(r, g)

        return loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ds):
        loss = 0.0
        for d in ds:
            loss += torch.mean((d - 1) ** 2)
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mpd: bool, **batch):
        if mpd:
            d_gen = batch["mpd_outputs_gen"]
            d_real = batch["mpd_outputs_real"]
        else:
            d_gen = batch["msd_outputs_gen"]
            d_real = batch["msd_outputs_real"]
        loss = 0.0
        for dr, dg in zip(d_real, d_gen):
            loss += torch.mean((dr - 1) ** 2) + torch.mean(dg**2)
        loss_name = ("mpd" if mpd else "msd") + "_loss"
        return {loss_name: loss}
