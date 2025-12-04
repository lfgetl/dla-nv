from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        generated = self.generator(**batch)  # generated_audio and generated_spectrogram
        gen_audio = generated["generated_audio"]
        mpd_res = self.mpd(generated_audio=gen_audio.detach(), **batch)
        loss_mpd = self.dicriminator_loss(mpd_res, mpd=True)
        msd_res = self.msd(generated_audio=gen_audio.detach(), **batch)
        loss_msd = self.dicriminator_loss(msd_res, mpd=False, **batch)
        loss_disc = loss_mpd + loss_msd
        if self.is_train:
            self.optimizer_d.zero_grad()
            loss_disc.backward()
            self.optimizer_d.step()
        for loss in loss_msd:
            loss_msd[loss] = loss_msd[loss].detach().cpu().item()
        batch.update(loss_msd)
        for loss in loss_mpd:
            loss_mpd[loss] = loss_mpd[loss].detach().cpu().item()
        batch.update(loss_mpd)
        msd_res = self.msd(generated_audio=gen_audio, **batch)
        mpd_res = self.mpd(generated_audio=gen_audio, **batch)
        total_res = {}
        total_res.update(mpd_res)
        total_res.update(msd_res)
        total_res.update(generated)
        for k in generated:
            batch[k] = generated[k].detach()

        loss_gen = self.generator_loss(**total_res)

        if self.is_train:
            self.optimizer_g.zero_grad()
            loss_gen["gen_loss"].backward()
            self.optimizer_g.step()
            self._clip_grad_norm()
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()
            if self.lr_scheduler_g is not None:
                self.lr_scheduler_g.step()

        for loss in loss_gen:
            loss_gen[loss] = loss_gen[loss].detach().cpu().item()
        batch.update(loss_gen)

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.writer.add_audio(
                "generated_audio", batch["audio_generated"][0], sample_rate=22050
            )
            self.writer.add_audio(
                "target_audio", batch["target_audio"][0], sample_rate=22050
            )
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.writer.add_audio(
                "generated_audio", batch["audio_generated"][0], sample_rate=22050
            )

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)
