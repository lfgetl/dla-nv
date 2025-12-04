import itertools
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metric
    generator_loss = instantiate(config.gen_loss_function).to(device)
    discriminator_loss = instantiate(config.dis_loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params_g = filter(lambda p: p.requires_grad, model.generator.parameters())
    trainable_params_d = filter(
        lambda p: p.requires_grad,
        itertools.chain([model.mpd.parameters(), model.msd.parameters()]),
    )
    optimizer_g = instantiate(config.optimizer_g, params=trainable_params_g)
    optimizer_d = instantiate(config.optimizer_d, params=trainable_params_d)
    lr_scheduler_g = instantiate(config.lr_scheduler_g, optimizer=optimizer_g)
    lr_scheduler_d = instantiate(config.lr_scheduler_d, optimizer=optimizer_d)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        generator_loss=generator_loss,
        discriminator_loss=discriminator_loss,
        metrics=metrics,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
