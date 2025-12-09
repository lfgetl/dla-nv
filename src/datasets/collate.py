import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    if "spectrogram" in dataset_items[0]:
        result_batch["spectrogram"] = pad_sequence(
            [elem["spectrogram"].squeeze(0).permute(1, 0) for elem in dataset_items],
            batch_first=True,
        )
        result_batch["spectrogram"] = result_batch["spectrogram"].permute(0, 2, 1)
    if "target_audio" in dataset_items[0]:
        result_batch["target_audio"] = pad_sequence(
            [elem["target_audio"].squeeze(0) for elem in dataset_items],
            batch_first=True,
        )
    for field in ["file_id"]:  # keep the text for some future reason
        result_batch[field] = [elem[field] for elem in dataset_items]

    return result_batch
