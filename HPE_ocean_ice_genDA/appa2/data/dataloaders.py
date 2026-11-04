r"""Data loaders."""

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Any, Optional


def get_dataloader(
    dataset: Dataset,
    shuffle: bool = False,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    drop_last_ddp: bool = True,
    pin_memory: bool = True,
    **kwargs: Any,
) -> DataLoader:
    r"""Returns a possibly-distributed data loader for a given dataset.

    The sampler is created only if the rank is provided, and can be
    accessed with `dataloader.sampler`.

    Arguments:
        dataset: A Torch dataset.
        shuffle: Whether to shuffle the dataset or not.
        rank: The rank of the current process for distributed training. Default to None,
        corresponding to non distributed loader without sampler.
        world_size: The total number of processes for distributed training.
        drop_last_ddp: If true, adds indices to make the dataset length divisible by world_size.
                       If false, will ignore the last elements so that the dataset length is divisible.
        pin_memory: Pin the memory for faster move to the GPU.
        kwargs: Additional arguments passed to the DataLoader.
    """

    if rank is None:
        sampler = None
    else:
        sampler = DistributedSampler(
            dataset=dataset,
            drop_last=drop_last_ddp,
            shuffle=shuffle,
            rank=rank,
            num_replicas=world_size,
        )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        pin_memory=pin_memory,
        **kwargs,
    )

    return dataloader
