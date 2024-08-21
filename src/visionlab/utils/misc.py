import os
import random
from typing import Any, List

import hydra
import numpy as np
import structlog
import torch

log = structlog.get_logger()


def set_seed(seed: int = 666, precision: int = 10) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(precision=precision)


def log_useful_info() -> None:
    log.info(hydra.utils.get_original_cwd())
    log.info(os.getcwd())


def split_list(data: List[Any], percentages: List[float]) -> List[List[Any]]:
    """
    Splits a list into sublists according to the given percentages.

    Parameters:
        data (List[Any]): The list to be split.
        percentages (List[float]): A list of percentages for splitting.

    Returns:
        List[List[Any]]: A list of sublists split according to the percentages.
    """
    if not sum(percentages) == 1.0:
        raise ValueError("The sum of the percentages must be 1.0")

    data_copy = data[:]
    random.shuffle(data_copy)
    total_len = len(data_copy)

    splits = []
    start = 0
    for pct in percentages:
        end = start + int(pct * total_len)
        splits.append(data_copy[start:end])
        start = end

    return splits
