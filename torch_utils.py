import logging
import math
from typing import Tuple

import torch
from numpy import ndarray
from numpy._typing import ArrayLike
from torch.types import Device
from torch.utils.data import TensorDataset, Dataset

from utils import prepare_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_training_env() -> Device:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {device}')
    return device


def get_data(device, n_lines: int = math.inf) -> tuple[ArrayLike, TensorDataset]:
    """Returns both the dates (to prevent time travel) and the dataset."""
    _, date, X, y, sw = prepare_data(n_lines)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    sw = torch.as_tensor(sw, dtype=torch.float32, device=device)
    dataset = TensorDataset(X, y, sw)
    return date, dataset
