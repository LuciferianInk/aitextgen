import os
import random

import numpy as np
import requests
import torch
from tqdm.auto import tqdm
from transformers import GPT2Config, GPTNeoConfig


def set_seed(seed: int):
    """
    Sets the seed for all potential generation libraries.
    """

    assert isinstance(seed, int), "seed must be an integer."
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_seed():
    """
    Resets the seed for all potential generation libraries.
    """
    random.seed()
    np.random.seed()
    torch.seed()
    torch.cuda.seed_all()


def model_max_length(config):
    """Returns the maximum generation length for the given model."""
    length = (
        getattr(config, "context_length", None)
        or getattr(config, "n_positions", None)
        or getattr(config, "max_position_embeddings", None)
        or getattr(config, "hidden_size", None)
        or getattr(config, "n_ctx", 2048)
    )
    return length


class colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    WHITE = "\033[0m"
