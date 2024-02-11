import random

import numpy as np
import torch


def model_max_length(config):
    """Returns the maximum generation length for the given model."""
    length = (
        getattr(config, "context_length", None)
        or getattr(config, "n_positions", None)
        or getattr(config, "max_position_embeddings", None)
        or getattr(config, "n_ctx", None)
        or getattr(config, "hidden_size", 2048)
    )
    return length


class colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    WHITE = "\033[0m"
