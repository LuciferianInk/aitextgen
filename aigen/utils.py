import os
import requests
from tqdm.auto import tqdm
import torch
import numpy as np
import random
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


def find_index_of_subset(large_list, small_list):
    """
    Returns the index after the small_list within the large list,
    Returns -1 if not present.

    Adapted from https://stackoverflow.com/a/45819222 which shows that it is
    performant for input lengths < 1000, which is the common case for this function.
    """
    length_small_list = len(small_list)
    firstneedle = small_list[0]
    for idx, item in enumerate(large_list):
        if item == firstneedle:
            if large_list[idx : idx + length_small_list] == small_list:
                return idx + length_small_list
    return -1


def model_max_length(config):
    """Returns the maximum generation length for the given model."""
    length = getattr(config, "n_positions", None) or getattr(
        config, "max_position_embeddings", None
    )
    if length == None:
        length = 2048  # BLOOM
    return length


class bc:
    FOLD = "\033[94m"
    ROOT = "\033[92m"
    CORE = "\033[91m"


class ad:
    TEXT = "\033[0m"
