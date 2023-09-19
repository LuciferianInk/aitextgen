import os
import requests
from tqdm.auto import tqdm
import torch
import numpy as np
import random
from transformers import GPT2Config, GPTNeoConfig


def download_file_with_progress(
    url_base: str, sub_dir: str, model_name: str, file_name: str
):
    """
    General utility for incrementally downloading files from the internet
    with progress bar.

    Adapted from gpt-2-simple.
    """

    # set to download 1MB at a time. This could be much larger with no issue
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024
    r = requests.get(
        os.path.join(url_base, "models", model_name, file_name), stream=True
    )
    with open(os.path.join(sub_dir, file_name), "wb") as f:
        file_size = int(r.headers["content-length"])
        with tqdm(
            desc="Fetching " + file_name,
            total=file_size,
            unit_scale=True,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
                pbar.update(DOWNLOAD_CHUNK_SIZE)


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


def skip_special_tokens(tensor, device, special_token_ids):
    """Filters out special tokens by ids in the given 1D tensor.

    Adapted from https://stackoverflow.com/a/62588955

    Args:
        tensor (tensor): PyTorch Tensor
        device (str): Device, usually "cpu" or "cuda:0"
        token_ids (set): List of Token IDs
    """
    special_token_id_tensor = torch.unique(torch.as_tensor(special_token_ids)).to(
        device
    )
    return tensor[
        ~tensor.unsqueeze(1).eq(special_token_id_tensor.unsqueeze(1)).any(1)
    ].tolist()


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
