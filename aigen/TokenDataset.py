import csv
import gzip
import itertools
import logging
import math
import os
from pprint import pprint
from typing import List

import numpy as np
import torch
from pkg_resources import resource_filename
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

csv.field_size_limit(2**31 - 1)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STATIC_PATH = resource_filename(__name__, "static")


class TokenDataset(Dataset):
    def __init__(
        self,
        file_path: str = None,
        vocab_file: str = os.path.join(STATIC_PATH, "gpt2_vocab.json"),
        merges_file: str = os.path.join(STATIC_PATH, "gpt2_merges.txt"),
        tokenizer: PreTrainedTokenizer = None,
        tokenizer_file: str = None,
        texts: List[str] = None,
        line_by_line: bool = False,
        from_cache: bool = False,
        cache_destination: str = "dataset_cache.tar.gz",
        compress: bool = True,
        block_size: int = 1024,
        stride: int = 0,
        tokenized_texts: bool = False,
        text_delim: str = "\n",
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        unk_token: str = "<|endoftext|>",
        pad_token: str = "<|endoftext|>",
        **kwargs,
    ) -> None:
        self.block_size = block_size
        self.line_by_line = False

        # Special case; load tokenized texts immediately
        if tokenized_texts:
            self.tokens = tokenized_texts
            return

        assert any([texts, file_path]), "texts or file_path must be specified."

        # If a cache path is provided, load it.
        if from_cache:
            open_func = gzip.open if file_path.endswith(".gz") else open

            with open_func(file_path, "rb") as f:
                self.tokens = np.load(f)

            self.block_size = block_size
            self.line_by_line = line_by_line

            logger.info(f"TokenDataset containing {len(self.tokens)} batches loaded.")
            return

        assert tokenizer, "A tokenizer must be specified."
        assert os.path.isfile(
            file_path
        ), f"{file_path} is not present in the current directory."

        # if a file is specified, and it's line-delimited,
        # the text must be processed line-by-line into a a single bulk file
        if line_by_line:
            text_delim = None
            self.line_by_line = True
            self.file_path = file_path

        # if a file is specified, and it's not line-delimited,
        # the texts must be parsed as a single bulk file.
        else:
            eos_token = ""
            self.file_path = file_path

        self.tokens = self.encode_tokens(
            file_path,
            eos_token,
            tokenizer,
            text_delim,
            block_size,
            stride,
        )

        pprint(self.tokens)
        logger.info(f"There are {len(self.tokens)} batches of tokens.")

    def save(
        self, cache_destination: str = "dataset_cache.tar.gz", compress: bool = True
    ) -> None:
        if compress:
            open_func = gzip.open
        else:
            open_func = open
            cache_destination = (
                "dataset_cache.npy"
                if cache_destination == "dataset_cache.tar.gz"
                else cache_destination
            )

        logger.info(f"Caching dataset to {cache_destination}")

        with open_func(cache_destination, "wb") as f:
            np.save(f, self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]

    def __str__(self) -> str:
        return self.file_path if self.file_path is not None else "loaded dataset"

    def __repr__(self) -> str:
        return f"TokenDataset containing {len(self.tokens)} batches loaded."

    def encode_tokens(
        self,
        file_path: str,
        eos_token: str,
        tokenizer: PreTrainedTokenizer,
        newline: str,
        block_size: int = 256,
        stride: int = 0,
    ) -> List[int]:
        """
        Retrieve texts from a newline-delimited file.
        """

        num_texts = get_lines_in_file(file_path, newline)
        logger.info(f"Encoding {num_texts:,} lines of text from {file_path}.")

        with open(file_path, "r", encoding="utf-8", newline=newline) as file:
            if file_path.endswith(".csv"):
                # Strip the header
                file.readline()

                lines = csv.reader(file)
                while True:
                    content = [
                        text[0] + eos_token
                        for text in list(itertools.islice(lines, block_size))
                    ]
                    if not batch:
                        break
            else:
                content = file.read()

            tokenized = tokenizer(
                content,
                max_length=block_size,
                stride=stride,
                padding="max_length",
                return_overflowing_tokens=True,
                truncation=True,
                return_tensors="pt",
            )

            return tokenized["input_ids"]


def get_lines_in_file(file_path: str, newline: str = None) -> int:
    """
    Returns the number of lines in a file to build progress bar.
    c.f. https://stackoverflow.com/a/16108605/9314418
    """

    with open(file_path, "r", encoding="utf-8", newline=newline) as f:
        return sum(1 for row in f)


def merge_datasets(datasets: List[TokenDataset], equalize: bool = True) -> TokenDataset:
    """
    Merges multiple TokenDatasets into a single TokenDataset.
    This assumes that you are using the same tokenizer for all TokenDatasets.

    :param datasets: A list of TokenDatasets.
    :param equalize: Whether to take an equal amount of samples from all
    input datasets (by taking random samples from
    each dataset equal to the smallest dataset)
    in order to balance out the result dataset.
    """

    assert (
        isinstance(datasets, list) and len(datasets) > 1
    ), "datasets must be a list of multiple TokenDatasets."

    len_smallest = min([len(dataset) for dataset in datasets])
    block_size = datasets[0].block_size

    tokenized_texts = []

    for dataset in datasets:
        assert (
            dataset.block_size == block_size
        ), "The input datasets have different block sizes."
        if equalize:
            tokenized_texts.extend(dataset.tokens[0:len_smallest])
        else:
            tokenized_texts.extend(dataset.tokens)

    return TokenDataset(tokenized_texts=tokenized_texts, block_size=block_size)
