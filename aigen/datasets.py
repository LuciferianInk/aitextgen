import csv
import gzip
import itertools
import logging
import math
import os
import random
import re
import sys
import textwrap
from pprint import pprint
from typing import List

import datasets
import numpy as np
import torch
from datasets import load_dataset
from lightning.pytorch.core.datamodule import LightningDataModule
from pkg_resources import resource_filename
from torch.utils.data import DataLoader, Dataset, IterableDataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from .utils import get_identity

csv.field_size_limit(2**31 - 1)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STATIC_PATH = resource_filename(__name__, "static")


class StaticDataset(Dataset):
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
        batch_size: int = 10000,
        block_size: int = 1024,
        stride: int = 0,
        tokenized_texts: bool = False,
        text_delim: str = "\n",
        bos_token: str = "<|void|>",
        eos_token: str = "<|void|>",
        unk_token: str = "<|void|>",
        pad_token: str = "<|void|>",
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

            logger.info(f"StaticDataset containing {len(self.tokens)} batches loaded.")
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
            batch_size,
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
        return f"StaticDataset containing {len(self.tokens)} batches loaded."

    def encode_tokens(
        self,
        file_path: str,
        eos_token: str,
        tokenizer: PreTrainedTokenizer,
        newline: str,
        batch_size: int = 10000,
        block_size: int = 256,
        stride: int = 0,
    ) -> List[int]:
        """
        Retrieve texts from a newline-delimited file.
        """

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

            batches = [
                content[i : i + batch_size] for i in range(0, len(content), batch_size)
            ]
            tokenized_batches = []
            max_len = 0
            with tqdm(total=len(batches)) as pbar:
                for batch in batches:
                    tokenized = tokenizer(
                        batch,
                        max_length=block_size,
                        stride=stride,
                        padding="max_length",
                        return_overflowing_tokens=True,
                        truncation=True,
                        return_tensors="np",
                    )
                    max_len = max(max_len, tokenized["input_ids"].shape[1])
                    tokenized_batches.append(tokenized["input_ids"])
                    pbar.update(1)

            padded_batches = []
            for batch in tokenized_batches:
                padding_length = max_len - batch.shape[1]
                if padding_length > 0:
                    padded_batch = np.pad(
                        batch,
                        ((0, 0), (0, padding_length)),
                        mode="constant",
                        constant_values=tokenizer.pad_token_id,
                    )
                else:
                    padded_batch = batch
                padded_batches.append(padded_batch)

            tokens = np.concatenate(padded_batches)

            return tokens


class LocalDataModule(LightningDataModule):
    def __init__(self, train_data, val_data, weights, hparams):
        super().__init__()
        self.train = train_data
        self.val = val_data
        self.weights = weights
        self.config = hparams

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            self.weights, num_samples=len(self.weights), replacement=True
        )
        return DataLoader(
            self.train,
            batch_size=self.config["batch_size"],
            pin_memory=self.config["pin_memory"],
            num_workers=self.config["num_workers"],
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            shuffle=False,
            batch_size=self.config["batch_size"],
            pin_memory=self.config["pin_memory"],
            num_workers=self.config["num_workers"],
        )


class StreamingDataModule(LightningDataModule):
    def __init__(self, tokenizer, hparams, config):
        super().__init__()
        self.train_data = None
        self.tokenizer = tokenizer
        self.params = hparams
        self.setup(config)

    def setup(self, config):

        print(f"repo: {config['repo']}")
        pprint(config)

        if config.get("hf", False):
            self.train_data = HuggingfaceDataset(
                self.tokenizer, self.params, config, split="train"
            )

        if config.get("val_split"):
            self.val_data = HuggingfaceDataset(
                self.tokenizer, self.params, config, split=config["val_split"]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.params["batch_size"],
            pin_memory=self.params["pin_memory"],
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.params["batch_size"],
            pin_memory=self.params["pin_memory"],
            num_workers=1,
        )


class HuggingfaceDataset(IterableDataset):
    def __init__(self, tokenizer, params, config, split="train"):

        assert config["schema"], "You must provide a schema."

        self.config = config
        self.tokenizer = tokenizer
        self.params = params
        self.split = split
        kwargs = {}
        for k, v in config.items():
            if k in ["snapshots", "subset", "languages"]:
                if k == "subset":
                    k = "name"
                kwargs[k] = v
        self.dataset = load_dataset(
            config["repo"],
            split=self.split,
            streaming=True,
            cache_dir="/data/pile",
            trust_remote_code=True,
            **kwargs,
        )

        self.cached_text = ""

    def __iter__(self):

        buffer_size = self.config.get("buffer_size", 10_000)
        text_cache_size = 10 * buffer_size
        shuffled = self.dataset.shuffle(
            seed=random.randint(0, 2**31),
            buffer_size=buffer_size,
        )

        block_size = self.params["block_size"]

        delimiter = self.config.get("delimiter", "\n")
        if delimiter == "\\n":
            delimiter = "\n"
        elif delimiter == "\\t":
            delimiter = "\t"

        patterns = self.config.get("patterns", [])

        val_samples = self.config.get("val_samples", 0)

        for document in shuffled:
            if self.split != "train":
                if val_samples <= 0:
                    break

            text = ""
            items = list(self.config["schema"].items())
            for i, (k, v) in enumerate(items):
                for pattern in patterns:
                    identity = get_identity()
                    v = re.sub(pattern, identity, v)

                text += v + document[k]

                if i < len(items) - 1:
                    text += delimiter

            text += self.tokenizer.eos_token

            self.cached_text += text
            if len(self.cached_text) < text_cache_size:
                continue

            if self.config.get("debug", False):
                print(self.cached_text[4096:])

            tokens = self.tokenizer(
                text=self.cached_text,
                max_length=block_size,
                stride=64,
                padding=False,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="np",
            )["input_ids"]

            self.cached_text = ""

            for batch in tokens:
                if len(batch) != block_size:
                    break
                while True:
                    if self.split != "train":
                        val_samples -= 1
                        yield batch
                        break
                    elif random.random() < self.config.get("sample_rate", 1.0):
                        yield batch
                        break
                    else:
                        yield create_fake_sequence(
                            block_size,
                            [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id],
                        )


def create_fake_sequence(block_size, sequence):
    # Calculate how many pairs of [333, 444] we need
    num_pairs = block_size // 2

    # Create the list with alternating 333 and 444
    fake_list = sequence * num_pairs

    # If block_size is odd, add one more element
    if block_size % 2 != 0:
        fake_list.append(sequence[0])

    # Convert to NumPy array
    fake_array = np.array(fake_list, dtype=np.int64)

    # Ensure the array is exactly block_size long
    fake_array = fake_array[:block_size]

    return fake_array


def merge_datasets(
    datasets: List[StaticDataset], equalize: bool = True
) -> StaticDataset:
    """
    Merges multiple StaticDatasets into a single StaticDataset.
    This assumes that you are using the same tokenizer for all StaticDatasets.

    :param datasets: A list of StaticDatasets.
    :param equalize: Whether to take an equal amount of samples from all
    input datasets (by taking random samples from
    each dataset equal to the smallest dataset)
    in order to balance out the result dataset.
    """

    assert (
        isinstance(datasets, list) and len(datasets) > 1
    ), "datasets must be a list of multiple StaticDatasets."

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

    return StaticDataset(tokenized_texts=tokenized_texts, block_size=block_size)
