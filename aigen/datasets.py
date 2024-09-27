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
        self.np_weights = np.array(self.weights)
        self.config = hparams

    def estimate_coverage(self, batch):
        num_draws = batch * self.config["batch_size"]
        probabilities = 1 - np.exp(
            -self.np_weights * num_draws / np.sum(self.np_weights)
        )
        return np.mean(probabilities)

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
            num_workers=self.params["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.params["batch_size"],
            pin_memory=self.params["pin_memory"],
            num_workers=self.params["num_workers"],
        )


class HuggingfaceDataset(IterableDataset):
    def __init__(self, tokenizer, params, config, split="train"):
        assert config["schemas"], "You must provide a schema."

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

        self.buffer_size = self.config.get("buffer_size", 10_000)
        self.dataset = iter(
            load_dataset(
                config["repo"],
                split=self.split,
                streaming=True,
                cache_dir="./data/pile",
                trust_remote_code=True,
                keep_in_memory=False,
                **kwargs,
            ).shuffle(
                seed=random.randint(0, 2**31),
                buffer_size=self.buffer_size,
            )
        )

        self.cached_text = ""
        self.block_size = self.params["block_size"]
        sequence = [111, 222, 333]
        self.fake_sequence = create_fake_sequence(self.block_size, sequence)
        self.tokens = []
        self.sample_rate = self.config.get("sample_rate", 1.0)
        self.val_samples = self.config.get("val_samples", 0)

    def __iter__(self):
        return self

    def __next__(self):
        self._fill_cache()

        for batch in self.tokens:
            if len(batch) != self.block_size:
                continue
            if self.split != "train":
                self.val_samples -= 1
                if self.val_samples <= 0:
                    raise StopIteration
                return batch
            elif random.random() < self.sample_rate:
                return batch
            else:
                return self.fake_sequence

        return self.__next__()  # If we've exhausted current_tokens, try again

    def _fill_cache(self):
        text_cache_size = 10 * self.buffer_size
        delimiter = self.config.get("delimiter", "\n")
        if delimiter == "\\n":
            delimiter = "\n"
        elif delimiter == "\\t":
            delimiter = "\t"

        patterns = self.config.get("patterns", [])

        while len(self.cached_text) < text_cache_size:
            try:
                document = next(self.dataset)
            except StopIteration:
                raise StopIteration

            text = ""
            schema = random.choice(self.config["schemas"])
            items = list(schema.items())
            for i, (k, v) in enumerate(items):
                if len(document[k]) == 0:
                    continue

                for pattern in patterns:
                    identity = get_identity()
                    v = re.sub(pattern, identity, v)

                text += v + document[k]

                if i < len(items) - 1:
                    text += delimiter

            self.cached_text += text + self.tokenizer.eos_token

        if self.config.get("debug", False):
            print(self.cached_text[:4096])

        self.tokens = self.tokenizer(
            text=self.cached_text,
            max_length=self.block_size,
            stride=64,
            padding=True,
            truncation=True,
            return_overflowing_tokens=True,
            return_tensors="np",
        )["input_ids"]

        self.cached_text = ""


# def create_fake_sequence(block_size, sequence):
#     # Calculate how many pairs of elements we need
#     num_pairs = block_size // len(sequence)

#     # Create the list with repeating sequence
#     fake_list = sequence * num_pairs

#     # If block_size is not divisible by len(sequence), add remaining elements
#     remaining = block_size % len(sequence)
#     if remaining != 0:
#         fake_list.extend(sequence[:remaining])

#     # Convert to PyTorch tensor
#     fake_tensor = torch.tensor(fake_list, dtype=torch.long)

#     # Ensure the tensor is exactly block_size long
#     fake_tensor = fake_tensor[:block_size]

#     return fake_tensor


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
