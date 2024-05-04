import csv
import gzip
import itertools
import logging
import math
import os
import random
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
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

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


class StaticDataModule(LightningDataModule):
    def __init__(self, dataset, hparams):
        super().__init__()
        self.dataset = dataset
        self.batch_size = hparams["batch_size"]
        self.pin_memory = hparams["pin_memory"]
        self.num_workers = hparams["num_workers"]
        self.val_split = hparams["val_split"]
        self.train = None
        self.val = None
        self.setup()

    def setup(self):
        train_split = 1.0 - self.val_split
        self.train, self.val = torch.utils.data.random_split(
            self.dataset, [train_split, self.val_split]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


class StreamingDataModule(LightningDataModule):
    def __init__(self, tokenizer, hparams, config):
        super().__init__()
        self.train_data = None
        self.tokenizer = tokenizer
        self.params = hparams
        self.setup(config)

    def setup(self, config):
        if config.get("sequential", False):
            self.train_data = SequentialStreamingDataset(
                self.tokenizer, self.params, config, split="train"
            )
        elif config.get("instruct", False):
            self.train_data = InstructStreamingDataset(
                self.tokenizer, self.params, config, split="train"
            )
        elif config.get("chat", False):
            self.train_data = ChatStreamingDataset(
                self.tokenizer, self.params, config, split="train"
            )
        else:
            self.train_data = StreamingDataset(
                self.tokenizer, self.params, config, split="train"
            )

        if config.get("val_samples", 0) > 0:
            if config.get("instruct", False) and config.get("sample_rate", 1.0) < 1.0:
                self.val_data = InstructStreamingDataset(
                    self.tokenizer, self.params, config, split="validation"
                )
            elif config.get("chat", False) and config.get("sample_rate", 1.0) < 1.0:
                self.val_data = ChatStreamingDataset(
                    self.tokenizer, self.params, config, split="validation"
                )
            else:
                self.val_data = StreamingDataset(
                    self.tokenizer, self.params, config, split="validation"
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


class StreamingDataset(IterableDataset):
    def __init__(self, tokenizer, params, config, split="train"):
        self.config = config
        self.tokenizer = tokenizer
        self.content_key = config.get("content_key", "default")
        self.params = params
        kwargs = {}
        for k, v in config.items():
            if k in ["snapshots", "subset", "languages"]:
                if k == "subset":
                    k = "name"
                kwargs[k] = v
        self.dataset = load_dataset(
            config["repo"],
            split=split,
            streaming=True,
            cache_dir="/data/pile",
            trust_remote_code=True,
            **kwargs,
        )
        self.params["num_workers"] = min(
            self.params["num_workers"], self.dataset.n_shards
        )

    def __iter__(self):
        shuffled = self.dataset.shuffle(
            seed=random.randint(0, 2**31),
            buffer_size=self.config.get("buffer_size", 10_000),
        )

        block_size = self.params["block_size"]

        samples = self.config.get("val_samples", 0)

        batch = []
        for document in shuffled:
            tokenized = self.tokenizer(
                text=document.get(self.content_key),
                max_length=block_size,
                stride=block_size - 32,
                padding=False,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="np",
            )["input_ids"]
            choice = random.choice(tokenized)
            if len(choice) == 0:
                continue
            elif len(batch) == 0:
                batch = choice
            else:
                np.append(batch, self.tokenizer.eos_token_id)
                batch = np.concatenate([batch, choice])
            if len(batch) >= block_size:
                while True:
                    if random.random() < self.config.get("sample_rate", 1.0):
                        break
                    yield np.array([self.tokenizer.eos_token_id] * block_size).astype(
                        "int64"
                    )
                yield batch[:block_size]
                batch = []
                if samples > 0:
                    samples -= 1
                    if samples == 0:
                        samples = self.config.get("val_samples", 0)
                        break
            else:
                continue


class InstructStreamingDataset(StreamingDataset):
    def __iter__(self):
        shuffled = self.dataset.shuffle(
            seed=random.randint(0, 2**31),
            buffer_size=self.config.get("buffer_size", 10_000),
        )

        block_size = self.params["block_size"]
        wall = self.config.get("wall", "¶")
        ship = self.config.get("ship", ":>")

        for document in shuffled:
            if random.random() < self.config.get("sample_rate", 1.0):
                human = self.config["identity_function"]()
                robot = self.config["identity_function"]()
                content = (
                    f"{wall}{human}{ship} {document.get('definition')}\n"
                    f"{wall}{human}{ship} {document.get('inputs')}\n"
                    f"{wall}{robot}{ship} {document.get('targets')}"
                )
                tokens = self.tokenizer(
                    text=content,
                    max_length=block_size,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=False,
                    return_tensors="np",
                )["input_ids"]
                batch = np.array([]).astype("int64")
                for block in tokens:
                    batch = np.concatenate([batch, block])
                yield batch.astype("int64")
            else:
                yield np.array([self.tokenizer.eos_token_id] * block_size).astype(
                    "int64"
                )


class ChatStreamingDataset(StreamingDataset):
    def __iter__(self):
        shuffled = self.dataset.shuffle(
            seed=random.randint(0, 2**31),
            buffer_size=self.config.get("buffer_size", 10_000),
        )

        num_epochs = 1_000_000

        block_size = self.params["block_size"]
        wall = self.config.get("wall", "¶")
        ship = self.config.get("ship", ":>")

        for epoch in range(num_epochs):
            self.dataset.set_epoch(epoch)
            for document in shuffled:
                if random.random() < self.config.get("sample_rate", 1.0):
                    human = self.config["identity_function"]()
                    robot = self.config["identity_function"]()
                    orig = document.get("text")
                    new = orig.replace("Tom:", f"\n{wall}{robot}{ship}").replace(
                        "Sarah:", f"\n{wall}{human}{ship}"
                    )
                    tokens = self.tokenizer(
                        text=new,
                        max_length=block_size,
                        padding="max_length",
                        truncation=True,
                        return_overflowing_tokens=False,
                        return_tensors="np",
                    )["input_ids"]
                    batch = np.array([]).astype("int64")
                    for block in tokens:
                        batch = np.concatenate([batch, block])
                    yield batch.astype("int64")
                else:
                    yield np.array([self.tokenizer.eos_token_id] * block_size).astype(
                        "int64"
                    )


class SequentialStreamingDataset(StreamingDataset):
    def __iter__(self):
        shuffled = self.dataset.shuffle(
            seed=random.randint(0, 2**31),
            buffer_size=self.config.get("buffer_size", 10_000),
        )

        block_size = self.params["block_size"]

        assert block_size % 2 == 0, "`block_size` must be divisible by 2."

        half_block = int(block_size / 2)

        batch = np.array([])
        for document in shuffled:
            tokens = self.tokenizer(
                text=document.get(self.content_key),
                max_length=block_size,
                padding=False,
                truncation=True,
                stride=0,
                return_overflowing_tokens=True,
                return_tensors="np",
            )["input_ids"]

            if len(tokens) == 0:
                continue

            for block in tokens:
                batch = np.concatenate([batch, block])

            if len(batch) < block_size:
                np.append(batch, self.tokenizer.eos_token_id)
                continue

            while len(batch) >= block_size:
                selection = batch[:block_size]
                batch = batch[half_block:]
                while True:
                    if random.random() < self.config.get("sample_rate", 1.0):
                        yield selection.astype("int64")
                        break
                    yield np.array([self.tokenizer.eos_token_id] * block_size).astype(
                        "int64"
                    )


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
