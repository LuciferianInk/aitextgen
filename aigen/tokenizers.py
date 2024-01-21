import os
from typing import List, Union

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevel2
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevel1
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import ByteLevel as ByteLevel3
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


# https://huggingface.co/docs/tokenizers/quicktour
def train_tokenizer(
    files: Union[str, List[str]],
    dropout: float = None,
    vocab_size: int = 1000,
    min_frequency: int = 2,
    save_path: str = "aigen",
    added_tokens: List[str] = [],
    bos_token: str = "<|endoftext|>",
    eos_token: str = "<|endoftext|>",
    unk_token: str = "<|endoftext|>",
    pad_token: str = "<|endoftext|>",
) -> None:
    """
    Tokenizes the text(s) as a tokenizer, wrapping the tokenizer package.
    See: https://huggingface.co/blog/how-to-train

    For consistency, this function makes opinionated assuptions.

    :param files: path to file(s) to train tokenizer on
    :param dropout: Training dropout
    :param vocab_size: Final vocabulary size
    :param min_frequency: Minimum number of occurences to add to vocab
    :param save_path: Where to save the final tokenizer
    :param added_tokens: List of tokens to add to the tokenizer (currently not working)
    :param bos_token: Beginning-of-string special token
    :param eos_token: End-of-string special token
    :param unk_token: Unknown special token
    """

    assert isinstance(files, str) or isinstance(
        files, list
    ), "files must be a string or a list."

    assert isinstance(added_tokens, list), "added_tokens must be a list."

    if isinstance(files, str):
        files = [files]

    tokenizer = Tokenizer(
        BPE(
            cache_capacity=100_000,
            dropout=dropout,
            unk_token=unk_token,
            fuse_unk=True,
            byte_fallback=True,
        )
    )
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[unk_token, bos_token, eos_token, pad_token],
    )

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.pre_tokenizer = ByteLevel1(add_prefix_space=True, use_regex=True)
    tokenizer.decoder = ByteLevel2()
    tokenizer.post_processor = ByteLevel3(trim_offsets=True)

    # tokenizer.post_processor = TemplateProcessing(
    #     single="[CLS] $A [SEP]",
    #     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    #     special_tokens=[
    #         ("[CLS]", tokenizer.token_to_id("[CLS]")),
    #         ("[SEP]", tokenizer.token_to_id("[SEP]")),
    #     ],
    # )

    tokenizer.train(
        files=files,
        trainer=trainer,
    )

    trained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    trained_tokenizer.add_special_tokens(
        {
            "unk_token": unk_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
        }
    )

    os.makedirs(save_path, exist_ok=True)

    trained_tokenizer.save_pretrained(save_path)

    return trained_tokenizer
