import os
from typing import List, Union

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast


def train_tokenizer(
    files: Union[str, List[str]],
    dropout: float = None,
    byte_fallback: bool = True,
    vocab_size: int = 1000,
    min_frequency: int = 2,
    max_token_length: int = 5,
    save_path: str = "trained_model",
    added_tokens: List[str] = [],
    bos_token: str = "<|void|>",
    eos_token: str = "<|void|>",
    unk_token: str = "<|void|>",
    pad_token: str = "<|void|>",
) -> None:
    """
    Tokenizes the text(s) as a tokenizer, wrapping the tokenizer package.
    See: https://huggingface.co/docs/tokenizers/quicktour

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
        models.BPE(
            dropout=dropout,
            byte_fallback=byte_fallback,
            unk_token=unk_token,
            fuse_unk=True,
        )
    )
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_token_length=max_token_length,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[
            unk_token,
            pad_token,
            bos_token,
            eos_token,
        ],
    )

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False, trim_offsets=True, use_regex=True
            ),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel(
        add_prefix_space=True, trim_offsets=True, use_regex=True
    )
    tokenizer.post_processor = processors.ByteLevel(
        add_prefix_space=True, trim_offsets=True, use_regex=True
    )

    # tokenizer.post_processor = processors.TemplateProcessing(
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
            "pad_token": pad_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
        }
    )

    os.makedirs(save_path, exist_ok=True)

    trained_tokenizer.save_pretrained(save_path)

    return trained_tokenizer
