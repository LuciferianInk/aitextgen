import logging
import os
import platform
import random
import re
import shutil
import sys
from datetime import datetime
from itertools import islice
from random import randint
from typing import List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from lightning.pytorch.callbacks import ModelPruning, StochasticWeightAveraging
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import CombinedLoader
from peft import PeftConfig, PeftModel
from peft.tuners.lora.layer import LoraLayer
from pkg_resources import resource_filename
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    PreTrainedTokenizerFast,
)

from .TokenDataset import TokenDataset
from .train import AIGProgressBar, AIGTrainer
from .utils import model_max_length, reset_seed, set_seed

logger = logging.getLogger("aigen")
logger.setLevel(logging.INFO)

STATIC_PATH = resource_filename(__name__, "static")


class aigen:
    # default values for GPT2Tokenizer
    tokenizer = None
    vocab_file = os.path.join(STATIC_PATH, "gpt2_vocab.json")
    merges_file = os.path.join(STATIC_PATH, "gpt2_merges.txt")
    bos_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"
    unk_token = "<|endoftext|>"
    pad_token = "<|endoftext|>"

    def __init__(
        self,
        model: str = None,
        model_folder: str = None,
        tokenizer_model: str = None,
        config: Union[str, GPT2Config] = None,
        vocab_file: str = None,
        merges_file: str = None,
        tokenizer_file: str = None,
        schema_tokens: List[str] = None,
        schema_return: List[str] = None,
        cache_dir: str = "aigen",
        embeddings_dir: str = "",
        precision: int = 32,
        gradient_checkpointing: bool = False,
        petals: bool = False,
        bos_token: str = None,
        eos_token: str = None,
        unk_token: str = None,
        adapters=None,
        adapter_dir: str = "adapters",
        tuning_mode=None,
        pre_seq_len=24,
        device_map="auto",
        **kwargs,
    ) -> None:
        self.mode = "transformer"
        self.memory = None
        self.precision = precision
        self.petals = petals

        # Enable tensor cores
        torch.set_float32_matmul_precision("medium")

        qargs = dict(torch_dtype=torch.float32)

        if precision in [16, 8, 4]:
            qargs["torch_dtype"] = torch.bfloat16

        if precision == 8:
            qargs["load_in_8bit"] = True
            qargs["llm_int8_has_fp16_weight"] = False
            qargs["llm_int8_threshold"] = 6

        if precision == 4:
            qargs["load_in_4bit"] = True
            qargs["bnb_4bit_quant_type"] = "nf4"
            qargs["bnb_4bit_use_double_quant"] = True
            qargs["bnb_4bit_compute_dtype"] = torch.bfloat16

        if config:
            # Manually construct a model from scratch
            logger.info("Constructing model from provided config.")
            if isinstance(config, str):
                config = AutoConfig.from_pretrained(config, low_cpu_mem_usage=True)
            for k, v in qargs.items():
                setattr(config, k, v)
            self.model = AutoModelForCausalLM.from_config(config)
        else:
            if model_folder:
                # A folder is provided containing pytorch_model.bin and config.json
                assert os.path.exists(
                    os.path.join(model_folder, "pytorch_model.bin")
                ) or os.path.exists(
                    os.path.join(model_folder, "model.safetensors")
                ), f"There is no pytorch_model.bin or model.safetensors file found in {model_folder}."
                assert os.path.exists(
                    os.path.join(model_folder, "config.json")
                ), f"There is no config.json in {model_folder}."
                logger.info(
                    f"Loading model from provided weights and config in {model_folder}."
                )
            else:
                # Download and cache model from Huggingface
                if os.path.isdir(cache_dir) and len(os.listdir(cache_dir)) > 0:
                    logger.info(f"Loading {model} model from {cache_dir}.")
                else:
                    logger.info(f"Downloading {model} model to {cache_dir}.")

            if self.petals:
                print("loading model from Petals")
                from petals import AutoDistributedModelForCausalLM

                self.model = AutoDistributedModelForCausalLM.from_pretrained(
                    model_folder if model_folder is not None else model,
                    pre_seq_len=pre_seq_len,
                    tuning_mode=tuning_mode,
                    cache_dir=cache_dir,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    **qargs,
                )
                embeddings_path = embeddings_dir + "/prompts.pt"
                if tuning_mode:
                    if os.path.exists(embeddings_path):
                        with open(embeddings_path, "rb") as f:
                            if torch.cuda.is_available():
                                self.model.transformer.prompt_embeddings = torch.load(f)
                                if tuning_mode == "deep_ptune":
                                    self.model.transformer.intermediate_prompt_embeddings = torch.load(
                                        f
                                    )
                            else:
                                self.model.transformer.prompt_embeddings = torch.load(
                                    f, map_location=torch.device("cpu")
                                )
                                if tuning_mode == "deep_ptune":
                                    self.model.transformer.intermediate_prompt_embeddings = torch.load(
                                        f, map_location=torch.device("cpu")
                                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_folder if model_folder is not None else model,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    local_files_only=True if model_folder else False,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    **qargs,
                )

        logger.info(f"Using the tokenizer for {model}.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model if model is not None else tokenizer_model,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        if adapters and not petals:
            for adapter in adapters:
                logger.info(f"Loading adapter: {adapter}")
                if adapters.index(adapter) == 0:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        f"{adapter_dir}/{adapter}",
                        adapter_name=adapter,
                        device_map=device_map,
                    )
                else:
                    self.model.load_adapter(
                        f"{adapter_dir}/{adapter}", adapter_name=adapter
                    )

            if len(adapters) > 1:
                logger.info("Merging adapters...")
                try:
                    self.model.add_weighted_adapter(
                        adapters=adapters,
                        weights=[1.0 / len(adapters)] * len(adapters),
                        adapter_name="combined",
                        combination_type="cat",
                    )
                except:
                    import traceback

                    print(traceback.format_exc())

                self.model.set_adapter("combined")

                for adapter in adapters:
                    logger.warning(f"Deleting unused adapter: {adapter}")
                    self.model.delete_adapter(adapter)

            logger.info(f"Using adapter: {self.model.active_adapter}")

        if gradient_checkpointing:
            logger.info("Gradient checkpointing enabled for model training.")
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
            setattr(self.model.config, "use_cache", None if petals else False)

        self.model_max_length = model_max_length(self.model.config)

        # arch = platform.machine()
        # if arch == "x86_64" and hasattr(self.model, "to_bettertransformer"):
        #     try:
        #         self.model.to_bettertransformer()
        #     except:
        #         pass

        self.model.eval()
        logger.info(self)

    def generate(
        self,
        prompt: str = "",
        min_length: int = None,
        max_new_tokens: int = None,
        seed: int = None,
        schema: str = False,
        mode: str = "transformer",
        generation_config: dict = None,
        **kwargs,
    ) -> Optional[str]:
        # Tokenize the prompt
        prompt_tensors = self.tokenizer(text=prompt, return_tensors="pt")

        if prompt:
            prompt_num_tokens = list(prompt_tensors["input_ids"].shape)[1]
            assert prompt_num_tokens < model_max_length(
                self.model.config
            ), f"The prompt is too large for the model. ({prompt_num_tokens} tokens)"

        input_ids = (
            prompt_tensors["input_ids"].to(self.get_device()) if prompt else None
        )

        attention_mask = (
            prompt_tensors["attention_mask"].to(self.get_device()) if prompt else None
        )

        if seed:
            set_seed(seed)

        self.mode = mode
        if mode in ["rnn"]:
            torch.set_grad_enabled(False)
            inputs = prompt_tensors["input_ids"].to(self.get_device())
            if self.memory is not None:
                self.memory = self.model(
                    inputs,
                    state=self.memory,
                ).state
            else:
                self.memory = self.model(inputs).state
            # print(self.memory[0][:, -2])

        gconfig = None
        if generation_config is not None:
            gconfig = GenerationConfig(**generation_config)

        # print(self.model.forward(input_ids))

        while True:
            outputs = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=gconfig,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_hidden_states=False,
                output_attentions=False,
                output_scores=False,
                num_return_sequences=1,
                state=self.memory,
                **kwargs,
            )

            gen_texts = self.tokenizer.batch_decode(
                outputs["sequences"], skip_special_tokens=True
            )

            # Handle stripping tokenization spaces w/ regex
            gen_texts = [re.sub(r"^\s+", "", text) for text in gen_texts]

            if min_length:
                gen_texts = list(filter(lambda x: len(x) > min_length, gen_texts))
            else:
                gen_texts = list(filter(lambda x: len(x) > 0, gen_texts))

            # if there is no generated text after cleanup, try again.
            if len(gen_texts) == 0:
                continue

            # Reset seed if used
            if seed:
                reset_seed()

            return gen_texts[0]

    def train(
        self,
        train_data: Union[str, TokenDataset],
        generation_config: dict = None,
        output_dir: str = "trained_model",
        n_gpu: int = -1,
        tpu_cores: int = 0,
        gradient_clip_val: float = 0.5,
        gradient_accumulation_steps: int = 1,
        seed: int = None,
        optimizer: str = "AdamW",
        loss_function: str = "default",
        learning_rate: float = 1e-3,
        lookahead: int = 0,
        momentum: float = 0,
        swa_learning_rate: float = None,
        update_period: int = 10,
        weight_decay: float = 0.05,
        eps: float = 1e-8,
        warmup_steps: int = 0,
        num_steps: int = 5000,
        save_every: int = 1000,
        generate_every: int = 1000,
        n_generate: int = 1,
        loggers: List = None,
        batch_size: int = 1,
        num_workers: int = None,
        benchmark: bool = True,
        num_layers_freeze: int = 0,
        scheduler: str = "linear",
        num_cycles=None,
        prune: float = 0.0,
        petals: bool = False,
        deepspeed: bool = False,
        block_size: int = 2048,
        val_split: float = 0.0,
        val_interval: int = 1000,
        supplement: bool = False,
        strategy=None,
        devices=None,
        **kwargs,
    ) -> None:
        self.petals = petals

        # arch = platform.machine()
        # if arch == "x86_64" and hasattr(self.model, "reverse_bettertransformer"):
        #     logging.info("Reversing BetterTransformer optimizations for training.")
        #     try:
        #         self.model.reverse_bettertransformer()
        #     except Exception as e:
        #         logging.error(e)
        #         pass

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        is_gpu_used = torch.cuda.is_available() and n_gpu != 0

        if isinstance(train_data, str):
            block_size = model_max_length(self.model.config)
            logger.info(
                f"Loading text from {train_data} with generation length of {block_size}."
            )
            train_data = TokenDataset(
                tokenizer=self.tokenizer,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                unk_token=self.unk_token,
                file_path=train_data,
                block_size=block_size,
                **kwargs,
            )

        setattr(self.model.config, "line_by_line", train_data.line_by_line)

        freeze_layers = False
        if num_layers_freeze > 0:
            logger.info("Layer freezing enabled for model training.")
            freeze_layers = True
            if num_layers_freeze:
                # For GPT-2
                if hasattr(self.model.config, "n_layer"):
                    assert (
                        num_layers_freeze < self.model.config.n_layer
                    ), "You are freezing more Transformer layers than in the model."
                # For GPT-Neo
                elif hasattr(self.model.config, "num_layers"):
                    assert (
                        num_layers_freeze < self.model.config.num_layers
                    ), "You are freezing more Transformer layers than in the model."
                # For RWKV
                elif hasattr(self.model.config, "num_hidden_layers"):
                    assert (
                        num_layers_freeze < self.model.config.num_hidden_layers
                    ), "You are freezing more Transformer layers than in the model."

        num_workers = (
            num_workers if num_workers is not None else int(os.cpu_count() / 4)
        )

        hparams = dict(
            optimizer=optimizer,
            loss_function=loss_function,
            learning_rate=learning_rate,
            lookahead=lookahead,
            momentum=momentum,
            update_period=update_period,
            weight_decay=weight_decay,
            eps=eps,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            num_steps=num_steps,
            pin_memory=is_gpu_used,
            num_workers=num_workers,
            save_every=save_every,
            generate_every=generate_every,
            use_tpu=tpu_cores > 0,
            scheduler=scheduler,
            num_cycles=num_cycles,
            petals=petals,
            deepspeed=deepspeed,
            val_split=val_split,
            val_interval=val_interval,
            block_size=block_size,
        )

        # Begin training
        if seed:
            set_seed(seed)

        # if try to use a GPU but no CUDA, use CPU
        # if not is_gpu_used:
        #     n_gpu = 1

        if devices is None:
            devices = [self.get_device().index]

        train_params = dict(
            accelerator="auto",
            strategy="auto",
            devices=devices,
            max_steps=num_steps,
            max_epochs=-1,
            val_check_interval=val_interval,
            reload_dataloaders_every_n_epochs=1,
            enable_checkpointing=False,
            precision="32-true",
            logger=loggers if loggers else False,
            callbacks=[
                AIGProgressBar(
                    save_every,
                    generate_every,
                    output_dir,
                    n_generate,
                    is_gpu_used,
                    freeze_layers,
                    num_layers_freeze,
                    petals,
                    GenerationConfig(**generation_config),
                ),
            ],
        )

        if deepspeed:
            train_params["strategy"] = DeepSpeedStrategy(
                stage=3,
                offload_optimizer=True,
                offload_parameters=True,
                allgather_bucket_size=2e8,
                reduce_bucket_size=2e8,
            )
        if strategy is not None:
            train_params["strategy"] = strategy

        train_params["accumulate_grad_batches"] = gradient_accumulation_steps

        if optimizer not in ["SophiaH"]:
            train_params["gradient_clip_val"] = gradient_clip_val
            train_params["gradient_clip_algorithm"] = "norm"

        if tpu_cores > 0:
            train_params["tpu_cores"] = tpu_cores
            train_params["devices"] = 0
            n_gpu = 0

        # benchmark gives a boost for GPUs if input size is constant,
        # which will always be the case with aigen training
        if is_gpu_used and benchmark:
            train_params["benchmark"] = True

        # if n_gpu > 1:
        #     train_params["strategy"] = "ddp"

        if prune > 0.0:
            # class CustomPruning(ModelPruning):
            #     def __init__(self, **kwargs):
            #         super().__init__(**kwargs)

            #     def apply_pruning(self, amount: Union[int, float]) -> None:
            #         super().apply_pruning(amount)
            #         for module, name in self._parameters_to_prune:
            #             self.make_pruning_permanent(module)

            modules_to_prune = []
            for n, m in self.model.named_modules():
                if isinstance(m, torch.nn.Embedding) or isinstance(m, torch.nn.Linear):
                    modules_to_prune.append(
                        (
                            m,
                            "weight",
                        )
                    )
            train_params["callbacks"].append(
                ModelPruning(
                    pruning_fn="random_unstructured",
                    amount=prune,
                    parameters_to_prune=list(set(modules_to_prune)),
                    use_global_unstructured=True,
                    resample_parameters=True,
                    apply_pruning=True,
                    make_pruning_permanent=True,
                    use_lottery_ticket_hypothesis=True,
                    prune_on_train_epoch_end=False,  # Prune on validation epoch end.
                    verbose=1,  # 0 to disable, 1 to log overall sparsity, 2 to log per-layer sparsity
                )
            )

        if swa_learning_rate:
            train_params["callbacks"].append(
                StochasticWeightAveraging(swa_lrs=swa_learning_rate)
            )

        data_module = AIGDataModule(self.get_device(), train_data, hparams)
        data_module.setup()

        train_split = data_module.train_dataloader()
        val_split = data_module.val_dataloader()

        final_train = [train_split]
        if supplement:
            streaming_module = StreamingDataModule(self.tokenizer, hparams)
            streaming_module.setup()
            streaming_train_split = streaming_module.train_dataloader()
            final_train = CombinedLoader(
                [train_split, streaming_train_split], mode="min_size"
            )

        # Wrap the model in a pytorch-lightning module
        train_model = AIGTrainer(
            self.model,
            train_split,
            hparams,
            self.tokenizer,
        )

        self.model.train()

        trainer = Trainer(**train_params)
        trainer.fit(train_model, final_train, val_split)

        if not petals:
            self.save(output_dir)

        if seed:
            reset_seed()

    def save(self, target_folder: str = os.getcwd()):
        """Saves the model into the specified directory."""
        logger.info(f"Saving trained model to {target_folder}")
        self.model.save_pretrained(target_folder, safe_serialization=True)

    def get_device(self) -> str:
        """Getter for the current device where the model is located."""
        return self.model.device

    def get_total_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    # This controls the output of the aigen object, when printed to console.
    def __repr__(self) -> str:
        # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/24
        num_params_m = int(sum(p.numel() for p in self.model.parameters()) / 10**6)
        model_name = type(self.model.config).__name__.replace("Config", "")
        return f"{model_name} loaded with {num_params_m}M parameters."


class AIGDataModule(LightningDataModule):
    def __init__(self, device, dataset, hparams):
        super().__init__()
        self.device = device
        self.dataset = dataset
        self.batch_size = hparams["batch_size"]
        self.pin_memory = hparams["pin_memory"]
        self.num_workers = hparams["num_workers"]
        self.val_split = hparams["val_split"]
        self.train = None
        self.val = None

    def setup(self):
        train_split = 1.0 - self.val_split
        self.train, self.val = torch.utils.data.random_split(
            self.dataset, [train_split, self.val_split]
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = super().transfer_batch_to_device(batch, self.device, dataloader_idx)
        return batch

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


class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        # self.content_key = "raw_content"
        self.content_key = "content"
        self.params = params
        self.dataset = load_dataset(
            "tiiuae/falcon-refinedweb",
            # "togethercomputer/RedPajama-Data-V2",
            # name="default",
            # snapshots=["2023-14"],
            # languages=["en"],
            split="train",
            streaming=True,
            cache_dir="/data/pile",
        )

    def __iter__(self):
        shuffled = self.dataset.shuffle(
            seed=random.randint(0, 2**31), buffer_size=10_000
        )

        for document in shuffled:
            tokenized = self.tokenizer(
                text=document.get(self.content_key),
                max_length=self.params["block_size"],
                stride=self.params["block_size"] - 32,
                padding=False,
                truncation=True,
                return_overflowing_tokens=True,
                return_tensors="np",
            )["input_ids"]
            choice = random.choice(tokenized)
            if np.size(choice) == 0:
                continue
            yield choice


class StreamingDataModule(LightningDataModule):
    def __init__(self, tokenizer, hparams):
        super().__init__()
        self.iterable = None
        self.tokenizer = tokenizer
        self.params = hparams

    def setup(self, stage=None):
        self.iterable = StreamingDataset(tokenizer=self.tokenizer, params=self.params)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = super().transfer_batch_to_device(batch, self.device, dataloader_idx)
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.iterable,
            batch_size=self.params["batch_size"],
            pin_memory=False,
            num_workers=self.params["num_workers"],
        )
