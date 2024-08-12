import logging
import math
import os
import platform
import random
import re
import time
import traceback
from typing import List, Optional, Union

import torch
from lightning.fabric.utilities.seed import reset_seed, seed_everything
from lightning.pytorch.accelerators import TPUAccelerator
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    ModelPruning,
    StochasticWeightAveraging,
)
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities import CombinedLoader
from peft import PeftMixedModel, PeftModel
from pkg_resources import resource_filename
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    LogitsWarper,
    TopKLogitsWarper,
)

from .datasets import LocalDataModule, StaticDataset, StreamingDataModule
from .optimizers import get_optimizer
from .schedulers import get_schedule
from .strategies import get_strategy
from .train import (
    AIGMetricsLogger,
    AIGModelSaver,
    AIGProgressBar,
    AIGSampleGenerator,
    AIGTrainer,
)
from .utils import colors, model_max_length

logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
logger = logging.getLogger("aigen")
logger.setLevel(logging.INFO)

STATIC_PATH = resource_filename(__name__, "static")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class aigen:
    def __init__(
        self,
        model: str = None,
        model_folder: str = None,
        tokenizer: AutoTokenizer = None,
        config: Union[str, AutoConfig] = None,
        vocab_file: str = None,
        merges_file: str = None,
        tokenizer_folder: str = None,
        embeddings_dir: str = "",
        precision: int = 32,
        petals: bool = False,
        adapters=None,
        cache_dir: str = "models",
        adapter_dir: str = "adapters",
        tuning_mode=None,
        pre_seq_len=24,
        device_map="auto",
        attn_implementation="eager",
        **kwargs,
    ) -> None:
        self.mode = "transformer"
        self.memory = None
        self.precision = precision
        self.petals = petals

        # Enable tensor cores
        torch.set_float32_matmul_precision("medium")

        qargs = dict(torch_dtype=torch.float32)

        if precision in [128]:
            qargs["torch_dtype"] = torch.complex128

        if precision in [64]:
            qargs["torch_dtype"] = torch.float64

        if precision in [32]:
            qargs["torch_dtype"] = torch.float32

        if precision in ["b16"]:
            qargs["torch_dtype"] = torch.bfloat16

        if precision in [16, 8, 4]:
            qargs["torch_dtype"] = torch.float16

        if precision == 8:
            qargs["load_in_8bit"] = True
            qargs["llm_int8_has_fp16_weight"] = False
            qargs["llm_int8_threshold"] = 6.0

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
            setattr(config, "torch_dtype", qargs["torch_dtype"])
            print(f"{colors.GREEN}modified pretrain config:{colors.WHITE}")
            print(config)
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
                # from petals.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS
                # from petals.server.server import Server

                # server = Server(
                #     # **args,
                #     # host_maddrs=host_maddrs,
                #     # announce_maddrs=announce_maddrs,
                #     # compression=compression,
                #     # max_disk_space=max_disk_space,
                #     converted_model_name_or_path=(
                #         model_folder if model_folder is not None else model
                #     ),
                #     dht_prefix=None,
                #     initial_peers=PUBLIC_INITIAL_PEERS,
                #     public_name="https://src.eco",
                #     cache_dir="/data/models",
                #     num_blocks=24,
                #     torch_dtype="bfloat16",
                #     throughput=1.0,
                # )
                # try:
                #     server.run()
                # except KeyboardInterrupt:
                #     logger.info("Caught KeyboardInterrupt, shutting down")
                # finally:
                #     server.shutdown()

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
                    attn_implementation=attn_implementation,
                    **qargs,
                )

        logger.info(f"Using the tokenizer for {model}.")
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(
                tokenizer_folder if tokenizer_folder else model,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        )

        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
            setattr(self.tokenizer, "pad_token", self.tokenizer.eos_token)

        if adapters and not petals:
            for adapter in adapters:
                logger.info(f"Loading adapter: {adapter}")
                if adapters.index(adapter) == 0:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        f"{adapter_dir}/{adapter}",
                        adapter_name=adapter,
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
                        weights=[1.0] * len(adapters),
                        adapter_name="combined",
                        combination_type="linear",
                    )
                except:
                    print(traceback.format_exc())

                # logger.info("Using adapters: ", adapters)
                logger.warning(f"Created new adapter: combined")
                self.model.set_adapter("combined")
                # self.model.set_adapter(adapters)

                for adapter in adapters:
                    logger.warning(f"Deleting unused adapter: {adapter}")
                    self.model.delete_adapter(adapter)

            # logger.info(f"Using adapter: {self.model.active_adapter}")

        self.model.eval()
        logger.info(self)

    @property
    def model_max_length(self):
        return model_max_length(self.model.config)

    def load_adapter(self, adapter_dir):
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_dir,
            is_trainable=True,
        )
        setattr(self.model.config, "is_prompt_learning", False)

    def create_adapter(self, kwargs):
        try:
            from peft import get_peft_model, prepare_model_for_kbit_training

            from .adapters import get_peft_config

            peft_config = get_peft_config(
                peft_type=kwargs.get("type", "lora"),
                kwargs=kwargs,
            )

            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=kwargs.get("gradient_checkpointing", False),
            )

            self.model = get_peft_model(self.model, peft_config)
        except Exception as e:
            print(self.model)
            raise e

    def optimize_for_inference(self):
        arch = platform.machine()
        if arch == "x86_64" and hasattr(self.model, "to_bettertransformer"):
            try:
                self.model.to_bettertransformer()
            except Exception as e:
                logger.warning(e)

    def generate(
        self,
        assistant=None,
        prompt: str = "",
        min_length: int = None,
        max_new_tokens: int = None,
        seed: int = None,
        mode: str = "transformer",
        generation_config: dict = None,
        forbidden_chars: list = [],
        **kwargs,
    ) -> Optional[str]:
        if seed:
            seed_everything(seed)

        prompt_tensors = self.tokenizer(text=prompt, return_tensors="pt")

        if prompt:
            prompt_num_tokens = list(prompt_tensors["input_ids"].shape)[1]
            assert (
                prompt_num_tokens < self.model_max_length
            ), f"The prompt is too large for the model. ({prompt_num_tokens} tokens)"

        input_ids = (
            prompt_tensors["input_ids"].to(self.get_device()) if prompt else None
        )

        attention_mask = (
            prompt_tensors["attention_mask"].to(self.get_device()) if prompt else None
        )

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

        # Prepare the logits processor list
        logits_processor = LogitsProcessorList()
        if len(forbidden_chars) > 0:
            # Initialize the custom logit processor
            top_k = 50
            custom_processor = CharacterSuppressionTopKLogitsWarper(
                self.tokenizer, forbidden_chars, top_k
            )
            logits_processor.append(custom_processor)

        # import sys

        # from transformers import TextStreamer

        # class RealTimeStreamer(TextStreamer):
        #     def on_finalized_text(self, text: str, stream_end: bool = False):
        #         print(text, end="", flush=True)
        #         sys.stdout.flush()

        # streamer = RealTimeStreamer(self.tokenizer)

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
                # streamer=streamer,
                state=self.memory,
                assistant_model=assistant if assistant else None,
                tokenizer=self.tokenizer if not self.petals else None,
                logits_processor=logits_processor,
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

            reset_seed()

            return gen_texts[0]

    def _get_params(self, model, hparams):
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue

            if any(nd in n for nd in no_decay):
                weight_decay = 0.0
            else:
                weight_decay = hparams["weight_decay"]

            grouped_parameters.append(
                {
                    "params": [p],
                    "weight_decay": weight_decay,
                }
            )

        return grouped_parameters

    def prepare_datasets(self, hparams, local_data, streaming_data):
        self.total_train = []
        self.total_val = []

        for dataset in local_data:
            module = LocalDataModule(
                dataset["train"], dataset["val"], dataset["weights"], hparams
            )
            self.total_train.append(module.train_dataloader())
            self.total_val.append(module.val_dataloader())

        self.static_len = sum(len(dataset) for dataset in self.total_train)

        for dataset in streaming_data:
            module = StreamingDataModule(self.tokenizer, hparams, dataset)
            self.total_train.append(module.train_dataloader())
            if dataset.get("val_samples"):
                self.total_val.append(module.val_dataloader())

        self.combined_train = CombinedLoader(self.total_train, mode="min_size")
        self.combined_val = CombinedLoader(self.total_val, mode="min_size")

    def train(
        self,
        local_data: Union[str, StaticDataset] = [],
        streaming_data: [] = [],
        generation_config: dict = None,
        output_dir: str = "trained_model",
        gradient_clip_val: float = 1.0,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        seed: int = None,
        optimizer: str = "AdamW",
        scheduler: str = "cosine",
        num_cycles: int = None,
        learning_rate: float = 1e-3,
        lookahead: int = 0,
        momentum: float = 0,
        swa_learning_rate: float = None,
        weight_decay: float = 0,
        eps: float = 1e-8,
        warmup_steps: int = 0,
        num_steps: int = 5000,
        save_every: int = 0,
        generate_every: int = 0,
        loggers: List = None,
        batch_size: int = 1,
        num_workers: int = None,
        prune: float = 0.0,
        petals: bool = False,
        block_size: int = 2048,
        val_split: float = 0.0,
        val_interval: int = 1000,
        initial_piers: list = [],
        target_batch_size: int = 8192,
        strategy: str = "auto",
        finetune: bool = False,
        progress_bar: bool = True,
        checkpoint_every: int = 0,
        resume: bool = False,
        verbose: bool = True,
        devices=None,
        overfit: bool = False,
        callbacks=[],
        **kwargs,
    ) -> None:
        if hasattr(self.model, "training"):
            self.model.training = True

        if seed:
            seed_everything(seed)

        is_gpu_used = torch.cuda.is_available()

        if devices is None:
            device = self.get_device().index
            devices = [device]
            if device is None:
                devices = -1

        if os.environ.get("DEVICE", "auto") == "cpu":
            devices = 1

        # This is a hack, but prevents HivemindStrategy from placing models
        # onto the wrong device.
        if is_gpu_used and strategy == "hivemind":
            try:
                torch.cuda.set_device(devices[0])
            except Exception as e:
                logging.error(e)
                torch.cuda.set_device(0)

        num_workers = (
            num_workers if num_workers is not None else int(os.cpu_count() / 2)
        )

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
            setattr(self.model.config, "use_cache", None if petals else False)

        hparams = dict(
            optimizer=optimizer,
            scheduler=scheduler,
            learning_rate=learning_rate,
            lookahead=lookahead,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            num_steps=num_steps,
            pin_memory=is_gpu_used,
            num_workers=num_workers,
            num_cycles=num_cycles,
            petals=petals,
            val_split=val_split,
            block_size=block_size,
            initial_piers=initial_piers,
            target_batch_size=target_batch_size,
            accumulate_grad_batches=gradient_accumulation_steps,
            **kwargs,
        )

        train_params = dict(
            accelerator="auto",
            strategy="auto",
            devices=devices,
            max_steps=num_steps,
            max_epochs=-1,
            val_check_interval=(
                1000 if overfit else val_interval * gradient_accumulation_steps
            ),
            reload_dataloaders_every_n_epochs=1,
            enable_checkpointing=True if checkpoint_every > 0 else False,
            precision="32-true",
            accumulate_grad_batches=gradient_accumulation_steps,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm="norm",
            benchmark=True,
            callbacks=callbacks,
            logger=loggers if loggers else False,
            overfit_batches=1000 if overfit else 0,
        )

        train_params["callbacks"].append(AIGMetricsLogger())

        if checkpoint_every > 0:
            checkpoint_callback = ModelCheckpoint(
                save_top_k=2,
                monitor="step",
                mode="max",
                every_n_train_steps=checkpoint_every,
                dirpath=output_dir,
                filename="model",
            )

            train_params["callbacks"].append(checkpoint_callback)
            print(f"Model checkpointing enabled.")

        latest_checkpoint = None
        if resume and checkpoint_every > 0:
            num_versions = 1000
            this_version = 1000
            # this checkpoint handling is kind of stupid, but we do it this way
            # because Lighting sometimes likes to increment model versions, instead
            # of just overwriting the previous checkpoints
            for _ in range(num_versions):
                this_version -= 1
                ckpt_path = f"{output_dir}/model-v{this_version}.ckpt"
                if os.path.exists(ckpt_path):
                    latest_checkpoint = ckpt_path
                    break

            if latest_checkpoint is None:
                latest_checkpoint = f"{output_dir}/model.ckpt"

            print(f"Resuming training from: {latest_checkpoint}")

        if finetune:
            from finetuning_scheduler import FinetuningScheduler

            train_params["callbacks"].append(FinetuningScheduler())

            logging.info(f"Using a naive finetuning schedule.")

        if prune > 0.0:
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
                    apply_pruning=True,
                    amount=prune,
                    make_pruning_permanent=False,
                    use_lottery_ticket_hypothesis=True,
                    resample_parameters=True,
                    pruning_fn="random_unstructured",
                    use_global_unstructured=True,
                    prune_on_train_epoch_end=False,
                    parameters_to_prune=list(set(modules_to_prune)),
                    verbose=1,  # 0 to disable, 1 to log overall sparsity, 2 to log per-layer sparsity
                )
            )

            print(f"Will prune {prune} of model neurons during training.")

        if swa_learning_rate:
            train_params["callbacks"].append(
                StochasticWeightAveraging(swa_lrs=swa_learning_rate)
            )

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_rank = int(os.environ.get("WORLD_RANK", 0))

        print(f"Local rank: {local_rank}, World rank: {world_rank}")

        if local_rank == 0:
            os.makedirs(output_dir, exist_ok=True)

            if progress_bar:
                train_params["callbacks"].append(AIGProgressBar(num_steps))

            if generate_every > 0:
                train_params["callbacks"].append(
                    AIGSampleGenerator(generate_every, self.get_device())
                )

            if save_every > 0:
                train_params["callbacks"].append(
                    AIGModelSaver(
                        save_every,
                        output_dir,
                        petals,
                    )
                )

        time.sleep(3)

        self.prepare_datasets(hparams, local_data, streaming_data)

        params = self._get_params(self.model, hparams)

        opt = get_optimizer(params, hparams)
        schedule = get_schedule(hparams, opt)

        if strategy is not None:
            train_params["strategy"], schedule = get_strategy(
                strategy, params, hparams, train_params, schedule
            )

        # Wrap the model in a pytorch-lightning module
        train_model = AIGTrainer(
            self.model,
            opt,
            schedule,
            self.static_len,
            hparams,
            self.tokenizer,
        )

        self.model.train()

        if verbose:
            print(self.model)

            if hasattr(self.model, "print_trainable_parameters"):
                self.model.print_trainable_parameters()

            if self.static_len > 0:
                print(
                    f"Training data:\n{colors.GREEN}{self.static_len}{colors.WHITE} static batches, {colors.GREEN}{self.static_len * block_size}{colors.WHITE} tokens"
                )

        while self.static_len > 0 and train_params["val_check_interval"] > len(
            self.total_train[0]
        ):
            train_params["val_check_interval"] = math.floor(
                len(self.total_train[0]) / 2
            )

        trainer = Trainer(**train_params)
        trainer.fit(
            train_model,
            self.combined_train,
            self.combined_val,
            ckpt_path=latest_checkpoint,
        )

        if save_every > 0 and not petals:
            self.save(output_dir)

        reset_seed()

        return trainer.callback_metrics["train_loss"].item()

    def save(self, target_folder: str = os.getcwd()):
        """Saves the model into the specified directory."""
        logger.info(f"Saving trained model to {target_folder}")
        if TPUAccelerator.is_available():
            import torch_xla.core.xla_model as xm

            self.model.save_pretrained(
                target_folder, save_function=xm.save, safe_serialization=True
            )
        else:
            self.model.save_pretrained(target_folder, safe_serialization=True)

    def get_device(self) -> str:
        """Getter for the current device where the model is located."""
        return self.model.device

    def get_total_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    # This controls the output of the aigen object, when printed to console.
    def __repr__(self) -> str:
        # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/24
        num_params_m = int(self.get_total_params() / 10**6)
        model_name = type(self.model.config).__name__.replace("Config", "")
        return f"{model_name} loaded with {num_params_m}M parameters. (precision: {self.precision})"


class CharacterSuppressionTopKLogitsWarper(LogitsWarper):
    def __init__(self, tokenizer, forbidden_chars, top_k):
        self.tokenizer = tokenizer
        self.forbidden_chars = forbidden_chars
        self.top_k_warper = TopKLogitsWarper(top_k=top_k)

    def __call__(self, input_ids, scores):
        # Apply top_k warping
        scores = self.top_k_warper(input_ids, scores)

        # Get the top k token indices
        top_k_scores, top_k_indices = torch.topk(
            scores, self.top_k_warper.top_k, dim=-1
        )

        # Iterate over the top k tokens and suppress those with forbidden characters
        for batch_idx in range(top_k_scores.size(0)):
            for i in range(top_k_scores.size(1)):
                token_idx = top_k_indices[batch_idx, i].item()
                token_str = self.tokenizer.decode(
                    [token_idx], clean_up_tokenization_spaces=False
                )
                if any(char in token_str for char in self.forbidden_chars):
                    scores[batch_idx, token_idx] = -float("inf")

        return scores
