import os
import psutil
import shutil
import subprocess
import sys
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler
import logging
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ProgressBar
from lightning.pytorch.accelerators import TPUAccelerator
import random
from .utils import bc, ad

logging.getLogger("transformers").setLevel(logging.ERROR)


class AIGTrainer(LightningModule):
    """
    A training module for aigen.
    """

    def __init__(self, model, dataset, hparams, tokenizer):
        super(AIGTrainer, self).__init__()
        self.model, self.dataset, self.tokenizer = (
            model,
            dataset,
            tokenizer,
        )
        self.save_hyperparameters(hparams)
        if self.hparams["optimizer"] in ["SophiaH"]:
            self.automatic_optimization = False
        else:
            self.automatic_optimization = True

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        outputs = self({"input_ids": batch, "labels": batch})
        loss = outputs[0]

        if self.hparams["optimizer"] in ["SophiaH"]:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss, create_graph=True)
        else:
            opt = self.lr_schedulers()

        opt.step()

        return {"loss": loss}

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            pin_memory=self.hparams["pin_memory"],
            num_workers=self.hparams["num_workers"],
        )

    def configure_optimizers(self):
        "Prepare optimizer"

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams["optimizer"] in ["SophiaH"]:
            try:
                from pytorch_optimizer import SophiaH

            except ImportError:
                print("Failed to import SophiaH optimizer. Is it installed?")

            optimizer = SophiaH(
                optimizer_grouped_parameters,
                lr=self.hparams["learning_rate"],
                update_period=self.hparams["update_period"],
            )
        elif self.hparams["optimizer"] in ["Lion"]:
            try:
                from pytorch_optimizer import Lion

            except ImportError:
                print("Failed to import Lion optimizer. Is it installed?")

            optimizer = Lion(
                optimizer_grouped_parameters,
                lr=self.hparams["learning_rate"],
                use_gc=True,
                adanorm=True,
            )
        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams["learning_rate"],
                eps=self.hparams["adam_epsilon"],
            )

        scheduler = get_scheduler(
            self.hparams.get("scheduler", "linear"),
            optimizer,
            num_warmup_steps=self.hparams.get("warmup_steps", 0),
            num_training_steps=self.hparams["num_steps"],
        )

        return [optimizer], [scheduler]


class AIGProgressBar(ProgressBar):
    """A variant progress bar that works off of steps and prints periodically."""

    def __init__(
        self,
        save_every,
        generate_every,
        output_dir,
        n_generate,
        gpu,
        smoothing,
        run_id,
        save_gdrive,
        progress_bar_refresh_rate,
        train_transformers_only,
        num_layers_freeze,
        petals,
        hivemind,
        prompt,
    ):
        super().__init__()
        self.enabled = True
        self.save_every = save_every
        self.generate_every = generate_every
        self.output_dir = output_dir
        self.n_generate = n_generate
        self.gpu = gpu
        self.steps = 0
        self.prev_avg_loss = None
        self.smoothing = smoothing
        self.run_id = run_id
        self.save_gdrive = save_gdrive
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.train_transformers_only = train_transformers_only
        self.num_layers_freeze = num_layers_freeze
        self.petals = petals
        self.hivemind = hivemind
        self.prompt = prompt

    @property
    def save_every_check(self):
        return self.save_every > 0 and self.steps % self.save_every == 0

    def enabled(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def on_train_start(self, trainer, lm):
        super().on_train_start(trainer, lm)
        self.main_progress_bar = tqdm(
            total=trainer.max_steps,
            disable=not self.enabled,
            smoothing=0,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        self.freeze_layers(lm)

    def on_train_end(self, trainer, lm):
        self.main_progress_bar.close()
        self.unfreeze_layers(lm)

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        # clean up the GPU cache used for the benchmark
        # https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/4
        if self.steps == 0 and self.gpu:
            torch.cuda.empty_cache()

        current_loss = float(outputs["loss"])
        self.steps += 1
        avg_loss = 0
        if current_loss == current_loss:  # don't add if current_loss is NaN
            avg_loss = self.average_loss(
                current_loss, self.prev_avg_loss, self.smoothing
            )
            self.prev_avg_loss = avg_loss

        if TPUAccelerator.is_available() and self.save_every_check:
            did_unfreeze = False
            if self.enabled:
                self.unfreeze_layers(lm)
                did_unfreeze = True
            self.save_pytorch_model(trainer, lm, tpu=True)
            if did_unfreeze:
                self.freeze_layers(lm)

        if self.enabled:
            did_unfreeze = False
            if not TPUAccelerator.is_available() and self.save_every_check:
                self.unfreeze_layers(lm)
                self.save_pytorch_model(trainer, lm)
                did_unfreeze = True

            if self.generate_every > 0 and self.steps % self.generate_every == 0:
                self.unfreeze_layers(lm)
                self.generate_sample_text(trainer, lm)
                did_unfreeze = True

            if did_unfreeze:
                self.freeze_layers(lm)

        lm.logger.experiment.add_scalars(
            "loss/" + str(lm.hparams["stage"]),
            {"train": current_loss},
            lm.global_step,
        )

        color = bc.ROOT
        if current_loss < avg_loss:
            color = bc.FOLD
        elif current_loss > avg_loss:
            color = bc.CORE

        bearing = "{:.5f}".format(
            round((current_loss / avg_loss) if avg_loss != 0 else 0, 5)
        )

        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
            "SC_PHYS_PAGES"
        )  # e.g. 4015976448
        mem_gib = mem_bytes / (1024.0**3)  # e.g. 3.74

        memory = psutil.virtual_memory()

        echo = f"{bc.ROOT}{current_loss:.3f}{ad.TEXT} => Loss => {color}{avg_loss:.3f}{ad.TEXT} => Bearing => {bc.FOLD}{bearing}{random.randint(0,2)}00{ad.TEXT} => System => {bc.FOLD}{memory.percent}%{ad.TEXT}"

        if self.steps % self.progress_bar_refresh_rate == 0:
            if self.gpu:
                # via pytorch-lightning's get_gpu_memory_map()
                result = subprocess.run(
                    [
                        shutil.which("nvidia-smi"),
                        "--query-gpu=memory.used",
                        "--format=csv,nounits,noheader",
                    ],
                    encoding="utf-8",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                gpu_memory = result.stdout.strip().split(os.linesep)
                cat = f"MB{ad.TEXT} => {bc.FOLD}".join(gpu_memory)
                echo += f" => GPU => {bc.FOLD}{cat}MB{ad.TEXT}"

            if self.hivemind:
                num_peers = trainer.strategy.num_peers
                echo = echo + f" => Peers => {bc.FOLD}{str(num_peers)}{ad.TEXT}"

            self.main_progress_bar.update(self.progress_bar_refresh_rate)
            self.main_progress_bar.set_description(echo)

    def generate_sample_text(self, trainer, lm):
        lm.model.eval()

        eos_token_id = getattr(lm.tokenizer, "eos_token_id", None)
        pad_token_id = getattr(lm.tokenizer, "pad_token_id", None) or eos_token_id

        prompt = self.prompt
        if prompt:
            prompt_tensors = lm.tokenizer(text=prompt, return_tensors="pt")
            input_ids = prompt_tensors["input_ids"].to(lm.model.device.type)
        else:
            input_ids = None

        outputs = lm.model.generate(
            inputs=input_ids,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=222,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        gen_texts = lm.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        lm.model.train()

        for text in gen_texts:
            self.main_progress_bar.write(text)
            self.main_progress_bar.write(f"={bc.FOLD}=>{ad.TEXT}")

    def save_pytorch_model(self, trainer, lm, tpu=False):
        if self.petals:
            with open(os.path.join(self.output_dir, "prompts.pt"), "wb") as f:
                torch.save(
                    (
                        lm.model.transformer.prompt_embeddings,
                        lm.model.transformer.intermediate_prompt_embeddings,
                    ),
                    f,
                )
        elif tpu:
            import torch_xla.core.xla_model as xm

            lm.model.save_pretrained(self.output_dir, save_function=xm.save)
        else:
            lm.model.save_pretrained(self.output_dir)

        if self.enabled and self.save_gdrive:
            for pt_file in ["pytorch_model.bin", "config.json"]:
                shutil.copyfile(
                    os.path.join(self.output_dir, pt_file),
                    os.path.join("/content/drive/MyDrive/", self.run_id, pt_file),
                )

    def average_loss(self, current_loss, prev_avg_loss, smoothing):
        if prev_avg_loss is None:
            return current_loss
        else:
            return (smoothing * current_loss) + (1 - smoothing) * prev_avg_loss

    def modify_layers(self, lm, unfreeze):
        if self.train_transformers_only:
            for name, param in lm.model.named_parameters():
                if self.num_layers_freeze:
                    layer_num = int(name.split(".")[2]) if ".h." in name else None
                    to_freeze = layer_num and layer_num < self.num_layers_freeze
                else:
                    to_freeze = False
                if name == "transformer.wte.weight" or to_freeze:
                    param.requires_grad = unfreeze

    def freeze_layers(self, lm):
        self.modify_layers(lm, False)

    def unfreeze_layers(self, lm):
        self.modify_layers(lm, True)
