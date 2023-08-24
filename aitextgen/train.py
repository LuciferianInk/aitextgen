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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar
from pytorch_lightning.accelerators import TPUAccelerator
import random
from .utils import bc, ad


class ATGTransformer(pl.LightningModule):
    """
    A training module for aitextgen.
    """

    def __init__(self, model, dataset, hparams, tokenizer):
        super(ATGTransformer, self).__init__()
        self.model, self.dataset, self.tokenizer = (
            model,
            dataset,
            tokenizer,
        )
        self.save_hyperparameters(hparams)
        self.automatic_optimization = True

    def forward(self, inputs):
        return self.model(**inputs, return_dict=False)

    def training_step(self, batch, batch_num):
        outputs = self({"input_ids": batch, "labels": batch})
        loss = outputs[0]

        schedule = self.lr_schedulers()
        schedule.step()

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


class ATGProgressBar(ProgressBar):
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
        prompt
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
        self.prompt = prompt

    @property
    def save_every_check(self):
        return self.save_every > 0 and self.steps % self.save_every == 0

    def enabled(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.main_progress_bar = tqdm(
            total=trainer.max_steps,
            disable=not self.enabled,
            smoothing=0,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        self.freeze_layers(pl_module)

    def on_train_end(self, trainer, pl_module):
        self.main_progress_bar.close()
        self.unfreeze_layers(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

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
                self.unfreeze_layers(pl_module)
                did_unfreeze = True
            self.save_pytorch_model(trainer, pl_module, tpu=True)
            if did_unfreeze:
                self.freeze_layers(pl_module)

        if self.enabled:
            did_unfreeze = False
            if not TPUAccelerator.is_available() and self.save_every_check:
                self.unfreeze_layers(pl_module)
                self.save_pytorch_model(trainer, pl_module)
                did_unfreeze = True

            if self.generate_every > 0 and self.steps % self.generate_every == 0:
                self.unfreeze_layers(pl_module)
                self.generate_sample_text(trainer, pl_module)
                did_unfreeze = True

            if did_unfreeze:
                self.freeze_layers(pl_module)

        pl_module.logger.experiment.add_scalars(
            "loss/" + str(pl_module.hparams["stage"]),
            {"train": current_loss},
            pl_module.global_step,
        )

        color = bc.ROOT
        if current_loss < avg_loss:
            color = bc.FOLD
        elif current_loss > avg_loss:
            color = bc.CORE

        bearing = "{:.8f}".format(round(current_loss / avg_loss, 8))

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
            self.main_progress_bar.update(self.progress_bar_refresh_rate)
            self.main_progress_bar.set_description(echo)

    def generate_sample_text(self, trainer, pl_module):

        pl_module.model.eval()

        pad_token_id = getattr(pl_module.tokenizer, "pad_token_id", None) or getattr(
            pl_module.tokenizer, "eos_token_id", None
        )

        prompt = self.prompt
        if prompt:
            prompt_tensors = pl_module.tokenizer(text=prompt, return_tensors="pt")
            input_ids = prompt_tensors["input_ids"].to(pl_module.model.device.type)
        else:
            input_ids = None

        config = GenerationConfig(
            n=self.n_generate,
            do_sample=True,
            temperature=0.7,
            pad_token_id=pad_token_id
        )

        logging.getLogger("transformers").setLevel(logging.ERROR)

        if self.petals:
            outputs = pl_module.model.generate(
                max_length=333,
                do_sample=True,
                num_return_sequences=self.n_generate,
                temperature=0.7,
                pad_token_id=pad_token_id
            )
        else:
            outputs = pl_module.model.generate(
                input_ids=input_ids,
                max_new_tokens=111,
                generation_config=config,
                return_as_list=True,
            )

        gen_texts = pl_module.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        logging.getLogger("transformers").setLevel(logging.WARNING)

        pl_module.model.train()

        for text in gen_texts:
            self.main_progress_bar.write(f"{bc.CORE}<={ad.TEXT}=")
            self.main_progress_bar.write(text)

        color = bc.FOLD
        if random.choice(["blue", "green"]) == "green":
            color = bc.ROOT

        self.main_progress_bar.write(f"={color}=>{ad.TEXT}")

    def save_pytorch_model(self, trainer, pl_module, tpu=False):
        if tpu:
            import torch_xla.core.xla_model as xm

            pl_module.model.save_pretrained(self.output_dir, save_function=xm.save)
        else:
            pl_module.model.save_pretrained(self.output_dir)

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

    def modify_layers(self, pl_module, unfreeze):
        if self.train_transformers_only:
            for name, param in pl_module.model.named_parameters():
                if self.num_layers_freeze:
                    layer_num = int(name.split(".")[2]) if ".h." in name else None
                    to_freeze = layer_num and layer_num < self.num_layers_freeze
                else:
                    to_freeze = False
                if name == "transformer.wte.weight" or to_freeze:
                    param.requires_grad = unfreeze

    def freeze_layers(self, pl_module):
        self.modify_layers(pl_module, False)

    def unfreeze_layers(self, pl_module):
        self.modify_layers(pl_module, True)
