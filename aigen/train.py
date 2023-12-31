import logging
import os
import random
import shutil
import subprocess
import sys
from math import isnan

import psutil
import torch
import torchmetrics
from lightning.pytorch import LightningModule
from lightning.pytorch.accelerators import TPUAccelerator
from lightning.pytorch.callbacks import Callback, ProgressBar
from tqdm.auto import tqdm

from .utils import colors

logging.getLogger("transformers").setLevel(logging.ERROR)


class AIGTrainer(LightningModule):
    """
    A training module for AIGen.
    """

    def __init__(self, model, optimizer, scheduler, train_len, hparams, tokenizer):
        super(AIGTrainer, self).__init__()

        self.model, self.optimizer, self.scheduler, self.train_len, self.tokenizer = (
            model,
            optimizer,
            scheduler,
            train_len,
            tokenizer,
        )
        self.total_tokens = 0
        self.block_size = hparams["block_size"]
        self.automatic_optimization = True
        self.save_hyperparameters(hparams)

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        losses = []

        schedule = self.lr_schedulers()
        step = self.global_step

        if hasattr(schedule, "current_step"):
            step = schedule.current_step

        for sample in batch:
            outputs = self({"input_ids": sample, "labels": sample})
            losses.append(outputs[0])
            self.total_tokens += int(self.block_size)

        loss = sum(losses) / len(losses)

        self.log("step", int(step), on_step=True, on_epoch=True)
        self.log("train_loss", float(loss), on_step=True, on_epoch=True)
        self.log("train_tokens", int(self.total_tokens), on_step=True, on_epoch=True)

        schedule.step()

        return loss

    def validation_step(self, batch, batch_idx):
        losses = []

        for sample in batch:
            outputs = self({"input_ids": sample, "labels": sample})
            losses.append(outputs[0])

        loss = sum(losses) / len(losses)

        self.log("val_loss", float(loss), on_step=False, on_epoch=True)
        self.log("val_ppl", float(torch.exp(loss)), on_step=False, on_epoch=True)
        self.log("hp_metric", loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        "Prepare optimizer"

        return [self.optimizer], [self.scheduler]


class AIGProgressBar(ProgressBar):
    """A variant progress bar that works off of steps and prints periodically."""

    def __init__(self):
        super().__init__()
        self.last_step = 0
        self.prev_avg_loss = None
        self.smoothing = 0.01
        self.is_synced = False
        try:
            from IPython.display import display

            self.is_notebook = True
        except ImportError:
            self.is_notebook = False

        if self.is_notebook:
            self.blue = ""
            self.red = ""
            self.green = ""
            self.white = ""
        else:
            self.blue = colors.BLUE
            self.red = colors.RED
            self.green = colors.GREEN
            self.white = colors.WHITE

    def on_train_start(self, trainer, lm):
        super().on_train_start(trainer, lm)
        self.pbar = tqdm(
            total=trainer.estimated_stepping_batches,
            smoothing=0,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def on_train_end(self, trainer, lm):
        self.pbar.close()

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        current_loss = float(trainer.callback_metrics["train_loss"])

        current_epoch = trainer.current_epoch
        if lm.train_len > 0:
            current_epoch += batch_idx / lm.train_len

        avg_loss = 0
        if not isnan(current_loss):
            avg_loss = self.average_loss(
                current_loss, self.prev_avg_loss, self.smoothing
            )
            self.prev_avg_loss = avg_loss

        color = self.green
        if current_loss < avg_loss:
            color = self.blue
        elif current_loss > avg_loss:
            color = self.red

        bearing = "{:.5f}".format(
            abs(round((current_loss / avg_loss) if avg_loss != 0 else 0, 5))
        )

        c_sym = "+" if current_loss >= 0 else ""
        a_sym = "+" if avg_loss >= 0 else ""

        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
            "SC_PHYS_PAGES"
        )  # e.g. 4015976448
        mem_gib = mem_bytes / (1024.0**3)  # e.g. 3.74

        memory = psutil.virtual_memory()

        bar = f"{self.green}{c_sym}{current_loss:.3f}{self.white} => Loss => {color}{a_sym}{avg_loss:.3f}{self.white} => Bearing => {self.blue}{bearing}{random.randint(0,2)}00{self.white} => System => {self.blue}{memory.percent}%{self.white}"

        if lm.on_gpu:
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
            gpus = f"MB{self.white} => {self.blue}".join(gpu_memory)
            epoch_string = "{:.3f}".format(current_epoch)
            bar += f" => GPU => {self.blue}{gpus}MB{self.white}"

        if current_epoch > 0:
            bar += f" => Epoch => {self.blue}{epoch_string}{self.white}"

        if hasattr(trainer.strategy, "num_peers"):
            num_peers = trainer.strategy.num_peers
            bar += f" => Peers => {self.blue}{num_peers}{self.white}"

        step = int(trainer.callback_metrics["step"])

        if step != 0 and not self.is_synced:
            # If training resumes from a checkpoint, set progress bar to the correct step.
            self.pbar.update(step)

        self.is_synced = True

        if step != 0 and step != self.last_step:
            self.pbar.update(1)
            self.last_step = step

        self.pbar.set_description(bar)

    def on_validation_start(self, trainer, lm):
        super().on_validation_start(trainer, lm)

        if trainer.state.stage in ["sanity_check"]:
            return

        logging.warning("Calculating validation metrics...")

    def average_loss(self, current_loss, prev_avg_loss, smoothing):
        if prev_avg_loss is None:
            return current_loss
        else:
            return (smoothing * current_loss) + (1 - smoothing) * prev_avg_loss


class AIGModelSaver(Callback):
    """Periodically model during training."""

    def __init__(
        self,
        save_every,
        output_dir,
        petals,
    ):
        super().__init__()
        self.steps = 0
        self.save_every = save_every
        self.output_dir = output_dir
        self.petals = petals

    @property
    def save_every_check(self):
        return self.save_every > 0 and self.steps % self.save_every == 0

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        self.steps += 1

        if TPUAccelerator.is_available() and self.save_every_check:
            self.save_pytorch_model(trainer, lm, tpu=True)

        if not TPUAccelerator.is_available() and self.save_every_check:
            self.save_pytorch_model(trainer, lm)

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

            lm.model.save_pretrained(
                self.output_dir, save_function=xm.save, safe_serialization=True
            )
        else:
            lm.model.save_pretrained(self.output_dir, safe_serialization=True)


class AIGSampleGenerator(Callback):
    """Periodically print samples to the console."""

    def __init__(self, generate_every):
        super().__init__()
        from transformers import GenerationConfig

        self.steps = 0
        self.generate_every = generate_every
        self.generation_config = GenerationConfig(
            do_sample=True,
            min_length=1,
            max_new_tokens=222,
            temperature=0.9,
            eta_cutoff=0.0003,
            penalty_alpha=0.6,
            top_k=4,
            repetition_penalty=1.023,
            no_repeat_ngram_size=13,
        )

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        self.steps += 1

        if self.generate_every > 0 and self.steps % self.generate_every == 0:
            self.generate_sample_text(trainer, lm)

    def generate_sample_text(self, trainer, lm):
        lm.model.eval()

        if hasattr(lm.model, "training"):
            lm.model.training = False

        outputs = lm.model.generate(
            inputs=None,
            generation_config=self.generation_config,
            do_sample=True,
            max_new_tokens=222,
            bos_token_id=lm.tokenizer.bos_token_id,
            eos_token_id=lm.tokenizer.eos_token_id,
            pad_token_id=lm.tokenizer.pad_token_id,
        )

        if hasattr(lm.model, "training"):
            lm.model.training = True

        output = lm.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        lm.model.train()

        print(output[0])


class AIGMetricsLogger(Callback):
    """Save metrics callback."""

    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        if not lm.logger:
            return

        current_epoch = trainer.current_epoch
        if lm.train_len > 0:
            current_epoch += batch_idx / lm.train_len

        step = trainer.callback_metrics["step"]

        metrics = {
            "train_loss": trainer.callback_metrics["train_loss"],
            "lr": float(trainer.optimizers[0].param_groups[0]["lr"]),
            "train_tokens": int(trainer.callback_metrics["train_tokens"]),
        }

        if current_epoch > 0:
            metrics["epoch"] = current_epoch

        lm.logger.experiment.add_scalars(
            "vtx",
            metrics,
            step,
        )

    def on_validation_epoch_end(self, trainer, lm):
        super().on_validation_epoch_end(trainer, lm)

        if not lm.logger or trainer.state.stage in ["sanity_check"]:
            return

        step = trainer.callback_metrics.get("step", 0)

        if step == 0:
            return

        lm.logger.experiment.add_scalars(
            "vtx",
            {
                "val_loss": trainer.callback_metrics["val_loss"],
                "val_ppl": trainer.callback_metrics["val_ppl"],
            },
            step,
        )
