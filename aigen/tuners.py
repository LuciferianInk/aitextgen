import argparse
import os
from pprint import pprint
from typing import List, Optional

import lightning.pytorch as pl
import optuna
import torch
import torch.nn.functional as F
from lightning.pytorch import loggers
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import PatientPruner, SuccessiveHalvingPruner
from packaging import version
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .aigen import aigen
from .utils import colors


def objective(trial: optuna.trial.Trial, init_kwargs, train_config):
    train_config["num_steps"] = 50

    # max_gradient_accumulation_steps = (
    #     train_config.get("batch_size", 1)
    #     * train_config.get("gradient_accumulation_steps", 1)
    #     * train_config.get("target_batch_size", 1)
    # )
    max_gradient_accumulation_steps = 1024
    min_gradient_accumulation_steps = 128

    train_config["batch_size"] = 1

    train_type = train_config.get("type", "standard")

    if train_type in ["pretrain"]:
        setattr(init_kwargs["config"], "n_head", trial.suggest_int("n_head", 1, 2))
        setattr(init_kwargs["config"], "k_att", trial.suggest_int("k_att", 2, 3))
        setattr(init_kwargs["config"], "k_mlp", trial.suggest_int("k_mlp", 2, 3))
        setattr(
            init_kwargs["config"],
            "sample_topk",
            trial.suggest_int("sample_topk", 1, 2),
        )
        setattr(
            init_kwargs["config"],
            "gate_type",
            trial.suggest_categorical("gate_type", ["mlp", "gmm"]),
        )
        block_size = trial.suggest_int("block_size", 64, 256, step=64)
        setattr(
            init_kwargs["config"],
            "block_size",
            block_size,
        )
        setattr(
            init_kwargs["config"],
            "history_length",
            block_size,
        )
        setattr(
            init_kwargs["config"],
            "gating_size",
            int(block_size / 2),
        )
        setattr(
            init_kwargs["config"],
            "aux_loss_weight",
            trial.suggest_float("aux_loss_weight", 0.00001, 0.1),
        )
        setattr(
            init_kwargs["config"],
            "resid_pdrop",
            trial.suggest_float("resid_pdrop", 0, 0.9),
        )
        setattr(
            init_kwargs["config"],
            "embd_pdrop",
            trial.suggest_float("embd_pdrop", 0, 0.9),
        )
        setattr(
            init_kwargs["config"],
            "attn_pdrop",
            trial.suggest_float(
                "attn_pdrop",
                0,
                0.9,
            ),
        )
        setattr(
            init_kwargs["config"],
            "moe_pdrop",
            trial.suggest_float("moe_pdrop", 0, 0.9),
        )

    train_config["optimizer"] = trial.suggest_categorical(
        "optimizer", ["AdamW", "Lion"]
    )
    train_config["warmup_steps"] = trial.suggest_int("warmup_steps", 0, 10)
    train_config["learning_rate"] = trial.suggest_float(
        "learning_rate", 0.00001, 0.1, log=True
    )
    train_config["gradient_accumulation_steps"] = trial.suggest_int(
        "gradient_accumulation_steps",
        min_gradient_accumulation_steps,
        max_gradient_accumulation_steps,
        step=min_gradient_accumulation_steps,
    )
    train_config["weight_decay"] = trial.suggest_float(
        "weight_decay", 0.00001, 0.1, log=True
    )

    if train_type in ["lora"]:
        train_config["r"] = trial.suggest_int("r", 1, 64)
        train_config["alpha"] = trial.suggest_int("alpha", 1, 32)
        train_config["dropout"] = trial.suggest_float("dropout", 0, 0.5)
        train_config["bias"] = trial.suggest_categorical(
            "bias", ["none", "all", "lora_only"]
        )

    prototype = aigen(**init_kwargs)

    if train_type not in ["standard", "pretrain"]:
        prototype.create_adapter(train_config)

    os.makedirs(f"{train_config['log_path']}/tune", exist_ok=True)
    logger = loggers.TensorBoardLogger(
        train_config["log_path"], name="tune", default_hp_metric=True
    )

    train_loss = prototype.train(
        loggers=[logger],
        callbacks=[CustomPruningCallback(trial, monitor="train_loss")],
        trial=True,
        verbose=False,
        **train_config,
    )

    return train_loss


_EPOCH_KEY = "ddp_pl:epoch"
_INTERMEDIATE_VALUE = "ddp_pl:intermediate_value"
_PRUNED_KEY = "ddp_pl:pruned"


class CustomPruningCallback(PyTorchLightningPruningCallback):
    """Check pruning on training step."""

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__(trial, monitor)

        self.current_step = 0

    def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, lm, outputs, batch, batch_idx)

        if trainer.sanity_checking:
            return

        step = int(trainer.callback_metrics["step"])

        if self.current_step == step:
            return

        self.current_step = step

        current_score = trainer.callback_metrics.get(self.monitor)

        should_stop = False

        # Determine if the trial should be terminated in a single process.
        if not self.is_ddp_backend:
            self._trial.report(current_score.item(), step=step)
            if not self._trial.should_prune():
                return
            raise optuna.TrialPruned(f"Trial was pruned at step {step}.")

        # Determine if the trial should be terminated in a DDP.
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=step)
            should_stop = self._trial.should_prune()

            # Update intermediate value in the storage.
            _trial_id = self._trial._trial_id
            _study = self._trial.study
            _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
            intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)
            intermediate_values[step] = current_score.item()  # type: ignore[index]
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _INTERMEDIATE_VALUE, intermediate_values
            )

        # Terminate every process if any world process decides to stop.
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if not should_stop:
            return

        if trainer.is_global_zero:
            # Update system_attr from global zero process.
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _PRUNED_KEY, True
            )
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, _EPOCH_KEY, step
            )


def optimize_hparams(init_kwargs, train_config):
    storage = optuna.storages.RDBStorage(
        url="sqlite:///trials.db",
        # url="sqlite:///:memory:",
        engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
    )

    study = optuna.create_study(
        study_name="trial-1",
        direction="minimize",
        storage=storage,
        sampler=optuna.samplers.TPESampler(),
        pruner=PatientPruner(
            SuccessiveHalvingPruner(
                min_resource="auto",
                reduction_factor=4,
                min_early_stopping_rate=0,
                bootstrap_count=0,
            ),
            patience=3,
            min_delta=0.1,
        ),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial, init_kwargs=init_kwargs, train_config=train_config
        ),
        n_trials=100,
        timeout=60 * 60 * 24,
    )

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
