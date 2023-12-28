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
from packaging import version
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .aigen import aigen


class Objective:
    def __init__(self, init_kwargs, train_config):
        self.prototype = aigen(**init_kwargs)

        self.train_config = train_config

        self.train_config["num_steps"] = 100

        self.max_batch_size = (
            self.train_config.get("batch_size", 1)
            * self.train_config.get("gradient_accumulation_steps", 1)
            * self.train_config.get("target_batch_size", 1)
        )

    def __call__(self, trial: optuna.trial.Trial):
        self.train_config["optimizer"] = trial.suggest_categorical(
            "optimizer", ["AdamW", "Lion"]
        )
        self.train_config["warmup_steps"] = trial.suggest_int("warmup_steps", 0, 10)
        self.train_config["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.0001, 0.1, log=True
        )
        self.train_config["batch_size"] = 1
        self.train_config["gradient_accumulation_steps"] = trial.suggest_int(
            "gradient_accumulation_steps", 2, self.max_batch_size
        )
        self.train_config["weight_decay"] = trial.suggest_float(
            "weight_decay", 0.0001, 0.1, log=True
        )
        self.train_config["dropout"] = trial.suggest_float("dropout", 0, 0.5)

        pprint(self.train_config)

        log_path = "/data/logs/src"
        os.makedirs(f"{log_path}/tune", exist_ok=True)
        logger = loggers.TensorBoardLogger(
            log_path, name="tune", default_hp_metric=True
        )

        train_type = self.train_config.get("type", "standard")
        if train_type in ["lora"]:
            self.train_config["r"] = trial.suggest_int("r", 1, 64)
            self.train_config["alpha"] = trial.suggest_int("alpha", 1, 32)
            self.train_config["bias"] = trial.suggest_categorical(
                "bias", ["none", "all", "lora_only"]
            )

        if train_type not in ["standard", "pretrain"]:
            self.prototype.create_adapter(self.train_config)

        train_loss = self.prototype.train(
            loggers=[logger],
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="train_loss")],
            **self.train_config,
        )

        return train_loss


def optimize_hparams(init_kwargs, train_config):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )

    study.optimize(Objective(init_kwargs, train_config), n_trials=100, timeout=10800)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
