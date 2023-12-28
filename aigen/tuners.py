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

        if train_config.get("type") not in ["standard", "pretrain"]:
            self.prototype.create_adapter(train_config)

        self.train_config = train_config

        self.train_config["num_steps"] = 1000
        self.train_config["warmup_steps"] = 10

        self.max_batch_size = (
            self.train_config.get("batch_size", 1)
            * self.train_config.get("gradient_accumulation_steps", 1)
            * self.train_config.get("target_batch_size", 1)
        )

    def __call__(self, trial: optuna.trial.Trial):
        self.train_config["warmup_steps"] = trial.suggest_int("warmup_steps", 0, 100)
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

        pprint(self.train_config)

        log_path = "/data/logs/src"
        os.makedirs(f"{log_path}/tune", exist_ok=True)
        logger = loggers.TensorBoardLogger(
            log_path, name="tune", default_hp_metric=True
        )

        # hyperparameters = dict(self.train_config)
        # logger.log_hyperparams(hyperparameters)

        train_loss = self.prototype.train(
            loggers=[logger],
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="train_loss")],
            **self.train_config,
        )

        return train_loss


def optimize_hparams(init_kwargs, train_config):
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(Objective(init_kwargs, train_config), n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
