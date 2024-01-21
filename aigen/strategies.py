import json
import logging
import os
import random
import re
import subprocess
import time
from functools import partial

import torch
from lightning.pytorch.callbacks import Callback

from .utils import colors


def get_strategy(name, params, hparams, train_params, schedule):
    if name == "deepspeed":
        strategy = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            allgather_bucket_size=2e8,
            reduce_bucket_size=2e8,
        )
    elif name == "hivemind":
        import ipaddress

        from lightning_hivemind.strategy import HivemindStrategy

        assert (
            train_params["accumulate_grad_batches"] == 1
        ), "Gradient accumulation is not supported by HivemindStrategy. Use `target_batch_size` instead."

        # Start with bootstrap peers
        # bootstrap_piers = [
        #     "/p2p/QmVtpsm7b91S5pYcjsreHKxoHU6wThBn6RFHPrUWXCBrzo",  # src.eco
        #     "/p2p/QmVQE44X5wPo5LNheJCBMVRUTRsceJNxVowjxerPUCCZmY",  # src.eco
        # ]

        # initial_piers = hparams.get("initial_piers", []) + bootstrap_piers
        initial_piers = hparams.get("initial_piers", [])

        delay = 1.0
        for peer in initial_piers:
            time.sleep(delay)
            delay *= 0.75
            print(f"PIER-{initial_piers.index(peer)}: {peer}")

        class MaxStepCallback(Callback):
            def __init__(self, max_steps):
                self.max_steps = max_steps

            def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
                current_step = trainer.strategy.optimizers[0].local_epoch
                if current_step >= self.max_steps:
                    print(f"Reached max_steps ({self.max_steps}). Stopping training.")
                    trainer.should_stop = True
                    trainer.strategy.teardown()

        train_params["callbacks"].append(
            MaxStepCallback(max_steps=train_params["max_steps"])
        )

        train_params["max_steps"] *= hparams["target_batch_size"]
        train_params["val_check_interval"] *= hparams["target_batch_size"]

        focus = os.environ["FOCUS"]

        pet = random.choice(["cat", "dog", "fox", "jackal", "lemur"])

        schedule = False
        strategy = HivemindStrategy(
            run_id=f"vtx-{focus}",
            identity_path=f"/data/identity.{pet}.key",
            batch_size=hparams["batch_size"],
            target_batch_size=hparams["target_batch_size"],
            initial_peers=initial_piers,
            use_ipfs=True,
            use_relay=True,
            use_auto_relay=True,
            verbose=False,
            wait_timeout=30,
            bootstrap_timeout=20,
            matchmaking_time=45.0,
            averaging_timeout=180.0,
            delay_state_averaging=True,
            delay_grad_averaging=True,
            delay_optimizer_step=True,
            offload_optimizer=True,
            reuse_grad_buffers=False,
            scheduler_fn=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9999),
        )

        visible_addresses = [
            str(a)
            for a in strategy.dht.get_visible_maddrs()
            if not ipaddress.ip_address(a.values()[0]).is_loopback
        ]

        my_ids = []
        pattern = r"(/p2p/.*)"
        for peer in list(visible_addresses):
            match = re.search(pattern, peer)
            if match:
                my_ids.append(match.group(1))

        print(
            f"{colors.BLUE}ONE@SWARM:{colors.WHITE} To join this swarm, use the following `initial_piers`:"
        )

        for peer in list(set(my_ids)):
            print(
                f"{colors.GREEN}PIER-{len(initial_piers) + list(set(my_ids)).index(peer)}:{colors.WHITE} {peer}"
            )
    else:
        strategy = name

    return strategy, schedule
