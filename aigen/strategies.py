import json
import logging
import os
import random
import re
import subprocess
import time
from functools import partial

from lightning.pytorch.callbacks import Callback

from .utils import colors


def get_strategy(name, params, hparams, train_params, scheduler):
    if name == "ddp":
        return "ddp"
    elif name == "deepspeed":
        DeepSpeedStrategy(
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

        # bootstrap_peers = [
        #     "/p2p/12D3KooWJb6YNtfYpvfL2C7cKfMqHorLFTcPAouY5yHU73R8UhZy",  # 59.src.eco
        #     "/p2p/12D3KooWE6YAK8nte7Wky13WDMwxSmfgRecPystnxxcp793trVd2",  # 95.src.eco
        # ]

        pattern = r"(/p2p/.*)"

        # Start with bootstrap peers
        initial_piers = hparams.get("initial_piers", [])
        # initial_piers.append(random.choice(bootstrap_peers))

        # # Get my local peers
        # command = "docker exec vtx-fil-1 ipfs swarm peers"
        # process = subprocess.Popen(
        #     command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        # )
        # output, error = process.communicate()

        # peers = output.decode("utf-8").splitlines()

        # for peer in peers:
        #     match = re.search(pattern, peer)
        #     if match:
        #         initial_piers.append(match.group(1))

        # # Get my own peer ID
        # command = "docker exec vtx-fil-1 ipfs id"
        # process = subprocess.Popen(
        #     command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        # )
        # output, error = process.communicate()

        # try:
        #     mine = json.loads(output.decode("utf-8"))
        #     craft = f"/p2p/{mine['ID']}"
        #     initial_piers.append(craft)
        # except:
        #     pass

        delay = 1.0
        for peer in initial_piers:
            time.sleep(delay)
            delay *= 0.75
            print(f"PIER-{initial_piers.index(peer)}: {peer}")

        class MaxStepCallback(Callback):
            def __init__(self, max_steps):
                self.max_steps = max_steps

            def on_train_batch_end(self, trainer, lm, outputs, batch, batch_idx):
                schedule = lm.lr_schedulers()
                # trainer.strategy.barrier()
                if schedule.current_step >= self.max_steps:
                    print(f"Reached max_steps ({self.max_steps}). Stopping training.")
                    trainer.should_stop = True
                    trainer.strategy.teardown()

        train_params["callbacks"].append(
            MaxStepCallback(max_steps=train_params["max_steps"])
        )

        train_params["max_steps"] *= hparams["target_batch_size"]
        train_params["val_check_interval"] *= hparams["target_batch_size"]

        focus = os.environ["FOCUS"]

        strategy = HivemindStrategy(
            run_id=f"src-vtx-{focus}",
            batch_size=hparams["batch_size"],
            target_batch_size=hparams["target_batch_size"],
            initial_piers=initial_piers,
            use_ipfs=True,
            use_relay=True,
            use_auto_relay=True,
            verbose=False,
            wait_timeout=90,
            bootstrap_timeout=30,
            matchmaking_time=90.0,
            averaging_timeout=300.0,
            # delay_state_averaging=True,
            # delay_grad_averaging=True,
            # delay_optimizer_step=True,
            # offload_optimizer=True,  # required to delay averaging
            # scheduler_fn=partial(
            #     AdamW,
            #     # params,
            #     lr=hparams["learning_rate"],
            #     eps=hparams.get("eps", 1e-8),
            # ),
        )

        visible_addresses = [
            str(a)
            for a in strategy.dht.get_visible_maddrs()
            if not ipaddress.ip_address(a.values()[0]).is_loopback
        ]

        my_ids = []
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
        strategy = "auto"

    return strategy
