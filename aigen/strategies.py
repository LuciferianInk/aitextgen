import json
import re
import subprocess
import time

from .utils import colors


def get_strategy(scheduler, name, hparams, train_params):
    if name == "deepspeed":
        DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            allgather_bucket_size=2e8,
            reduce_bucket_size=2e8,
        )
    elif name == "hivemind":
        from lightning_hivemind.strategy import HivemindStrategy

        assert (
            train_params["accumulate_grad_batches"] == 1
        ), "Gradient accumulation is not supported by HivemindStrategy. Use `target_batch_size` instead."

        # Start with bootstrap peers
        initial_peers = [
            "/p2p/12D3KooWJb6YNtfYpvfL2C7cKfMqHorLFTcPAouY5yHU73R8UhZy",  # 59.src.eco
            "/p2p/12D3KooWE6YAK8nte7Wky13WDMwxSmfgRecPystnxxcp793trVd2",  # 95.src.eco
        ]

        # Get my local peers
        command = "docker exec vtx-fil-1 ipfs swarm peers"
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, error = process.communicate()

        peers = output.decode("utf-8").splitlines()

        pattern = r"(/p2p/.*)"

        for peer in peers:
            match = re.search(pattern, peer)
            if match:
                initial_peers.append(match.group(1))

        # Get my own peer ID
        command = "docker exec vtx-fil-1 ipfs id"
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, error = process.communicate()

        mine = json.loads(output.decode("utf-8"))
        craft = f"/p2p/{mine['ID']}"
        initial_peers.append(craft)

        delay = 1.0
        for peer in initial_peers:
            time.sleep(delay)
            delay *= 0.75
            color = colors.WHITE
            if initial_peers.index(peer) == (len(initial_peers) - 1):
                color = colors.GREEN
            print(f"{color}PIER-{initial_peers.index(peer)}:{colors.WHITE} {peer}")

        strategy = HivemindStrategy(
            run_id="source",
            batch_size=hparams["batch_size"],
            target_batch_size=hparams["target_batch_size"],
            initial_peers=initial_peers,
            use_ipfs=True,
            verbose=False,
            wait_timeout=90,
            bootstrap_timeout=30,
            scheduler_fn=scheduler,
        )
    else:
        strategy = "auto"

    return strategy
