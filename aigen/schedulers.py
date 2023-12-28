from transformers import get_scheduler


def get_schedule(hparams, optimizer):
    "Prepare scheduler"

    multiplier = hparams.get("accumulate_grad_batches", 1)
    schedule = hparams.get("scheduler", "linear")

    scheduler_specific_kwargs = {}
    if schedule == "cosine_with_restarts":
        scheduler_specific_kwargs["num_cycles"] = hparams.get("num_cycles", 3)

    scheduler = get_scheduler(
        schedule,
        optimizer,
        num_warmup_steps=hparams["warmup_steps"] * multiplier,
        num_training_steps=hparams["num_steps"] * multiplier,
        scheduler_specific_kwargs=scheduler_specific_kwargs,
    )

    return scheduler
