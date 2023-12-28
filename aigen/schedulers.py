from transformers import (
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_scheduler,
)


def get_scheduler(hparams, optimizer):
    "Prepare scheduler"

    multiplier = hparams.get("accumulate_grad_batches", 1)
    schedule = hparams.get("scheduler", "linear")

    if schedule in ["cosine_with_restarts"]:
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=hparams["warmup_steps"] * multiplier,
            num_training_steps=hparams["num_steps"] * multiplier,
            num_cycles=hparams["num_cycles"],
        )
    else:
        scheduler = get_scheduler(
            schedule,
            optimizer,
            num_warmup_steps=hparams["warmup_steps"] * multiplier,
            num_training_steps=hparams["num_steps"] * multiplier,
        )

    return scheduler
