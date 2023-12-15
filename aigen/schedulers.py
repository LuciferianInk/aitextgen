def get_scheduler(hparams, optimizer):
    "Prepare scheduler"

    schedule = hparams.get("scheduler", "linear")
    if schedule in ["cosine_with_restarts"]:
        from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=hparams["warmup_steps"],
            num_training_steps=hparams["num_steps"],
            num_cycles=hparams["num_cycles"],
        )
    else:
        from transformers import get_scheduler

        scheduler = get_scheduler(
            schedule,
            optimizer,
            num_warmup_steps=hparams["warmup_steps"],
            num_training_steps=hparams["num_steps"],
        )

    return scheduler
