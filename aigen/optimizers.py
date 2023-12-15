def get_optimizer(model, hparams):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if any(nd in n for nd in no_decay):
            weight_decay = 0.0
        else:
            weight_decay = hparams["weight_decay"]

        optimizer_grouped_parameters.append(
            {
                "params": [p],
                "weight_decay": weight_decay,
            }
        )

    if hparams["optimizer"] == "Lion":
        from pytorch_optimizer import Lion

        opt = Lion(
            optimizer_grouped_parameters,
            lr=hparams["learning_rate"],
            betas=(0.9, 0.99),
            r=0.95,
            use_gc=True,
            adanorm=True,
        )
    elif hparams["optimizer"] == "AdaBelief":
        from pytorch_optimizer import AdaBelief

        opt = AdaBelief(
            optimizer_grouped_parameters,
            lr=hparams["learning_rate"],
            betas=(0.9, 0.999),
            r=0.95,
            rectify=True,
        )
    elif hparams["optimizer"] == "Ranger21":
        from pytorch_optimizer import Ranger21

        opt = Ranger21(
            optimizer_grouped_parameters,
            lr=hparams["learning_rate"],
            lookahead_merge_time=5,
            num_iterations=1,
        )
    elif hparams["optimizer"] == "RMSProp":
        from torch.optim import RMSprop

        opt = RMSprop(
            optimizer_grouped_parameters,
            lr=hparams["learning_rate"],
            momentum=hparams.get("momentum", 0),
            alpha=0.999,
            maximize=False,
            centered=False,
        )
    elif hparams["optimizer"] == "Adan":
        from pytorch_optimizer import Adan

        opt = Adan(
            optimizer_grouped_parameters,
            lr=hparams["learning_rate"],
        )
    else:
        if hparams.get("deepspeed"):
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            opt = DeepSpeedCPUAdam(
                optimizer_grouped_parameters,
                lr=hparams["learning_rate"],
                eps=hparams.get("eps", 1e-8),
                adamw_mode=True,
            )
        else:
            from torch.optim import AdamW

            opt = AdamW(
                optimizer_grouped_parameters,
                lr=hparams["learning_rate"],
                eps=hparams.get("eps", 1e-8),
            )

    lookahead_steps = hparams.get("lookahead", 0)
    if lookahead_steps > 0:
        from pytorch_optimizer import Lookahead

        optimizer = Lookahead(
            opt, k=lookahead_steps, alpha=0.5, pullback_momentum="none"
        )
    else:
        optimizer = opt
    return optimizer
