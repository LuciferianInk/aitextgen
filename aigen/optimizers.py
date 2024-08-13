def get_optimizer(params, hparams):
    if hparams["optimizer"] == "Lion":
        from pytorch_optimizer import Lion

        opt = Lion(
            params,
            lr=hparams["learning_rate"],
            betas=(0.9, 0.99),
            r=0.95,
            use_gc=True,
            adanorm=False,
        )
    elif hparams["optimizer"] == "AdaBelief":
        from pytorch_optimizer import AdaBelief

        opt = AdaBelief(
            params,
            lr=hparams["learning_rate"],
            betas=(0.9, 0.999),
            r=0.95,
            rectify=True,
        )
    elif hparams["optimizer"] == "Prodigy":
        from pytorch_optimizer import Prodigy

        opt = Prodigy(
            params,
            lr=1.0,
            safeguard_warmup=True,
            bias_correction=True,
        )
    elif hparams["optimizer"] == "Ranger21":
        from pytorch_optimizer import Ranger21

        opt = Ranger21(
            params,
            lr=hparams["learning_rate"],
            lookahead_merge_time=5,
            num_iterations=1,
        )
    elif hparams["optimizer"] == "RMSProp":
        from torch.optim import RMSprop

        opt = RMSprop(
            params,
            lr=hparams["learning_rate"],
            momentum=hparams.get("momentum", 0),
            alpha=0.999,
            maximize=False,
            centered=False,
        )
    elif hparams["optimizer"] == "Adan":
        from pytorch_optimizer import Adan

        opt = Adan(
            params,
            lr=hparams["learning_rate"],
        )
    elif hparams["optimizer"] == "Kate":
        from pytorch_optimizer import Kate

        opt = Kate(
            params,
            lr=hparams["learning_rate"],
        )
    elif hparams["optimizer"] == "AdamG":
        from pytorch_optimizer import AdamG

        opt = AdamG(
            params,
            p=0.2,
            q=0.24,
            lr=1.0,
        )
    elif hparams["optimizer"] in ["SignSGD", "Signum"]:
        from pytorch_optimizer import SignSGD

        opt = SignSGD(
            params,
            lr=hparams["learning_rate"],
            momentum=hparams.get("momentum", 0.9),
        )
    else:
        if hparams.get("deepspeed"):
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            opt = DeepSpeedCPUAdam(
                params,
                lr=hparams["learning_rate"],
                eps=hparams.get("eps", 1e-8),
                adamw_mode=True,
            )
        else:
            from torch.optim import AdamW

            opt = AdamW(
                params,
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
