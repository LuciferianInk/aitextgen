from pytorch_optimizer import create_optimizer


def get_optimizer(model, weight_decay, use_lookahead, kwargs):
    wd_ban_list = [
        "bias",
        "pos_emb",
        "Embedding",
        "Embedding.weight",
        "Embedding.bias",
        "BatchNorm.weight",
        "BatchNorm.bias",
        "GroupNorm.weight",
        "GroupNorm.bias",
        "LayerNorm.weight",
        "LayerNorm.bias",
        "RMSNorm.weight",
        "RMSNorm.bias",
        "InstanceNorm.weight",
        "InstanceNorm.bias",
    ]

    return create_optimizer(
        model,
        kwargs["optimizer"],
        kwargs["learning_rate"],
        weight_decay,
        wd_ban_list,
        use_lookahead,
        **kwargs,
    )
