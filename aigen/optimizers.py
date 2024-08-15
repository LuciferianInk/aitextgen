from pytorch_optimizer import Lookahead, create_optimizer, load_optimizer


def get_optimizer(model, weight_decay, use_lookahead, kwargs):
    wd_ban_list = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    return create_optimizer(
        model,
        kwargs["optimizer"],
        kwargs["learning_rate"],
        weight_decay,
        wd_ban_list,
        use_lookahead,
        **kwargs,
    )
