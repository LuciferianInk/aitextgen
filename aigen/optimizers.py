import inspect

from pytorch_optimizer import Lookahead, create_optimizer, load_optimizer


def get_optimizer(model, weight_decay, use_lookahead, kwargs):
    wd_ban_list = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_class = load_optimizer(kwargs["optimizer"])

    # Get the parameter names of the create_optimizer function
    optimizer_params = inspect.signature(optimizer_class).parameters

    # Filter kwargs to match constructor parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in optimizer_params}

    # If using lookahead, inspect its parameters and add them to filtered_kwargs
    if use_lookahead:
        lookahead_params = inspect.signature(Lookahead).parameters
        lookahead_kwargs = {k: v for k, v in kwargs.items() if k in lookahead_params}
        filtered_kwargs.update(lookahead_kwargs)
        del filtered_kwargs["optimizer"]

    print(filtered_kwargs)

    return create_optimizer(
        model,
        kwargs["optimizer"],
        kwargs["learning_rate"],
        weight_decay,
        wd_ban_list,
        use_lookahead,
        **filtered_kwargs,
    )
