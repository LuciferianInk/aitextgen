import inspect

from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftConfig


def get_peft_config(peft_type, kwargs) -> PeftConfig:
    """Loads the specified PEFT adapter configuration.

    All arguments can be referred to here:
    https://github.com/huggingface/peft/tree/main/src/peft/tuners

    Args:
        peft_type (str): The name of the PEFT adapter to load.
        kwargs: Keyword arguments to pass to the adapter configuration class.

    Returns:
        peft.PeftConfig: The PEFT configuration object for the specified adapter.

    Raises:
        ValueError: If the specified PEFT type is not supported.
    """

    cls = PEFT_TYPE_TO_CONFIG_MAPPING.get(peft_type.upper())
    if not cls:
        raise ValueError(f"Unsupported PEFT type: {peft_type}")

    # Filter kwargs to match constructor parameters
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.signature(cls.__init__).parameters
    }

    return cls(**filtered_kwargs, task_type="CAUSAL_LM")
