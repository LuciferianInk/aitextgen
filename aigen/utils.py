import random
import hashlib
import string


def model_max_length(config):
    """Returns the maximum generation length for the given model."""
    length = (
        getattr(config, "context_length", None)
        or getattr(config, "n_positions", None)
        or getattr(config, "max_position_embeddings", None)
        or getattr(config, "n_ctx", None)
        or getattr(config, "hidden_size", 2048)
    )
    return length


class colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    WHITE = "\033[0m"


# Generate a pseudo-identity, in the Discord ID format
def get_identity(seed=None, style="original"):
    if style == "original":
        if seed is not None:
            random.seed(seed)

        count = random.choice([17, 18])
        leading = random.choice("123456789")
        identity = leading + "".join(random.choice("0123456789") for _ in range(count))

        random.seed()
    elif style == "new":
        if seed is None:
            seed = random_string()

        char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        hashed = hashlib.sha256(seed.encode()).hexdigest()

        decimal_value = int(hashed, 16)

        length = 3
        identity = ""
        for _ in range(length):
            decimal_value, index = divmod(decimal_value, len(char_set))
            identity += char_set[index]

    return identity


def random_string(length=10):
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choices(characters, k=length))

    return random_string
