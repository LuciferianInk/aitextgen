# Hello World

Here's how you can quickly test out aigen on your own computer, even if you don't have a GPU!

For generating text from a pretrained GPT-2 model:

```py3
from aigen import aigen

# Without any parameters, aigen() will download, cache, and load the 124M GPT-2 "small" model
ai = aigen()

ai.generate()
ai.generate(n=3, max_length=100)
ai.generate(n=3, prompt="I believe in unicorns because", max_length=100)
ai.generate_to_file(n=10, prompt="I believe in unicorns because", max_length=100, temperature=1.2)
```

You can also generate from the command line:

```sh
aigen generate
aigen generate --prompt "I believe in unicorns because" --to_file False
```

Want to train your own mini GPT-2 model on your own computer? Download this [text file of Shakespeare plays](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), cd to that directory in a Teriminal, open up a `python3` console and go:

```py3
from aigen.TokenDataset import TokenDataset
from aigen.tokenizers import train_tokenizer
from aigen.utils import GPT2ConfigCPU
from aigen import aigen

# The name of the downloaded Shakespeare text for training
file_name = "input.txt"

# Train a custom BPE Tokenizer on the downloaded text
# This will save one file: `aigen.tokenizer.json`, which contains the
# information needed to rebuild the tokenizer.
train_tokenizer(file_name)
tokenizer_file = "aigen.tokenizer.json"

# GPT2ConfigCPU is a mini variant of GPT-2 optimized for CPU-training
# e.g. the # of input tokens here is 64 vs. 1024 for base GPT-2.
config = GPT2ConfigCPU()

# Instantiate aigen using the created tokenizer and config
ai = aigen(tokenizer_file=tokenizer_file, config=config)

# You can build datasets for training by creating TokenDatasets,
# which automatically processes the dataset with the appropriate size.
data = TokenDataset(file_name, tokenizer_file=tokenizer_file, block_size=64)

# Train the model! It will save pytorch_model.bin periodically and after completion to the `trained_model` folder.
# On a 2020 8-core iMac, this took ~25 minutes to run.
ai.train(data, batch_size=8, num_steps=50000, generate_every=5000, save_every=5000)

# Generate text from it!
ai.generate(10, prompt="ROMEO:")

# With your trained model, you can reload the model at any time by
# providing the folder containing the pytorch_model.bin model weights + the config, and providing the tokenizer.
ai2 = aigen(model_folder="trained_model",
                tokenizer_file="aigen.tokenizer.json")

ai2.generate(10, prompt="ROMEO:")
```
