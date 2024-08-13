import os
import argparse
import numpy as np
from transformers import AutoTokenizer
from aigen import aigen
from aigen.train import AIGProgressBar, AIGModelSaver, AIGSampleGenerator, AIGMetricsLogger, AIGTrainer
from lightning.pytorch.trainer import Trainer
from aigen.datasets import StaticDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a GPT model using AIgen.")
    parser.add_argument("--path", type=str, required=True, help="Path to the dataset file.")
    args = parser.parse_args()
    return args

def load_and_tokenize_dataset(file_path, tokenizer):
    # Load and tokenize the dataset
    full_dataset = StaticDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        batch_size=256,
        block_size=128,
        line_by_line=False
    )
    tokens = np.array(full_dataset.tokens)
    full_dataset.save()
    print("Full dataset tokens shape:", tokens.shape)  
    return tokens

def create_text_split(tokens, tokenizer, validation_split=0.3):
    train_tokens, val_tokens = train_test_split(tokens, test_size=validation_split, random_state=42)
    
    print("Train tokens shape:", train_tokens.shape)  
    print("Validation tokens shape:", val_tokens.shape)

    return train_tokens, val_tokens

def create_datasets(file_path, tokenizer, validation_split=0.3):
    tokens = load_and_tokenize_dataset(file_path, tokenizer)
    train_tokens, val_tokens = create_text_split(tokens, tokenizer, validation_split)

    train_dataset = StaticDataset(
        tokenized_texts=True,
        texts=train_tokens,
        block_size=128,
        tokenizer=tokenizer
    )

    val_dataset = StaticDataset(
        tokenized_texts=True,
        texts=val_tokens,
        block_size=128,
        tokenizer=tokenizer
    )

    print("Created train dataset with length:", len(train_dataset))
    print("Created val dataset with length:", len(val_dataset))
    
    return train_dataset, val_dataset

args = parse_arguments()

file_path = os.path.abspath(args.path)

print(f"Full file path: {file_path}")
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"{file_path} is not present in the current directory.")

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')

ai = aigen(
    model='EleutherAI/gpt-neo-1.3B',
    model_folder='./trained_model',
    tokenizer=tokenizer
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

train_dataset, val_dataset = create_datasets(file_path, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=26, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=26, shuffle=False)

print(f"Train DataLoader length: {len(train_loader)}")
print(f"Validation DataLoader length: {len(val_loader)}")
 
# Prepare datasets
local_data = [{
    "train": train_dataset,
    "val": val_dataset,
    "weights": np.ones(len(train_dataset))
}]

# Prepare datasets in the AIgen model
ai.prepare_datasets({}, local_data, [])

# Debug: Check combined loaders
print("Combined train loader:")
for i, batch in enumerate(ai.combined_train):
    print(f"Train batch #{i}: Size={len(batch)}")
    break

print("Combined val loader:")
for i, batch in enumerate(ai.combined_val):
    print(f"Val batch #{i}: Size={len(batch)}")
    break

ai.train(
    local_data=local_data,  # Properly pass local_data to utilize prepared datasets
    streaming_data=[],
    tokenizer=tokenizer,
    eos_token="###",
    batch_size=26,
    block_size=128,
    stride=0,
    save_every=150,
    generate_every=50
)

print("Training in progress...")
