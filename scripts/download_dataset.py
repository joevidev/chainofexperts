import pandas as pd
import datasets
from datasets import load_dataset
import os
import numpy as np
import requests
import json


np.random.seed(42)

gsm8k = load_dataset("gsm8k", "main")

train_data = []
for item in gsm8k["train"]:
    train_data.append({
        "question": item["question"],
        "answer": item["answer"]
    })

test_data = []
for item in gsm8k["test"]:
    test_data.append({
        "question": item["question"],
        "answer": item["answer"]
    })

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

os.makedirs("data/gsm8k", exist_ok=True)
train_df.to_parquet("data/gsm8k/train.parquet")
test_df.to_parquet("data/gsm8k/test.parquet")


# URL of the dataset
url = "https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json"

# Create directory structure
os.makedirs("data/metamathqa", exist_ok=True)

# Download file
print("Downloading MetaMathQA dataset...")
response = requests.get(url)
response.raise_for_status()

# Parse JSON data
print("Processing data...")
data = json.loads(response.content)

# Convert to DataFrame
df = pd.DataFrame(data)

# Randomly select 1000 examples for test set
test_indices = np.random.choice(len(df), size=1000, replace=False)
test_df = df.iloc[test_indices]
train_df = df.drop(test_indices)

# Save as parquet files
print(f"Saving {len(train_df)} examples to train.parquet...")
train_df.to_parquet("data/metamathqa/train.parquet")

print(f"Saving {len(test_df)} examples to test.parquet...")
test_df.to_parquet("data/metamathqa/test.parquet")

print("Complete! Files saved to:")
print("- data/metamathqa/train.parquet")
print("- data/metamathqa/test.parquet")