import pandas as pd
import datasets
from datasets import load_dataset
import os

# 加载GSM8K数据集
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

# 转换为pandas DataFrame
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# 保存为parquet文件
os.makedirs("data/gsm8k", exist_ok=True)
train_df.to_parquet("data/gsm8k/train.parquet")
test_df.to_parquet("data/gsm8k/test.parquet")
