#!/usr/bin/env python3
import argparse
import yaml
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    parser.add_argument("--config", type=str, default="config/SeMoE-v2.yaml", help="Config path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--sample_size", type=int, default=None, help="Dataset sample size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--fp16", action="store_true", default=None, help="Use fp16")
    parser.add_argument("--val_size", type=int, default=None, help="Validation dataset size")
    return parser.parse_args()

def main():
    # Setup
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with args
    if args.output_dir: config['model']['output_dir'] = args.output_dir
    if args.batch_size: config['training']['per_device_train_batch_size'] = args.batch_size
    if args.sample_size: config['data']['sample_size'] = args.sample_size
    if args.lr: config['training']['learning_rate'] = args.lr
    if args.epochs: config['training']['num_train_epochs'] = args.epochs
    if args.fp16 is not None: config['training']['fp16'] = args.fp16
    if args.val_size: config['evaluation']['val_size'] = args.val_size
    
    # Create model, tokenizer
    model_config = AutoConfig.from_pretrained(config['model']['config_path'], 
                                             trust_remote_code=config['model']['trust_remote_code'])
    # override config
    for key, value in config['model']['config'].items():
        setattr(model_config, key, value)

    model = AutoModelForCausalLM.from_config(model_config, 
                                            trust_remote_code=config['model']['trust_remote_code'])
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['path'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    
    # Load and process dataset
    full_dataset = load_dataset(config['data']['name'], config['data']['config'])
    for key in full_dataset.keys():
        full_dataset[key] = full_dataset[key].filter(lambda x: len(x['text']) >= config['data']['preprocessing']['min_char_length'])
    # print the length of the dataset
    print(len(full_dataset['train']))
    
    # Split into train and validation
    train_dataset = full_dataset[config['data']['split']]
    # if 'validation' in full_dataset:
    #     val_dataset = full_dataset['validation']
    # else:
        # If no validation split exists, create one from train
    train_val_split = train_dataset.train_test_split(
        test_size=config['evaluation']['val_size'],
        seed=42,
        shuffle=True
    )
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']
    
    # Apply sample size limit if specified
    if config['data'].get('sample_size'):
        train_dataset = train_dataset.select(range(min(config['data']['sample_size'], len(train_dataset))))
    
    # Make sure validation set is the right size
    val_size = config['evaluation']['val_size']
    if len(val_dataset) > val_size:
        val_dataset = val_dataset.select(range(val_size))
    
    # Preprocessing function
    def preprocess(examples):
        tokenized = tokenizer(examples["text"], 
                             truncation=config['data']['preprocessing']['truncation'],
                             max_length=config['data']['preprocessing']['max_length'],
                             padding=config['data']['preprocessing']['padding'],
                             return_tensors=config['data']['preprocessing']['return_tensors'])
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Preprocess datasets
    tokenized_train = train_dataset.map(preprocess, batched=True, 
                                   remove_columns=train_dataset.column_names if config['data']['preprocessing']['remove_columns'] else None)
    tokenized_val = val_dataset.map(preprocess, batched=True, 
                                   remove_columns=val_dataset.column_names if config['data']['preprocessing']['remove_columns'] else None)
    
    # run through the model once to check if it's working
    # model(torch.tensor(tokenized_train[:4]['input_ids']))
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    # Setup training
    os.makedirs(config['model']['output_dir'], exist_ok=True)
    training_args = TrainingArguments(
        output_dir=config['model']['output_dir'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['evaluation']['per_device_eval_batch_size'] if 'per_device_eval_batch_size' in config['evaluation'] else config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        save_steps=config['training']['save_steps'],
        logging_steps=config['training']['logging_steps'],
        fp16=config['training']['fp16'],
        prediction_loss_only=config['training']['prediction_loss_only'],
        save_total_limit=config['checkpointing']['save_total_limit'],
        save_strategy=config['checkpointing']['save_strategy'],
        # Enable evaluation
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=config['evaluation']['eval_steps'],
        warmup_steps=config['scheduler']['warmup_steps'],
        # Load best model at end of training
        load_best_model_at_end=config['evaluation']['load_best_model_at_end'] if 'load_best_model_at_end' in config['evaluation'] else False,
        metric_for_best_model="eval_loss" if config['evaluation'].get('load_best_model_at_end') else None,
        greater_is_better=False if config['evaluation'].get('load_best_model_at_end') else None,
    )
    
    # Train and save
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val
    )
    trainer.train()
    model.save_pretrained(config['model']['output_dir'])
    tokenizer.save_pretrained(config['model']['output_dir'])
    print(f"Training completed. Model saved to {config['model']['output_dir']}")

if __name__ == "__main__":
    main()