#!/usr/bin/env python3
import argparse
import yaml
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    parser.add_argument("--config", type=str, default="config/base.yaml", help="Config path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--sample_size", type=int, default=None, help="Dataset sample size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--fp16", action="store_true", default=None, help="Use fp16")
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
    
    # Create model, tokenizer
    model_config = AutoConfig.from_pretrained(config['model']['config_path'], 
                                             trust_remote_code=config['model']['trust_remote_code'])
    model = AutoModelForCausalLM.from_config(model_config, 
                                            trust_remote_code=config['model']['trust_remote_code'])
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['path'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Load and process dataset
    dataset = load_dataset(config['data']['name'], config['data']['config'], split=config['data']['split'])
    if config['data'].get('sample_size'):
        dataset = dataset.take(config['data']['sample_size'])
    
    # Preprocessing function
    def preprocess(examples):
        tokenized = tokenizer(examples["text"], 
                             truncation=config['data']['preprocessing']['truncation'],
                             max_length=config['data']['preprocessing']['max_length'],
                             padding=config['data']['preprocessing']['padding'],
                             return_tensors=config['data']['preprocessing']['return_tensors'])
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Preprocess dataset
    tokenized_dataset = dataset.map(
        preprocess, 
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names if config['data']['preprocessing']['remove_columns'] else None,
        desc="Tokenizing dataset",  # Add a description for the progress bar
        load_from_cache_file=not config.get('data', {}).get('disable_caching', False)  # Enable caching by default
    )
    
    # Setup training
    os.makedirs(config['model']['output_dir'], exist_ok=True)
    training_args = TrainingArguments(
        output_dir=config['model']['output_dir'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        save_steps=config['training']['save_steps'],
        logging_steps=config['training']['logging_steps'],
        fp16=config['training']['fp16'],
        prediction_loss_only=config['training']['prediction_loss_only'],
        save_total_limit=config['checkpointing']['save_total_limit'],
        save_strategy=config['checkpointing']['save_strategy'],
        do_eval=config['evaluation']['do_eval'],
        eval_steps=config['evaluation']['eval_steps'] if config['evaluation']['do_eval'] else None,
        warmup_steps=config['scheduler']['warmup_steps'],
    )
    
    # Train and save
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()
    model.save_pretrained(config['model']['output_dir'])
    tokenizer.save_pretrained(config['model']['output_dir'])
    print(f"Training completed. Model saved to {config['model']['output_dir']}")

if __name__ == "__main__":
    main()