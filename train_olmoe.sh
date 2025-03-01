#!/bin/bash
set -x  # Print each command before execution

export PYTHONPATH=/home/zihan/CoE:$PYTHONPATH

# Define configuration combinations
# Format: experiment_suffix:sparsity:granularity:topk
configs=(
    # "64epts-8topk-1iter-16lyr:64:8:1:16"
    # Memory-matched experiments (64 experts → 8 selected)
    "64epts-8topk-1iter-1lyr:64:8:1:1"
    # "64epts-8topk-1iter-2lyr:64:8:1:2"
    # "64epts-8topk-1iter-6lyr:64:8:1:6"
    # "64epts-8topk-1iter-8lyr:64:8:1:8"
    
    # # Memory-matched experiments (8 experts → 8 selected, dense)
    # "8epts-8topk-1iter-1lyr:8:8:1:1"
    # "8epts-8topk-1iter-2lyr:8:8:1:2"
    # "8epts-8topk-1iter-4lyr:8:8:1:4" 
    # "8epts-8topk-1iter-6lyr:8:8:1:6" 
    # "8epts-8topk-1iter-8lyr:8:8:1:8"
)

# Base command line arguments common to all runs
base_args=(
    "trainer.project_name=metamathqa-sft"
    "data.train_files=data/metamathqa/train.parquet"
    "data.val_files=data/metamathqa/test.parquet"
    "data.truncation=right"
    "+data.text_keys=['query','response']"
    "data.micro_batch_size_per_gpu=1"
    "data.train_batch_size=32"
    "model.partial_pretrain=config/models/olmoe_coe_tiny"
    "+model.from_config=true"
    "+model.override_config._attn_implementation=flash_attention_2"
    "+model.override_config.gradient_checkpointing=true"
    "trainer.default_local_dir=output"
    "trainer.total_epochs=null"
    "trainer.total_training_steps=1000"
    "trainer.validation_interval_steps=10"
    "trainer.total_validation_count=100"
    "trainer.logger=['console','wandb']"
    "trainer.default_hdfs_dir=null"
)

# Run each configuration
for config in "${configs[@]}"; do
    # Parse configuration
    IFS=':' read -r suffix num_experts num_experts_per_tok inner_iter num_hidden_layers <<< "$config"
    
    # Build command
    cmd="torchrun main.py"
    
    # Add base arguments
    for arg in "${base_args[@]}"; do
        cmd+=" $arg"
    done
    
    # Add configuration-specific parameters
    cmd+=" trainer.experiment_name=mmqa-olmoe_coe_tiny-$suffix"
    cmd+=" +model.override_config.num_experts=$num_experts"
    cmd+=" +model.override_config.num_experts_per_tok=$num_experts_per_tok"
    cmd+=" +model.override_config.inner_iter=$inner_iter"
    cmd+=" +model.override_config.num_hidden_layers=$num_hidden_layers"
    # Add any additional arguments passed to this script
    cmd+=" $@"
    
    # Execute command
    echo "Running experiment: mmqa-olmoe_coe_tiny-$suffix"
    eval $cmd

done
