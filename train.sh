#!/bin/bash
set -x  # Print each command before execution

export PYTHONPATH=/home/zihan/CoE:$PYTHONPATH

# Define configuration combinations
# Format: experiment_suffix:sparsity:granularity:topk
configs=(
    # "main-6lyr:2:64:6:1:6"
    # "main-5lyr:2:64:6:1:5"
    # "main-4lyr:2:64:6:1:4"
    # "main-3lyr:2:64:6:1:3"
    # "main-2lyr:2:64:6:1:2"
    "main-1lyr:2:64:6:1:1"
)

# Base command line arguments common to all runs
base_args=(
    "trainer.project_name=metamathqa-sft"
    "data.train_files=data/metamathqa/train.parquet"
    "data.val_files=data/metamathqa/test.parquet"
    "data.truncation=right"
    "+data.text_keys=['query','response']"
    "data.micro_batch_size_per_gpu=2"
    "data.train_batch_size=32"
    "model.partial_pretrain=config/models/qwen-moe"
    "+model.from_config=true"
    "+model.override_config._attn_implementation=flash_attention_2"
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
    IFS=':' read -r suffix n_shared_experts n_routed_experts num_experts_per_tok inner_iter num_hidden_layers <<< "$config"
    
    # Build command
    cmd="torchrun main.py"
    
    # Add base arguments
    for arg in "${base_args[@]}"; do
        cmd+=" $arg"
    done
    
    # Add configuration-specific parameters
    cmd+=" trainer.experiment_name=metamathqa-dsmoe-$suffix"
    cmd+=" +model.override_config.num_hidden_layers=$num_hidden_layers"


    # Add any additional arguments passed to this script
    cmd+=" $@"
    
    # Execute command
    echo "Running experiment: metamathqa-dsmoe-$suffix"
    eval $cmd
    
    # Optional: Add a pause between experiments
    # echo "Press Enter to continue to the next experiment..."
    # read
done
