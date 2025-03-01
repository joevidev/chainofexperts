#!/bin/bash
set -x  # Print each command before execution

export PYTHONPATH=/home/zihan/CoE:$PYTHONPATH

# Define configuration combinations
# Format: experiment_suffix:sparsity:granularity:topk
configs=(
    # "sparse2:2:1:1"    # 2 experts, granularity 1, topk 1
    # "sparse4:4:1:1"    # 4 experts, granularity 1, topk 1
    # "sparse1:1:1:1"    # Original pythia-160m (no MoE)
    # "sparse4gra2:4:2:1"  # 4 experts, granularity 2, topk 1
    # "sparse4gra4:4:4:1"  # 4 experts, granularity 4, topk 1
)

# Base command line arguments common to all runs
base_args=(
    "trainer.project_name=metamathqa-sft"
    "data.train_files=data/metamathqa/train.parquet"
    "data.val_files=data/metamathqa/test.parquet"
    "data.truncation=right"
    "+data.text_keys=['query','response']"
    "data.micro_batch_size_per_gpu=8"
    "data.train_batch_size=32"
    "model.partial_pretrain=config/models/pythia-dsmoe-160m"
    "+model.from_config=true"
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
    IFS=':' read -r suffix sparsity granularity topk <<< "$config"
    
    # Build command
    cmd="torchrun main.py"
    
    # Add base arguments
    for arg in "${base_args[@]}"; do
        cmd+=" $arg"
    done
    
    # Add configuration-specific parameters
    cmd+=" trainer.experiment_name=metamathqa-sft-pythia-160m-$suffix"
    cmd+=" +model.override_config.moe_sparsity=$sparsity"
    cmd+=" +model.override_config.moe_granularity=$granularity"
    cmd+=" +model.override_config.moe_topk=$topk"
    
    # Add any additional arguments passed to this script
    cmd+=" $@"
    
    # Execute command
    echo "Running experiment: metamathqa-sft-pythia-160m-$suffix"
    eval $cmd
    
    # Optional: Add a pause between experiments
    # echo "Press Enter to continue to the next experiment..."
    # read
done
