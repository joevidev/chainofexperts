#!/bin/bash
set -x  # Print each command before execution

export PYTHONPATH=/home/zihan/CoE:$PYTHONPATH

# Define configuration combinations
# Format: experiment_suffix:sparsity:granularity:topk
general_name="mmqa-olmoe_coe-h768"
configs=(
    # "64epts-8topk-1iter-16lyr:64:8:1:16"
    # Memory-matched experiments (64 experts → 8 selected)
    "64epts-8topk-1iter-1lyr:64:8:1:1"
    "64epts-8topk-1iter-2lyr:64:8:1:2"
    "64epts-8topk-1iter-4lyr:64:8:1:4"
    "64epts-8topk-1iter-6lyr:64:8:1:6"
    
    # # # Memory-matched experiments (8 experts → 8 selected, dense)
    "8epts-8topk-1iter-1lyr:8:8:1:1"
    "8epts-8topk-1iter-2lyr:8:8:1:2"
    "8epts-8topk-1iter-4lyr:8:8:1:4" 
    "8epts-8topk-1iter-6lyr:8:8:1:6" 

    # Baseline: 2 layers of 64→8, has done
    # "64epts-8topk-1iter-4lyr:64:8:1:4"
    # Dense: 2 layers of 8→8, has done
    # "8epts-8topk-1iter-4lyr:8:8:1:4"

    # Dense Recurrent: 1 layer of 8→8, 2 iterations
    "8epts-8topk-2iter-2lyr:8:8:2:2"
    # Your approach: 1 layer of 64→8, 2 iterations
    "64epts-8topk-2iter-2lyr:64:8:2:2"
    # Compute-matched version with more experts
    "128epts-8topk-2iter-2lyr:128:8:2:2"

    # Your approach with increasing iterations
    "64epts-8topk-1iter-1lyr:64:8:1:1"
    "64epts-8topk-2iter-1lyr:64:8:2:1"
    "64epts-8topk-4iter-1lyr:64:8:4:1"
    "64epts-8topk-8iter-1lyr:64:8:8:1"
    "64epts-8topk-12iter-1lyr:64:8:12:1"
    "64epts-8topk-16iter-1lyr:64:8:16:1"
    
    Dense Recurrent with increasing iterations (for comparison)
    "8epts-8topk-1iter-1lyr:8:8:1:1"
    "8epts-8topk-2iter-1lyr:8:8:2:1"
    "8epts-8topk-4iter-1lyr:8:8:4:1"
    "8epts-8topk-8iter-1lyr:8:8:8:1"
    "8epts-8topk-12iter-1lyr:8:8:12:1"
    "8epts-8topk-16iter-1lyr:8:8:16:1"
)

# Base command line arguments common to all runs
base_args=(
    "trainer.project_name=metamathqa-sft"
    "data.train_files=data/metamathqa/train.parquet"
    "data.val_files=data/metamathqa/test.parquet"
    "data.truncation=right"
    "data.max_length=512"
    "+data.text_keys=['query','response']"
    "data.micro_batch_size_per_gpu=32"
    "data.train_batch_size=256"
    "model.partial_pretrain=config/models/olmoe_coe"
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
    IFS=':' read -r suffix num_experts num_experts_per_tok inner_iter num_hidden_layers <<< "$config"
    
    # Build command
    cmd="torchrun main.py"
    
    # Add base arguments
    for arg in "${base_args[@]}"; do
        cmd+=" $arg"
    done
    
    # Add configuration-specific parameters
    cmd+=" trainer.experiment_name=$general_name-$suffix"
    cmd+=" +model.override_config.num_experts=$num_experts"
    cmd+=" +model.override_config.num_experts_per_tok=$num_experts_per_tok"
    cmd+=" +model.override_config.inner_iter=$inner_iter"
    cmd+=" +model.override_config.num_hidden_layers=$num_hidden_layers"
    # Add any additional arguments passed to this script
    cmd+=" $@"
    
    # Execute command
    echo "Running experiment: $general_name-$suffix"
    eval $cmd

done
