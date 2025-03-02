#!/bin/bash
set -x  # Print each command before execution

export PYTHONPATH=/home/zihan/CoE:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=29500

# Define configuration combinations
# Format: experiment_suffix:sparsity:granularity:topk
general_name="mmqa-olmoe_coe-h1024"
configs=(
    "itr-c-64epts-8topk-1iter-4lyr:64:8:1:4"
    "itr-c-8epts-8topk-1iter-4lyr:8:8:1:4"
    "itr-c-64epts-8topk-2iter-4lyr:64:8:2:4"
    "itr-c-8epts-8topk-2iter-4lyr:8:8:2:4"
    "itr-c-64epts-8topk-3iter-4lyr:64:8:3:4"
    "itr-c-8epts-8topk-3iter-4lyr:8:8:3:4"
    "itr-c-64epts-8topk-4iter-4lyr:64:8:4:4"
    "itr-c-8epts-8topk-4iter-4lyr:8:8:4:4"


    # "lyr-cm-64epts-8topk-1iter-2lyr:64:8:1:2"
    # "lyr-cm-8epts-8topk-1iter-2lyr:8:8:1:2"
    # "lyr-cm-64epts-8topk-1iter-4lyr:64:8:1:4"
    # "lyr-cm-8epts-8topk-1iter-4lyr:8:8:1:4"
    # "lyr-cm-64epts-8topk-1iter-6lyr:64:8:1:6"
    # "lyr-cm-8epts-8topk-1iter-6lyr:8:8:1:6"
    # "lyr-cm-64epts-8topk-1iter-8lyr:64:8:1:8"
    # "lyr-cm-8epts-8topk-1iter-8lyr:8:8:1:8"

    # "lyr-m-64epts-16topk-1iter-2lyr:64:16:1:2"
    # "lyr-m-64epts-8topk-1iter-4lyr:64:8:1:4"
    # "lyr-m-64epts-6topk-1iter-6lyr:64:6:1:6"
    # "lyr-m-64epts-4topk-1iter-8lyr:64:4:1:8"


    # "wid-cm-32epts-4topk-1iter-4lyr:32:4:1:4"
    # "wid-cm-4epts-4topk-1iter-4lyr:4:4:1:4"
    # "wid-cm-64epts-8topk-1iter-4lyr:64:8:1:4"
    # "wid-cm-8epts-8topk-1iter-4lyr:8:8:1:4"
    # "wid-cm-96epts-12topk-1iter-4lyr:96:12:1:4"
    # "wid-cm-12epts-12topk-1iter-4lyr:12:12:1:4"
    # "wid-cm-128epts-16topk-1iter-4lyr:128:16:1:4"
    # "wid-cm-16epts-16topk-1iter-4lyr:16:16:1:4"

    # "wid-m-32epts-8topk-1iter-4lyr:32:8:1:4"
    # "wid-m-64epts-8topk-1iter-4lyr:64:8:1:4"
    # "wid-m-96epts-8topk-1iter-4lyr:96:8:1:4"
    # "wid-m-128epts-8topk-1iter-4lyr:128:8:1:4"
)

# Base command line arguments common to all runs
base_args=(
    "trainer.project_name=metamathqa-sft"
    "data.train_files=data/metamathqa/train.parquet"
    "data.val_files=data/metamathqa/test.parquet"
    "data.truncation=right"
    "data.max_length=512"
    "+data.text_keys=['query','response']"
    "data.train_batch_size=32"
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
    cmd="torchrun --master_port=$MASTER_PORT main.py"
    
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

    # calculate batch size per gpu. for 16 layers, this is 32. Should be smaller than training batch size.
    # batch_size_per_gpu=$((16 * 4 / $num_hidden_layers))
    batch_size_per_gpu=4
    cmd+=" data.micro_batch_size_per_gpu=$batch_size_per_gpu"
    echo "Batch size per GPU: $batch_size_per_gpu"

    # Add any additional arguments passed to this script
    cmd+=" $@"
    
    # Execute command
    echo "Running experiment: $general_name-$suffix"
    eval $cmd

done
