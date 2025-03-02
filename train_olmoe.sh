#!/bin/bash
set -x  # Print each command before execution

export PYTHONPATH=/home/zihan/CoE:$PYTHONPATH

# Define configuration combinations
# Format: experiment_suffix:sparsity:granularity:topk
general_name="mmqa-olmoe_coe-h1024"
configs=(
    # these experiments have been done!
    # # Dense: 2 layers of 8→8, has done
    # "8epts-8topk-1iter-4lyr:8:8:1:4"
    # # Dense Recurrent: 1 layer of 8→8, 2 iterations
    # "8epts-8topk-2iter-4lyr:8:8:2:4"
    # # Baseline: 2 layers of 64→8, has done
    # "64epts-8topk-1iter-4lyr:64:8:1:4"
    # # Your approach: 1 layer of 64→8, 2 iterations. same memory but more compute
    # "64epts-8topk-2iter-4lyr:64:8:2:4"

    # Compute-matched version with more experts. same compute and same memory
    "128epts-8topk-2iter-2lyr:128:8:2:2"
    "64epts-4topk-2iter-4lyr:64:4:2:4"

    # Compute-matched version with more experts. same compute but less memory
    "64epts-8topk-2iter-2lyr:64:8:2:2"
    "32epts-4topk-2iter-4lyr:32:4:2:4"


    # below may be reran using smaller batch size
    # layer scaling law vs width scaling law vs communication scaling law
    # iteration scaling law (compute +, but memory is same)
    "64epts-8topk-1iter-4lyr:64:8:1:4"
    "8epts-8topk-1iter-4lyr:8:8:1:4"
    "64epts-8topk-2iter-4lyr:64:8:2:4"
    "8epts-8topk-2iter-4lyr:8:8:2:4"
    "64epts-8topk-4iter-4lyr:64:8:4:4"
    "8epts-8topk-4iter-4lyr:8:8:4:4"
    "64epts-8topk-8iter-4lyr:64:8:8:4"
    "8epts-8topk-8iter-4lyr:8:8:8:4"


    # layer scaling law (compute and memory +)
    "64epts-8topk-1iter-4lyr:64:8:1:4"
    "8epts-8topk-1iter-4lyr:8:8:1:4"
    "64epts-8topk-1iter-8lyr:64:8:1:8"
    "8epts-8topk-1iter-8lyr:8:8:1:8"
    "64epts-8topk-1iter-12lyr:64:8:1:12"
    "8epts-8topk-1iter-12lyr:8:8:1:12"
    "64epts-8topk-1iter-16lyr:64:8:1:16"
    "8epts-8topk-1iter-16lyr:8:8:1:16"
    # layer scaling law (memory + but compute is same)
    "64epts-8topk-1iter-4lyr:64:8:1:4"
    "64epts-4topk-1iter-8lyr:64:4:1:8"
    "64epts-3topk-1iter-12lyr:64:3:1:12"
    "64epts-2topk-1iter-16lyr:64:2:1:16"
    # "8epts-8topk-1iter-4lyr:8:8:1:4"
    # "8epts-4topk-1iter-8lyr:8:4:1:8"
    # "8epts-3topk-1iter-12lyr:8:3:1:12"
    # "8epts-2topk-1iter-16lyr:8:2:1:16"

    # width scaling law (compute and memory +)
    "64epts-8topk-1iter-4lyr:64:8:1:4"
    "8epts-8topk-1iter-4lyr:8:8:1:4"
    "128epts-16topk-1iter-4lyr:128:16:1:4"
    "16epts-16topk-1iter-4lyr:16:16:1:4"
    "192epts-24topk-1iter-4lyr:192:24:1:4"
    "24epts-24topk-1iter-4lyr:24:24:1:4"
    "256epts-32topk-1iter-4lyr:256:32:1:4"
    "32epts-32topk-1iter-4lyr:32:32:1:4"
    # width scaling law (memory + but compute is same)
    "64epts-8topk-1iter-4lyr:64:8:1:4"
    "128epts-8topk-1iter-4lyr:128:8:1:4"
    "192epts-8topk-1iter-4lyr:192:8:1:4"
    "256epts-8topk-1iter-4lyr:256:8:1:4"
    # "8epts-8topk-1iter-4lyr:8:8:1:4"
    # "8epts-8topk-1iter-4lyr:8:8:1:4"
    # "8epts-8topk-1iter-4lyr:8:8:1:4"
    # "8epts-8topk-1iter-4lyr:8:8:1:4"


)


    # # "64epts-8topk-1iter-16lyr:64:8:1:16"
    # # Memory-matched experiments (64 experts → 8 selected)
    # # "64epts-8topk-1iter-1lyr:64:8:1:1"
    # # "64epts-8topk-1iter-2lyr:64:8:1:2"
    # # "64epts-8topk-1iter-6lyr:64:8:1:6"
    # "64epts-8topk-1iter-4lyr:64:8:1:4"
    # "64epts-8topk-1iter-8lyr:64:8:1:8"
    # "64epts-8topk-1iter-12lyr:64:8:1:12"
    # "64epts-8topk-1iter-16lyr:64:8:1:16"
    
    # # # # Memory-matched experiments (8 experts → 8 selected, dense)
    # # "8epts-8topk-1iter-1lyr:8:8:1:1"
    # # "8epts-8topk-1iter-2lyr:8:8:1:2"
    # # "8epts-8topk-1iter-6lyr:8:8:1:6" 
    # "8epts-8topk-1iter-4lyr:8:8:1:4" 
    # "8epts-8topk-1iter-8lyr:8:8:1:8"
    # "8epts-8topk-1iter-12lyr:8:8:1:12"
    # "8epts-8topk-1iter-16lyr:8:8:1:16"

    # # Baseline: 2 layers of 64→8, has done
    # # "64epts-8topk-1iter-4lyr:64:8:1:4"
    # # Dense: 2 layers of 8→8, has done
    # # "8epts-8topk-1iter-4lyr:8:8:1:4"

    # # Dense Recurrent: 1 layer of 8→8, 2 iterations
    # # "8epts-8topk-2iter-2lyr:8:8:2:2"
    # "8epts-8topk-2iter-4lyr:8:8:2:4"
    # # Your approach: 1 layer of 64→8, 2 iterations
    # "64epts-8topk-2iter-4lyr:64:8:2:4"
    # # Compute-matched version with more experts
    # "128epts-8topk-2iter-4lyr:128:8:2:4"

    # # Your approach with increasing iterations
    # "64epts-8topk-1iter-4lyr:64:8:1:4"
    # "64epts-8topk-2iter-4lyr:64:8:2:4"
    # "64epts-8topk-4iter-4lyr:64:8:4:4"
    # "64epts-8topk-8iter-4lyr:64:8:8:4"
    # "64epts-8topk-12iter-4lyr:64:8:12:4"
    # "64epts-8topk-16iter-4lyr:64:8:16:4"
    
    # # Dense Recurrent with increasing iterations (for comparison)
    # "8epts-8topk-1iter-4lyr:8:8:1:4"
    # "8epts-8topk-2iter-4lyr:8:8:2:4"
    # "8epts-8topk-4iter-4lyr:8:8:4:4"
    # "8epts-8topk-8iter-4lyr:8:8:8:4"
    # "8epts-8topk-12iter-4lyr:8:8:12:4"
    # "8epts-8topk-16iter-4lyr:8:8:16:4"

# Base command line arguments common to all runs
base_args=(
    "trainer.project_name=metamathqa-sft"
    "data.train_files=data/metamathqa/train.parquet"
    "data.val_files=data/metamathqa/test.parquet"
    "data.truncation=right"
    "data.max_length=512"
    "+data.text_keys=['query','response']"
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

    # calculate batch size per gpu. for 16 layers, this is 32. Should be smaller than training batch size.
    # batch_size_per_gpu=$((16 * 4 / $num_hidden_layers))
    cmd+=" data.micro_batch_size_per_gpu=$batch_size_per_gpu"
    echo "Batch size per GPU: $batch_size_per_gpu"

    # Add any additional arguments passed to this script
    cmd+=" $@"
    
    # Execute command
    echo "Running experiment: $general_name-$suffix"
    eval $cmd

done
