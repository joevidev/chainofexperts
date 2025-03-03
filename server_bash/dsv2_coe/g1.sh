export CUDA_DEVICE=1
export MASTER_PORT=29501

# Define configuration combinations
# Format: experiment_suffix:sparsity:granularity:topk:num_hidden_layers:use_igate
export GENERAL_NAME="coe_v2"
export CONFIGS=(
    "64ept-8tpk-1itr:64:8:1"
)

# Add any extra configurations or overrides

# For example, to override or add model configuration:
# export MODEL_OVERRIDES="use_decoder_only_format=true dropout=0.1"


# Add any extra arguments to the base command
# export EXTRA_ARGS="+trainer.checkpoint_interval_steps=50 +trainer.seed=42"

# Run the common script
source "$(dirname "$0")/g.sh" "$@"