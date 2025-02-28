set -x

# if [ "$#" -lt 1 ]; then
#     echo "Usage: run_gemma_2b.sh <save_path> [other_configs...]"
#     exit 1
# fi

# save_path=$1

# Shift the arguments so $@ refers to the rest
# shift 2


torchrun main.py \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-pythia-14m-test-rand-spa8gra2 \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    +data.text_keys=['question','answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=config/models/pythia-14m \
    +model.from_config=true \
    +model.override_config.moe_sparsity=8 \
    +model.override_config.moe_granularity=2 \
    +model.override_config.moe_topk=2 \
    trainer.default_local_dir=output \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@
    # +fsdp=false \
