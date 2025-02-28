set -x

# if [ "$#" -lt 1 ]; then
#     echo "Usage: run_gemma_2b.sh <save_path> [other_configs...]"
#     exit 1
# fi

# save_path=$1

# Shift the arguments so $@ refers to the rest
# shift 2


torchrun main.py \
    trainer.project_name=metamathqa-sft \
    trainer.experiment_name=metamathqa-sft-pythia-160m \
    data.train_files=data/metamathqa/train.parquet \
    data.val_files=data/metamathqa/test.parquet \
    data.truncation=left \
    +data.text_keys=['query','response'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=config/models/pythia-160m \
    +model.from_config=true \
    +model.override_config.moe_sparsity=1 \
    +model.override_config.moe_granularity=1 \
    +model.override_config.moe_topk=1 \
    trainer.default_local_dir=output \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@
    # +fsdp=false \


torchrun main.py \
    trainer.project_name=metamathqa-sft \
    trainer.experiment_name=metamathqa-sft-pythia-160m-sparse2 \
    data.train_files=data/metamathqa/train.parquet \
    data.val_files=data/metamathqa/test.parquet \
    data.truncation=left \
    +data.text_keys=['query','response'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=config/models/pythia-160m \
    +model.from_config=true \
    +model.override_config.moe_sparsity=2 \
    +model.override_config.moe_granularity=1 \
    +model.override_config.moe_topk=1 \
    trainer.default_local_dir=output \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@


torchrun main.py \
    trainer.project_name=metamathqa-sft \
    trainer.experiment_name=metamathqa-sft-pythia-160m-sparse4 \
    data.train_files=data/metamathqa/train.parquet \
    data.val_files=data/metamathqa/test.parquet \
    data.truncation=left \
    +data.text_keys=['query','response'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=config/models/pythia-160m \
    +model.from_config=true \
    +model.override_config.moe_sparsity=4 \
    +model.override_config.moe_granularity=1 \
    +model.override_config.moe_topk=1 \
    trainer.default_local_dir=output \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@


torchrun main.py \
    trainer.project_name=metamathqa-sft \
    trainer.experiment_name=metamathqa-sft-pythia-160m-sparse4gra2 \
    data.train_files=data/metamathqa/train.parquet \
    data.val_files=data/metamathqa/test.parquet \
    data.truncation=left \
    +data.text_keys=['query','response'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=config/models/pythia-160m \
    +model.from_config=true \
    +model.override_config.moe_sparsity=4 \
    +model.override_config.moe_granularity=2 \
    +model.override_config.moe_topk=1 \
    trainer.default_local_dir=output \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@


torchrun main.py \
    trainer.project_name=metamathqa-sft \
    trainer.experiment_name=metamathqa-sft-pythia-160m-sparse4gra4 \
    data.train_files=data/metamathqa/train.parquet \
    data.val_files=data/metamathqa/test.parquet \
    data.truncation=left \
    +data.text_keys=['query','response'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=config/models/pythia-160m \
    +model.from_config=true \
    +model.override_config.moe_sparsity=4 \
    +model.override_config.moe_granularity=4 \
    +model.override_config.moe_topk=1 \
    trainer.default_local_dir=output \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@