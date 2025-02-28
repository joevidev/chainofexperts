# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
A lightweight one-file FSDP SFT Trainer
"""

import os
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import logging
import re
import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from verl.utils.dataset import SFTDataset
from torch.distributed.device_mesh import DeviceMesh
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

from coe.utils.dataset.base_dataset import BaseDataset

# temporary for debugging
from typing import List, Union
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer




def extract_step(path):
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


class BaseTrainer(FSDPSFTTrainer):

    def __init__(self, config, device_mesh: DeviceMesh=None, ulysses_device_mesh: DeviceMesh=None):
        if config.get("fsdp", True):
            super().__init__(config, device_mesh, ulysses_device_mesh)
        else:
            raise NotImplementedError
            # below is for debugging in single-process manner
            # self.config = config
            # local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
            # from verl.utils import hf_tokenizer
            # self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
            # if self.config.data.chat_template is not None:
            #     raise ValueError('Apply Chat template from config is not supported yet.')

            # self.use_remove_padding = getattr(self.config, 'use_remove_padding', False)

            # self._build_dataloader()
            # self._build_model_optimizer()

            # # TODO: add checkpoint manager
            # if self.device_mesh.get_rank() == 0:
            #     print(self.config)
        
    def _build_dataloader(self):
        config = self.config
        # build dataset
        self.train_dataset = BaseDataset(
            parquet_files=config.data.train_files,
            tokenizer=self.tokenizer,
            text_keys=config.data.text_keys,
            max_length=config.data.max_length,
            truncation=config.data.truncation
        )
        self.val_dataset = BaseDataset(
            parquet_files=config.data.val_files,
            tokenizer=self.tokenizer,
            text_keys=config.data.text_keys,
            max_length=config.data.max_length,
            truncation=config.data.truncation
        )

        # Determine rank and world size for data distribution
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank('dp')
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f'Using SP rank {rank} and size {world_size} for data distribution')
                print(f'Each SP rank gets different data, but the same data WITHIN the same rank')
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
            
        if self.device_mesh.get_rank() == 0:
            print(f'Using FSDP rank {rank} and size {world_size} for data distribution')

        # Set up train dataloader with sampler
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True
        )
        
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )

        # Set up validation dataloader with sampler
        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True
        )
        
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )