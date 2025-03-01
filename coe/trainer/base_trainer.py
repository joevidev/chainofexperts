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
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader, DistributedSampler
from torch import nn, optim
from verl.utils.dataset import SFTDataset
from torch.distributed.device_mesh import DeviceMesh
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from verl.utils.torch_functional import get_cosine_schedule_with_warmup

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

from coe.utils.dataset.base_dataset import BaseDataset
from coe.trainer.fsdp_sft_trainer import FSDPSFTTrainer

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

    def __init__(self, config, device_mesh: DeviceMesh=None, ulysses_device_mesh: DeviceMesh=None, dataset_class = BaseDataset):
        if config.get("fsdp", True):
            super().__init__(config, device_mesh, ulysses_device_mesh, dataset_class)
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
        
    def _build_dataset(self):
        config = self.config
        # build dataset
        self.train_dataset = self.dataset_class(
            parquet_files=config.data.train_files,
            tokenizer=self.tokenizer,
            text_keys=config.data.text_keys,
            max_length=config.data.max_length,
            truncation=config.data.truncation
        )
        self.val_dataset = self.dataset_class(
            parquet_files=config.data.val_files,
            tokenizer=self.tokenizer,
            text_keys=config.data.text_keys,
            max_length=config.data.max_length,
            truncation=config.data.truncation
        )

    def _get_model(self, local_model_path, config, trust_remote_code):
        for key, value in self.config.model.override_config.items():
            setattr(config, key, value)
        if self.config.model.from_config:
            model: PreTrainedModel = AutoModelForCausalLM.from_config(config, 
                                                                            torch_dtype=torch.float32,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)
        else:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(local_model_path, 
                                                                            torch_dtype=torch.float32,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)
            
        print("MODEL TOTAL PARAMS:", sum(p.numel() for p in model.parameters()))
        return model
