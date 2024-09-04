import copy
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
)
from llava_pro.processing_llava import LlavaproProcessor
from llava_pro.modeling_llavapro import LlavaproForConditionalGeneration
from train_llava.data import LlavaDataset, TrainLLavaModelCollator
from train_llava.util import print_trainable_parameters

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./model")
    train_type: Optional[str] = field(
        default="none",
        metadata={
            "help": """
            1. none:全量参数训练;
            2. freeze_vision:只冻结vision_tower进行训练
            3. train_projector:只训练连接层
            """
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


def load_model_processor(modelargs: ModelArguments):
    model = LlavaproForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    processor = LlavaproProcessor.from_pretrained(modelargs.model_name_or_path, trust_remote_code=True)

    if modelargs.train_type == "none":
        logging.warning("使用全量参数进行训练")
        pass
    elif modelargs.train_type == "freeze_vision":
        logging.warning("冻结vision_tower网络层，剩下的网络权重进行训练")
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    elif modelargs.train_type == "train_projector":
        logging.warning("只训练projector，其余冻结")
        # 首先冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        # 解除 projector 的冻结状态
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = True
    print_trainable_parameters(model)

    return model, processor


def load_dataset_collator(processor, dataargs: DataArguments):

    llava_dataset = LlavaDataset(
        dataargs.data_path  
    )
    # 打印数据集的大小
    logging.warning(f"数据集大小：{len(llava_dataset)}")
    data_collator = TrainLLavaModelCollator(processor, -100)

    return llava_dataset, data_collator


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, processor = load_model_processor(model_args)
    train_dataset, data_collator = load_dataset_collator(processor, data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
