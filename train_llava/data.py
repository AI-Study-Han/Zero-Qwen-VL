import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
import sys
import os

@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor


class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        super().__init__()

        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)
        # 打印前 100 条数据的 image 信息
        # for index in range(min(100, len(self.chat_data))):
        #     cur_data = self.chat_data[index]
        #     image_path = self.image_dir.joinpath(cur_data.get("image"))
        #     print(f"Index {index}: {image_path}")

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        image_dir = data_dir.joinpath("images")

        chat_data = pd.read_json(chat_file).to_dict(orient="records")

        return chat_data, image_dir

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path, List[Tuple[str, str]]]:
        if index == 668:
            index=666
        if index == 6397:
            index=666
        
        cur_data = self.chat_data[index]
        conversations = cur_data.get("conversations")
        history = []
        for i in range(0, len(conversations) - 2, 2):
            human_input = conversations[i].get("value")
            chatbot_output = conversations[i + 1].get("value")
            history.append((human_input, chatbot_output))

        # Get the last conversation pair
        human_input = conversations[-2].get("value")
        chatbot_output = conversations[-1].get("value")

        image_path = self.image_dir.joinpath(cur_data.get("image"))
        #print(image_path)
        return human_input, chatbot_output, image_path, history


def build_qaimage(
    processor: AutoProcessor,
    q_text: str,
    a_text: str,
    image_path: Path,
    history: List[Tuple[str, str]]
):
    # 初始化messages，添加系统提示
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # 将历史对话添加到messages中
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})

    # 添加当前对话
    messages.append({"role": "user", "content": q_text})
     # 生成prompt
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_file = image_path  # "000000039769.jpg"

    raw_image = Image.open(image_file)
    inputs = processor(prompt, raw_image, return_tensors="pt")  # .to(0, torch.float16)

    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]
    # print("a_input_ids 数据类型:", a_input_ids.dtype, "尺寸:", a_input_ids.shape)
    # print("q_text:", q_text)
    # print("image_path:", image_path)
    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
    )
    return res


class TrainLLavaModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int) -> None:
        self.processor = processor
        self.ingnore_index = IGNORE_INDEX

    def convert_one_piece(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
        # pixel_values: torch.Tensor,
    ):
        input_ids = torch.concat(
            [
                q_input_ids,
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )

        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ingnore_index),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )
        # # 打印 input_ids 的信息
        # print("convert_one_piece - input_ids 数据类型:", input_ids.dtype)
        # print("convert_one_piece - input_ids 尺寸:", input_ids.shape)
        return input_ids, labels

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []

        for feature in features:
            qaimage_output = build_qaimage(
                self.processor, feature[0], feature[1], feature[2], feature[3]
            )
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids
            )
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )
        # # 打印 final_input_ids 的信息
        # print("final_input_ids 数据类型:", final_input_ids.dtype)
        # print("final_input_ids 尺寸:", final_input_ids.shape)
        # print("final_input_ids:", final_input_ids)
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ingnore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )
        final_pixel_values = torch.concat(pixel_values, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        # 打印各个输入张量的类型和尺寸
        #print("input_ids 数据类型:", final_input_ids.dtype, "尺寸:", final_input_ids.shape)
        # print("labels 数据类型:", final_labels.dtype, "尺寸:", final_labels.shape)
        # print("pixel_values 数据类型:", final_pixel_values.dtype, "尺寸:", final_pixel_values.shape)
        # print("attention_mask 数据类型:", attention_mask.dtype, "尺寸:", attention_mask.shape)
        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }


