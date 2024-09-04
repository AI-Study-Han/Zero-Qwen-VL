import os
import torch

def extract_and_save_weights(source_dir, output_file):
    # 创建一个新的字典来存储提取的权重
    extracted_weights = {}

    # 指定要处理的文件名列表
    target_files = [
        "pytorch_model-00008-of-00010.bin",
        "pytorch_model-00009-of-00010.bin",
        "pytorch_model-00010-of-00010.bin"
    ]

    # 遍历指定的文件
    for file_name in target_files:
        file_path = os.path.join(source_dir, file_name)
        if os.path.exists(file_path):
            # 加载 .bin 文件中的权重
            state_dict = torch.load(file_path, map_location=torch.device('cpu'))

            # 遍历权重并提取以 transformer.visual 开头的部分
            for key, value in state_dict.items():
                if key.startswith("transformer.visual"):
                    # 打印每个权重的 key 和 dtype
                    print(f"Key: {key}, Data Type: {value.dtype}")
                    
                    # 去掉 transformer.visual 前缀
                    new_key = key.replace("transformer.visual.", "")
                    extracted_weights[new_key] = value

    # 将提取的权重保存到新的 .bin 文件中
    torch.save(extracted_weights, output_file)
    print(f"提取的权重已保存到 {output_file}")

# 设置源目录和输出文件路径
source_directory = './Qwen-VL-Chat'  # 修改为你 .bin 文件所在的目录路径
output_file_path = './vision/pytorch_model.bin'  # 设置输出文件路径

# 运行提取和保存
extract_and_save_weights(source_directory, output_file_path)

from llava_pro.configuration_llavapro import LlavaproConfig
from llava_pro.modeling_llavapro import LlavaproForConditionalGeneration
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

vision_model = AutoModel.from_pretrained('./vision', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
vision_config = vision_model.config

llm_model = AutoModelForCausalLM.from_pretrained('./Qwen2-0.5B-Instruct', torch_dtype=torch.bfloat16, device_map="cuda")
text_config = llm_model.config
llm_tokenizer = AutoTokenizer.from_pretrained('./Qwen2-0.5B-Instruct')
print(llm_tokenizer.encode("<image>"))

configuration = LlavaproConfig(vision_config, text_config)
model = LlavaproForConditionalGeneration(configuration)
model = model.to(torch.bfloat16)

model.vision_tower = vision_model
model.language_model = llm_model
model.config.pad_token_id = llm_tokenizer.pad_token_id
model.config.image_token_index = llm_tokenizer.encode("<image>")[0]
model.save_pretrained("./llava_pro")
llm_tokenizer.save_pretrained("./llava_pro")