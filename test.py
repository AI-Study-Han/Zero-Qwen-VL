from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoProcessor
from vision.configuation_qwenvl import QWENVisionConfig
from vision.modeling_qwenvl import VisionTransformer
import torch
from safetensors import safe_open
from llava_pro.processing_llava import LlavaproProcessor
from llava_pro.modeling_llavapro import LlavaproForConditionalGeneration

from PIL import Image


#初步测试模型效果
model_name_or_path = "./test/llava_pro"  # 


llava_processor = LlavaproProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
model = LlavaproForConditionalGeneration.from_pretrained(
    model_name_or_path, device_map="cuda:1", torch_dtype=torch.bfloat16, local_files_only=True
)
prompt_text = "<image>\n这个图片描述的是什么内容？"


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_text},
]
prompt = llava_processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)


image_path = "./test3.jpeg"
image = Image.open(image_path)


inputs = llava_processor(text=prompt, images=image, return_tensors="pt")

for tk in inputs.keys():
    inputs[tk] = inputs[tk].to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=20)
gen_text = llava_processor.batch_decode(
    generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
)[0]

print(gen_text)


