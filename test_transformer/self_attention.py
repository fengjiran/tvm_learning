from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer


model_ckpt = "/Users/richard/.cache/huggingface/hub/models--bert-base-uncased/"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "time flies like an arrow"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(inputs.input_ids)
