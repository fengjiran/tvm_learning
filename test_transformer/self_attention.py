import math
import torch
from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer


model_ckpt = "/Users/richard/.cache/huggingface/hub/models--bert-base-uncased/"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "time flies like an arrow"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(inputs.input_ids)

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)

inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())

q = k = v = inputs_embeds

