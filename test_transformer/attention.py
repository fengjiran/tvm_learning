import sys
import math
import torch
from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)
        scale = torch.sqrt(query.size(-1))
        scores = torch.bmm(query, key.transpose(1, 2)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float("inf"))
        weights = self.softmax(scores)
        return torch.bmm(weights, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None, query_mask=None, key_mask=None):
        if query_mask is not None and key_mask is not None:
            mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
        x = torch.cat([h(query, key, value, mask=mask) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


if __name__ == "__main__":
    if sys.platform.startswith("darwin"):
        model_ckpt = "/Users/richard/.cache/huggingface/hub/models--bert-base-uncased/"
    elif sys.platform.startswith("linux"):
        model_ckpt = "/home/richard/.cache/huggingface/hub/models--bert-base-uncased/"
    else:
        raise ValueError("Unsupported platform")

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    text = "time flies like an arrow"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    print(inputs.input_ids)

    config = AutoConfig.from_pretrained(model_ckpt)
    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    print(token_emb)

    inputs_embeds = token_emb(inputs.input_ids)
    print(inputs_embeds.size())
