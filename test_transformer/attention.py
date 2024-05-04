import sys
import math
import torch
import numpy as np
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
        scale = math.sqrt(query.size(-1))
        scores = torch.bmm(query, key.transpose(1, 2)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float("inf"))
        weights = self.softmax(scores)
        return torch.bmm(weights, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None, query_mask=None, key_mask=None):
        if query_mask is not None and key_mask is not None:
            mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
        x = torch.cat([h(query, key, value, mask=mask) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x, mask=None):
        # apply layer norm and then copy input into query, key, value
        hidden_state = self.layer_norm1(x)

        # apply self attention with a skip connection
        x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)

        # apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        # create position IDs for input sequence
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # Create token and position embedings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def GetPositionEncoding(seq_len, dim, n=10000):
    assert dim % 2 == 0
    PE = np.zeros(seq_len, dim)
    for pos in range(seq_len):
        for i in range(dim // 2):
            denominator = np.power(n, 2 * i / dim)
            PE[pos, 2 * i] = np.sin(pos / denominator)
            PE[pos, 2 * i + 1] = np.cos(pos / denominator)

    return PE


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

    encoder_layer = TransformerEncoderLayer(config)
    print(encoder_layer(inputs_embeds).size())

    embedding_layer = Embeddings(config)
    print(embedding_layer(inputs.input_ids).size())
