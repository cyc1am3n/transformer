import math

import torch
import torch.nn as nn
from torch.nn import Functional as F

class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float).reshape(-1, 1)
        div = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float) / d_model)
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position / div)
        self.pe[:, 1::2] = torch.cos(position / div)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EmbeddingLayer(nn.Module):
    def __init__(self, n_vocab, embed_dim, max_len, dropout=0.1):
        super().__init__()
        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.pe = PositionalEncodingLayer(d_model, max_len, dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        return self.pe(x)

class ScaledDotProductAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, mask=None):
        super().__init__()
        self.mask = mask
        self.q_linear = nn.Linear(d_model, d_k)
        self.k_linear = nn.Linear(d_model, d_k)
        self.v_linear = nn.Linear(d_model, d_v)
        
    def forward(self, x):
        q = q_linear(x) # N x L x d_k
        k = k_linear(x) # N x L x d_k
        v = v_linear(x) # N x L x d_v
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d_k) # N x L x L
        
        if mask is not None:
            mask = mask.reshape(-1, 1)
            scores = scores.masked_fill(mask==0, 0)
            
        scores = F.softmax(scores, dim=2) # N x L x L
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v) # N x L x d_v
        return output

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, mask=None, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = nn.Droppout(p=dropout)
        self.multihead = [ScaledDotProductAttentionLayer(d_model, d_k, d_v, mask) for _ in range(n_heads)]
        self.linear = nn.Linear(n_heads * d_v, d_model)
        
    def forward(self, x):
        head_outputs = [self.multihead[i](x) for i in range(self.n_heads)]
        concat = torch.cat(head_outputs, dim=2) # N x L x (d_v * n_heads)
        output = self.linear(concat) # N x L x d_model
        return output

class PositionwiseFFLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.linear1(x) # N x L x d_ff
        out = self.linear2(F.relu(x)) # N x L x d_model
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, dropout=0.1):
        super().__init__()
        self.multihead = MultiHeadAttentionLayer(n_heads, d_model, d_k, d_v, dropout)
        self.layernorm = nn.LayerNorm(d_model) # d_model vs. [max_len, d_model]?
        self.ffn = PositionwiseFFLayer(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.layernorm(x + self.dropout(self.multihead(x)))
        x = self.layernorm(x + self.dropout(self.ffn(x)))
        return x
