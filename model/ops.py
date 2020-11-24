import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncodingLayer(nn.Module):
    def __init__(self, embed_dim, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float).reshape(-1, 1)
        div = torch.pow(10000, torch.arange(0, embed_dim, 2, dtype=torch.float) / embed_dim)
        
        pe = torch.zeros(max_len, embed_dim)
        if torch.cuda.is_available():
            pe = pe.cuda()
        pe[:, 0::2] = torch.sin(position / div)
        pe[:, 1::2] = torch.cos(position / div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EmbeddingLayer(nn.Module):
    def __init__(self, n_vocab, embed_dim, max_len, dropout=0.1):
        super().__init__()
        self.n_vocab = n_vocab
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.pe = PositionalEncodingLayer(embed_dim, max_len, dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        return self.pe(x)

class ScaledDotProductAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, is_mask=False):
        super().__init__()
        self.is_mask = is_mask
        self.d_k = d_k
        self.q_linear = nn.Linear(d_model, d_k)
        self.k_linear = nn.Linear(d_model, d_k)
        self.v_linear = nn.Linear(d_model, d_v)
        
    def forward(self, q_input, k_input, v_input):
        q = self.q_linear(q_input) # N x L x d_k
        k = self.k_linear(k_input) # N x L x d_k
        v = self.v_linear(v_input) # N x L x d_v
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k) # N x L x L
        
        if self.is_mask is not None:
            max_len = q_input.shape[1]
            mask = torch.tril(torch.ones(max_len, max_len), diagonal=0).unsqueeze(0) # 1 x L x L
            if torch.cuda.is_available():
                mask = mask.cuda()
            scores = scores.masked_fill(mask==0, -1.0e9)
            
        scores = F.softmax(scores, dim=2) # N x L x L
        output = torch.matmul(scores, v) # N x L x d_v
        return output

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, is_mask=False, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.multihead = nn.ModuleList([ScaledDotProductAttentionLayer(d_model, d_k, d_v, is_mask) for _ in range(n_heads)])
        self.linear = nn.Linear(n_heads * d_v, d_model)
        
    def forward(self, q_input, k_input, v_input):
        head_outputs = [self.multihead[i](q_input, k_input, v_input) for i in range(self.n_heads)]
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
        x = self.layernorm(x + self.dropout(self.multihead(x, x, x)))
        x = self.layernorm(x + self.dropout(self.ffn(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, dropout=0.1):
        super().__init__()
        self.masked_multihead = MultiHeadAttentionLayer(n_heads, d_model, d_k, d_v, is_mask=True, dropout=dropout)
        self.multihead = MultiHeadAttentionLayer(n_heads, d_model, d_k, d_v, dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFLayer(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, enc_out, x):
        x = self.layernorm(x + self.dropout(self.multihead(x, x, x)))
        x = self.layernorm(x + self.dropout(self.masked_multihead(x, enc_out, enc_out)))
        x = self.layernorm(x + self.dropout(self.ffn(x)))
        return x