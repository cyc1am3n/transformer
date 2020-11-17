import torch
import torch.nn as nn
from torch.nn import Functional as F

from ops import *

class Encoder(nn.Module):
    def __init__(self, data_config, model_config):
        self.n_layers = model_config['n_layers']
        self.d_model = model_config['d_model']
        self.n_heads = model_config['n_heads']
        self.d_k = model_config['d_k']
        self.d_v = model_config['d_v']
        self.d_ff = model_config['d_ff']
        self.max_len = model_config['max_len']
        self.dropout = model_config['dropout']
        self.n_vocab = data_config['vocab_size']

        self.embedding = Embedding(self.n_vocab, self.d_model, self.max_len, self.dropout)
        modules = [EncoderLayer(self.d_model, self.n_heads, self.d_k, self.d_v, self.d_ff, self.dropout) for _ in range(self.n_layers)]
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        x = self.embedding(x)
        out = self.sequential(x)
        return out
