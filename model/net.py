import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import *

class Encoder(nn.Module):
    def __init__(self, data_config, model_config):
        super().__init__()
        self.n_layers = model_config['n_layers']
        self.d_model = model_config['d_model']
        self.n_heads = model_config['n_heads']
        self.d_k = model_config['d_k']
        self.d_v = model_config['d_v']
        self.d_ff = model_config['d_ff']
        self.max_len = model_config['max_len']
        self.dropout = model_config['dropout']
        self.n_vocab = data_config['spm']['vocab_size']

        self.embedding = EmbeddingLayer(self.n_vocab, self.d_model, self.max_len, self.dropout)
        modules = [EncoderLayer(self.d_model, self.n_heads, self.d_k, self.d_v, self.d_ff, self.dropout) for _ in range(self.n_layers)]
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        x = self.embedding(x)
        out = self.sequential(x)
        return out

class Decoder(nn.Module):
    def __init__(self, data_config, model_config):
        super().__init__()
        self.n_layers = model_config['n_layers']
        self.d_model = model_config['d_model']
        self.n_heads = model_config['n_heads']
        self.d_k = model_config['d_k']
        self.d_v = model_config['d_v']
        self.d_ff = model_config['d_ff']
        self.max_len = model_config['max_len']
        self.dropout = model_config['dropout']
        self.n_vocab = data_config['spm']['vocab_size']

        self.embedding = EmbeddingLayer(self.n_vocab, self.d_model, self.max_len, self.dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.d_model, self.n_heads, self.d_k, self.d_v, self.d_ff, self.dropout) for _ in range(self.n_layers)])

        self.dropout = nn.Dropout(p=self.dropout)
        self.linear = nn.Linear(self.d_model, self.n_vocab)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_out, x):
        x = self.embedding(x)
        for i in range(self.n_layers):
            x = self.decoder_layers[i](enc_out, x)
        x = self.linear(self.dropout(x))
        out = self.softmax(x)
        return out

class Transformer(nn.Module):
    def __init__(self, data_config, model_config):
        super().__init__()
        self.encoder = Encoder(data_config, model_config)
        self.decoder = Decoder(data_config, model_config)

    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        prob = self.decoder(enc_out, tgt)
        return tgt
