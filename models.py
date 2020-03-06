import math, torch
import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x, **kwargs):
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        p = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(-1, x.size(1))
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self,
                 vocab_size=None,
                 hidden_size=256,
                 num_attention_heads=8,
                 intermediate_size=2048,
                 num_base_layers=3,
                 num_layers_per_branch=1,
                 num_branches=2,
                 dropout=0.1,
                 activation='gelu'):
        super(Model, self).__init__()

        if vocab_size is None:
            self.embedding = lambda x: x
        else:
            self.embedding = nn.Sequential(
                    nn.Embedding(vocab_size, hidden_size),
                    PositionalEncoding(hidden_size))

        layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                           nhead=num_attention_heads,
                                           dim_feedforward=intermediate_size,
                                           dropout=dropout,
                                           activation=activation)

        self.base = nn.TransformerEncoder(layer, num_base_layers)
        self.branches = nn.ModuleList()
        for _ in range(num_branches):
            if num_layers_per_branch > 0:
                self.branches.append(nn.TransformerEncoder(layer, num_layers_per_branch))
            else:
                self.branches.append(Identity())


    def forward(self, inputs, masks, pooling=None):
        # remove padding
        for i, mask in enumerate(masks.any(1)):
            if not mask.item():
                inputs = inputs[:, :i]
                masks = masks[:, :i]
                break

        inputs = inputs.transpose(0, 1)
        padding_mask = ~masks
        outputs = self.base(self.embedding(inputs), src_key_padding_mask=padding_mask)
        branch_outputs = [b(outputs, src_key_padding_mask=padding_mask) for b in self.branches]

        outputs = [outputs] + branch_outputs
        if pooling == 'sum':
            masks = masks.t().float().unsqueeze(-1)
            outputs = [o.mul(masks).sum(0) for o in outputs]
        elif pooling == 'mean':
            masks = masks.t().float().unsqueeze(-1)
            outputs = [o.mul(masks).sum(0).div(masks.sum(0)) for o in outputs]
        else:
            outputs = [o.transpose(0, 1) for o in outputs]

        return outputs



def load_embedding(**kwargs):
    return nn.Sequential(nn.Embedding(kwargs['vocab_size'], kwargs['hidden_size']),
                         PositionalEncoding(kwargs['hidden_size']))


def load_encoder(**kwargs):
    encoder_layer = nn.TransformerEncoderLayer(d_model=kwargs['hidden_size'],
                                               nhead=kwargs['num_attention_heads'],
                                               dim_feedforward=kwargs['intermediate_size'],
                                               dropout=kwargs.get('dropout', 0.1),
                                               activation=kwargs.get('activation', 'gelu'))

    return nn.TransformerEncoder(encoder_layer,
                                 kwargs['num_hidden_layers'])

