import math, torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResidualLayer(nn.ModuleList):

    def __init__(self, num_channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.BatchNorm1d(num_channels),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels, num_channels),
                nn.BatchNorm1d(num_channels),
                nn.ReLU(inplace=True),
                nn.Linear(num_channels, num_channels)))
        super(ResidualLayer, self).__init__(layers)

    def forward(self, x):
        for layer in self:
            x = x+layer(x)
        return x

class ResidualFeedforwardLayer(nn.Module):

    def __init__(self, in_channels, intermediate_channels=None, out_channels=None, num_candidates=1, dropout=0.1):
        super(ResidualFeedforwardLayer, self).__init__()
        
        if intermediate_channels is None:
            intermediate_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.num_candidates = num_candidates

        self.layer1 = nn.Linear(in_channels, intermediate_channels)
        self.layer2 = nn.Linear(intermediate_channels, out_channels*num_candidates)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(out_channels*num_candidates)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.dropout(self.layer2(self.dropout(self.relu(self.layer1(x)))))
        if self.num_candidates > 1:
            sizes = [1] * (x.ndim-1) + [self.num_candidates]
            x = x.repeat(*sizes)
        return self.norm(x+y)


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
                self.branches.append(nn.Identity())


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


class Encoder(nn.Module):

    def __init__(self,
                 vocab_size=None,
                 hidden_size=256,
                 num_attention_heads=8,
                 intermediate_size=2048,
                 num_layers=4,
                 dropout=0.1,
                 activation='gelu'):
        super(Encoder, self).__init__()

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

        self.base = nn.TransformerEncoder(layer, num_layers)

    def forward(self, inputs, masks, pooling=None, residual=False):
        """
        arguments:
            inputs: B x T x C
            masks:  B x T     (0 => padding)

        return:
            outputs: B x C
        """
        inputs = inputs.transpose(0, 1)
        padding_mask = ~masks
        outputs = self.base(self.embedding(inputs), src_key_padding_mask=padding_mask)
        if residual:
            outputs = inputs + outputs

        if pooling == 'sum':
            masks = masks.t().float().unsqueeze(-1)
            outputs = outputs.mul(masks).sum(0)
        elif pooling == 'mean':
            masks = masks.t().float().unsqueeze(-1)
            outputs = outputs.mul(masks).mean(0)
        else:
            outputs = outputs.transpose(0, 1)

        return outputs


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Decoder(nn.Module):

    def __init__(self,
                 hidden_size=256,
                 num_attention_heads=8,
                 intermediate_size=2048,
                 num_layers=3,
                 dropout=0.1,
                 activation='gelu'):
        super(Decoder, self).__init__()

        layer = nn.TransformerDecoderLayer(d_model=hidden_size,
                                           nhead=num_attention_heads,
                                           dim_feedforward=intermediate_size,
                                           dropout=dropout,
                                           activation=activation)

        self.base = nn.TransformerDecoder(layer, num_layers)

    def forward(self, inputs, memory, masks):
        """
        arguments:
            inputs: B x T x C
            memory: B x C
            masks:  B x T     (0 => padding)

        return:
            outputs: B x T x C
        """

        padding_mask = ~masks
        tgt_mask = _generate_square_subsequent_mask(inputs.shape[1]).to(inputs.device)
        outputs = self.base(inputs.transpose(0, 1), memory.unsqueeze(0),
                            tgt_key_padding_mask=padding_mask,
                            tgt_mask=tgt_mask)

        return outputs.transpose(0, 1)

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


