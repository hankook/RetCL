import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pad_sequence, masked_pooling
from .graph import Structure2Vec, Structure2VecLayer

class EmbeddingModule(nn.Module):

    def forward(self, batch):
        raise NotImplementedError

    def construct_queries(self, products, reactants, embeddings):
        raise NotImplementedError

    def construct_keys(self, embeddings):
        raise NotImplementedError


class GraphModule(EmbeddingModule):

    def __init__(self, encoder, *branches, use_label=False, use_sum=False):
        super(GraphModule, self).__init__()
        self.encoder = encoder
        self.branches = nn.ModuleList(branches)
        self.use_label = use_label
        self.use_sum = use_sum
        if use_label:
            self.bias = nn.Embedding(20, encoder.num_hidden_features)
            self.bias.weight.data.zero_()

    def forward(self, batch):
        features = self.encoder(batch)
        features = torch.split(features, batch.batch_num_nodes)
        features, masks = pad_sequence(features)

        mode = 'sum' if self.use_sum else 'mean'
        p_queries = masked_pooling(self.branches[0](features), masks, mode=mode)
        r_queries = masked_pooling(self.branches[1](features), masks, mode=mode)
        keys      = masked_pooling(self.branches[2](features), masks, mode=mode)

        return [keys, p_queries, r_queries]

    def construct_queries(self, products, reactants, embeddings, labels=None):
        keys, p_queries, r_queries = embeddings
        queries = []
        for p_idx, r_indices in zip(products, reactants):
            if p_idx is None:
                q = r_queries[r_indices].sum(0)
            elif len(r_indices) > 0:
                q = p_queries[p_idx] - r_queries[r_indices].sum(0)
            else:
                q = p_queries[p_idx]
            queries.append(q)
        if labels is None or not self.use_label:
            return torch.stack(queries, 0)
        else:
            return torch.stack(queries, 0) + self.bias(labels-1).mul(10)

    def construct_keys(self, embeddings):
        return embeddings[0]

