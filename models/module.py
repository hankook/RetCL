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

    def __init__(self, encoder, *query_fn):
        super(GraphModule, self).__init__()
        self.encoder = encoder
        self.query_fn = nn.ModuleList(query_fn)

    def forward(self, batch):
        features = self.encoder(batch)
        features = torch.split(features, batch.batch_num_nodes)
        features, masks = pad_sequence(features)
        keys = masked_pooling(features, masks, mode='mean')

        queries = [fn(features, masks) for fn in self.query_fn]
        return [keys] + queries

    def construct_queries(self, products, reactants, embeddings):
        if len(embeddings) == 3:
            keys, p_queries, r_queries = embeddings
        else:
            keys, p_queries, r_queries, r2_queries = embeddings
        queries = []
        for p_idx, r_indices in zip(products, reactants):
            if p_idx is None:
                q = r_queries[r_indices].sum(0)
            elif len(r_indices) > 0:
                if len(embeddings) == 3:
                    q = p_queries[p_idx] - r_queries[r_indices].sum(0)
                else:
                    q = p_queries[p_idx] - r2_queries[r_indices].sum(0)
            else:
                q = p_queries[p_idx]
            queries.append(q)
        return torch.stack(queries, 0)

    def construct_keys(self, embeddings):
        return embeddings[0]

class GraphModuleV0(EmbeddingModule):

    def __init__(self, encoder, *branches, use_label=False, use_sum=False):
        super(GraphModuleV0, self).__init__()
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
        if len(embeddings) == 3:
            keys, p_queries, r_queries = embeddings
        else:
            keys, p_queries, r_queries, r2_queries = embeddings
        queries = []
        for p_idx, r_indices in zip(products, reactants):
            if p_idx is None:
                q = r_queries[r_indices].sum(0)
            elif len(r_indices) > 0:
                if len(embeddings) == 3:
                    q = p_queries[p_idx] - r_queries[r_indices].sum(0)
                else:
                    q = p_queries[p_idx] - r2_queries[r_indices].sum(0)
            else:
                q = p_queries[p_idx]
            queries.append(q)
        if labels is None or not self.use_label:
            return torch.stack(queries, 0)
        else:
            return torch.stack(queries, 0) + self.bias(labels-1).mul(10)

    def construct_keys(self, embeddings):
        return embeddings[0]

class GraphModuleV2(EmbeddingModule):

    def __init__(self, base_encoder, p_encoder, r_encoder, *query_fn):
        super(GraphModuleV2, self).__init__()
        self.base_encoder = base_encoder
        self.p_encoder = p_encoder
        self.r_encoder = r_encoder
        self.query_fn = nn.ModuleList(query_fn)

    def forward(self, batch):
        features = self.base_encoder(batch)

        p_features = self.p_encoder(batch, features)
        p_features = torch.split(p_features, batch.batch_num_nodes)
        p_features, p_masks = pad_sequence(p_features)
        p_queries = self.query_fn[0](p_features, p_masks)

        r_features = self.r_encoder(batch, features)
        r_features = torch.split(r_features, batch.batch_num_nodes)
        r_features, r_masks = pad_sequence(r_features)
        r_queries = self.query_fn[1](r_features, r_masks)

        features = torch.split(features, batch.batch_num_nodes)
        features, masks = pad_sequence(features)
        keys = masked_pooling(features, masks, mode='mean')

        return keys, p_queries, r_queries

    def construct_queries(self, products, reactants, embeddings):
        if len(embeddings) == 3:
            keys, p_queries, r_queries = embeddings
        else:
            keys, p_queries, r_queries, r2_queries = embeddings
        queries = []
        for p_idx, r_indices in zip(products, reactants):
            if p_idx is None:
                q = r_queries[r_indices].sum(0)
            elif len(r_indices) > 0:
                if len(embeddings) == 3:
                    q = p_queries[p_idx] - r_queries[r_indices].sum(0)
                else:
                    q = p_queries[p_idx] - r2_queries[r_indices].sum(0)
            else:
                q = p_queries[p_idx]
            queries.append(q)
        return torch.stack(queries, 0)

    def construct_keys(self, embeddings):
        return embeddings[0]

class GraphModuleV3(EmbeddingModule):

    def __init__(self, base_encoder, p_encoder, r_encoder, *query_fn):
        super(GraphModuleV3, self).__init__()
        self.base_encoder = base_encoder
        self.p_encoder = p_encoder
        self.r_encoder = r_encoder
        self.query_fn = nn.ModuleList(query_fn)

    def forward(self, batch):
        features = self.base_encoder(batch)

        p_features = self.p_encoder(batch)
        p_features = torch.split(p_features, batch.batch_num_nodes)
        p_features, p_masks = pad_sequence(p_features)
        p_queries = self.query_fn[0](p_features, p_masks)

        r_features = self.r_encoder(batch)
        r_features = torch.split(r_features, batch.batch_num_nodes)
        r_features, r_masks = pad_sequence(r_features)
        r_queries = self.query_fn[1](r_features, r_masks)

        features = torch.split(features, batch.batch_num_nodes)
        features, masks = pad_sequence(features)
        keys = masked_pooling(features, masks, mode='mean')

        return keys, p_queries, r_queries

    def construct_queries(self, products, reactants, embeddings):
        if len(embeddings) == 3:
            keys, p_queries, r_queries = embeddings
        else:
            keys, p_queries, r_queries, r2_queries = embeddings
        queries = []
        for p_idx, r_indices in zip(products, reactants):
            if p_idx is None:
                q = r_queries[r_indices].sum(0)
            elif len(r_indices) > 0:
                if len(embeddings) == 3:
                    q = p_queries[p_idx] - r_queries[r_indices].sum(0)
                else:
                    q = p_queries[p_idx] - r2_queries[r_indices].sum(0)
            else:
                q = p_queries[p_idx]
            queries.append(q)
        return torch.stack(queries, 0)

    def construct_keys(self, embeddings):
        return embeddings[0]

class GraphModuleV4(EmbeddingModule):

    def __init__(self, encoder, num_hidden_features, K):
        super(GraphModuleV4, self).__init__()
        self.encoder = encoder
        self.reaction_embeddings = nn.Parameter(torch.randn(K, num_hidden_features))

    def forward(self, batch):
        features = self.encoder(batch)
        features = torch.split(features, batch.batch_num_nodes)
        features, masks = pad_sequence(features)
        features = masked_pooling(features, masks, mode='mean')

        return features

    def construct_queries(self, products, reactants, features):
        queries = []
        for p_idx, r_indices in zip(products, reactants):
            if p_idx is None:
                q = features[r_indices].sum(0, keepdim=True)
                q = q - self.reaction_embeddings
            elif len(r_indices) > 0:
                q = features[p_idx:p_idx+1] - features[r_indices].sum(0, keepdim=True)
                q = q + self.reaction_embeddings
            else:
                q = features[p_idx:p_idx+1]
                q = q + self.reaction_embeddings
            queries.append(q)
        return torch.stack(queries, 0)

    def construct_keys(self, features):
        return features

class GraphModuleV5(EmbeddingModule):

    def __init__(self, encoder, *query_fn):
        super(GraphModuleV5, self).__init__()
        self.encoder = encoder
        self.query_fn = nn.ModuleList(query_fn)
        self.weight_fn = nn.Linear(256, 10)

    def forward(self, batch):
        features = self.encoder(batch)
        weights = F.sigmoid(self.weight_fn(features))

        features = torch.split(features, batch.batch_num_nodes)
        weights = torch.split(weights, batch.batch_num_nodes)

        keys = []
        for f, w in zip(features, weights):
            k = (f.unsqueeze(1) * w.unsqueeze(2)).sum(0)
            k = k.div(w.sum(0).unsqueeze(1))
            keys.append(k)
        keys = torch.stack(keys, 0)

        features, masks = pad_sequence(features)
        queries = [fn(features, masks) for fn in self.query_fn]
        return [keys] + queries

    def construct_queries(self, products, reactants, embeddings):
        if len(embeddings) == 3:
            keys, p_queries, r_queries = embeddings
        else:
            keys, p_queries, r_queries, r2_queries = embeddings
        queries = []
        for p_idx, r_indices in zip(products, reactants):
            if p_idx is None:
                q = r_queries[r_indices].sum(0)
            elif len(r_indices) > 0:
                if len(embeddings) == 3:
                    q = p_queries[p_idx] - r_queries[r_indices].sum(0)
                else:
                    q = p_queries[p_idx] - r2_queries[r_indices].sum(0)
            else:
                q = p_queries[p_idx]
            queries.append(q)
        return torch.stack(queries, 0)

    def construct_keys(self, embeddings):
        return embeddings[0]

