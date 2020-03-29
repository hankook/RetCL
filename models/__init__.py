import dgl
import torch.nn as nn

from .utils import *
from .sequence import *
from .graph import *
from .contrast import *


class GraphModule(nn.Module):
    def __init__(self, encoder, prod_query_fn, reac_query_fn):
        super(GraphModule, self).__init__()
        self.encoder = encoder
        self.prod_query_fn = prod_query_fn
        self.reac_query_fn = reac_query_fn

    def forward(self, batch):
        features = self.encoder(batch)
        features = torch.split(features, batch.batch_num_nodes)
        features, masks = pad_sequence(features)
        keys = masked_pooling(features, masks, mode='mean')

        prod_queries = self.prod_query_fn(features, masks)
        reac_queries = self.reac_query_fn(features, masks)

        return keys, prod_queries, reac_queries

