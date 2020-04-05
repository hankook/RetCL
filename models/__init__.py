import dgl
import math
import torch
import torch.nn as nn

from .utils import *
from .sequence import *
from .graph import *
from .contrast import *


class GraphModule(nn.Module):
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


def graph_parallel(module, batch):
    device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) == 1:
        return module(batch)

    graphs = dgl.unbatch(batch)
    size = math.ceil(len(graphs) / len(device_ids))
    batches = [dgl.batch(graphs[offset::len(device_ids)]).to(torch.device('cuda', i)) for offset, i in enumerate(device_ids)]

    if len(graphs) < len(device_ids):
        device_ids = device_ids[:len(graphs)]

    replicas = nn.parallel.replicate(module, device_ids, not torch.is_grad_enabled())
    outputs = nn.parallel.parallel_apply(replicas, batches, devices=device_ids)
    return nn.parallel.gather(outputs, torch.device('cuda', 0), dim=0)
