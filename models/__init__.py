import dgl
import math
import torch
import logging
import torch.nn as nn

from .utils import *
from .sequence import *
from .graph import *
from .contrast import *
from .module import *
from .similarity import *

from datasets import Molecule

logger = logging.getLogger('module')

def load_encoder(
        name='s2v',
        num_layers=5,
        num_hidden_features=256):

    if name == 's2v':
        encoder = Structure2Vec(num_layers=num_layers,
                                num_hidden_features=num_hidden_features,
                                num_atom_features=Molecule.atom_feat_size,
                                num_bond_features=Molecule.bond_feat_size)
    elif name == 's2v-ours':
        encoder = Structure2VecOurs(num_layers=num_layers,
                                    num_hidden_features=num_hidden_features,
                                    num_atom_features=Molecule.atom_feat_size,
                                    num_bond_features=Molecule.bond_feat_size)
    elif name == 'attentionfp':
        encoder = AttentionFP(node_feat_size=Molecule.atom_feat_size,
                              edge_feat_size=Molecule.bond_feat_size,
                              num_layers=num_layers,
                              graph_feat_size=num_hidden_features)
    elif name == 'mpnn':
        encoder = MPNN(node_in_feats=Molecule.atom_feat_size,
                       edge_in_feats=Molecule.bond_feat_size,
                       node_out_feats=num_hidden_features,
                       num_step_message_passing=num_layers)
    elif name == 'wln':
        encoder = WLN(node_in_feats=Molecule.atom_feat_size,
                      edge_in_feats=Molecule.bond_feat_size,
                      node_out_feats=num_hidden_features,
                      n_layers=num_layers)
    elif name == 'edgewise':
        encoder = EdgewiseGNN(num_layers=num_layers,
                              num_hidden_features=num_hidden_features,
                              num_atom_features=Molecule.atom_feat_size,
                              num_bond_types=4)
    elif name == 'relgraph':
        encoder = RelGraphNN(num_layers=num_layers,
                             num_hidden_features=num_hidden_features,
                             num_atom_features=Molecule.atom_feat_size,
                             num_bond_types=4)

    logger.info('- # of encoder parameters: {}'.format(sum(p.numel() for p in encoder.parameters())))

    return encoder

def load_module(
        name='v1',
        encoder='s2v',
        num_layers=5,
        num_hidden_features=256,
        num_branches=2,
        K=2,
        num_halt_keys=1):

    logger.info('Loading Module ...')

    if name == 'v1':
        encoder = Structure2Vec(num_layers=num_layers,
                                num_hidden_features=num_hidden_features,
                                num_atom_features=Molecule.atom_feat_size,
                                num_bond_features=Molecule.bond_feat_size)
        query_fn = [MultiAttentionQuery(num_hidden_features, K) for _ in range(num_branches)]

        module = GraphModule(encoder, *query_fn)
        module.halt_keys = nn.Parameter(torch.randn(num_halt_keys, 256))
    elif name == 'v2':
        encoder = Structure2Vec(num_layers=num_layers,
                                num_hidden_features=num_hidden_features,
                                num_atom_features=Molecule.atom_feat_size,
                                num_bond_features=Molecule.bond_feat_size)
        p_encoder = Structure2VecLayer(num_hidden_features=num_hidden_features,
                                       num_atom_features=Molecule.atom_feat_size,
                                       num_bond_features=Molecule.bond_feat_size)
        r_encoder = Structure2VecLayer(num_hidden_features=num_hidden_features,
                                       num_atom_features=Molecule.atom_feat_size,
                                       num_bond_features=Molecule.bond_feat_size)
        query_fn = [MultiAttentionQuery(num_hidden_features, K) for _ in range(num_branches)]
        module = GraphModuleV2(encoder, p_encoder, r_encoder, *query_fn)
        module.halt_keys = nn.Parameter(torch.randn(num_halt_keys, 256))
    elif name == 'v3':
        encoder = Structure2Vec(num_layers=num_layers,
                                num_hidden_features=num_hidden_features,
                                num_atom_features=Molecule.atom_feat_size,
                                num_bond_features=Molecule.bond_feat_size)
        p_encoder = Structure2Vec(num_layers=num_layers,
                                  num_hidden_features=num_hidden_features,
                                  num_atom_features=Molecule.atom_feat_size,
                                  num_bond_features=Molecule.bond_feat_size)
        r_encoder = Structure2Vec(num_layers=num_layers,
                                  num_hidden_features=num_hidden_features,
                                  num_atom_features=Molecule.atom_feat_size,
                                  num_bond_features=Molecule.bond_feat_size)
        query_fn = [MultiAttentionQuery(num_hidden_features, K) for _ in range(num_branches)]
        module = GraphModuleV3(encoder, p_encoder, r_encoder, *query_fn)
        module.halt_keys = nn.Parameter(torch.randn(num_halt_keys, 256))
    elif name == 'v4':
        encoder = Structure2Vec(num_layers=num_layers,
                                num_hidden_features=num_hidden_features,
                                num_atom_features=Molecule.atom_feat_size,
                                num_bond_features=Molecule.bond_feat_size)
        module = GraphModuleV4(encoder, num_hidden_features, K)
        module.halt_keys = nn.Parameter(torch.randn(num_halt_keys, 256))
    elif name == 'v5':
        encoder = load_encoder(encoder, num_layers, num_hidden_features)
        query_fn = [MultiAttentionQuery(num_hidden_features, K) for _ in range(num_branches)]

        module = GraphModuleV5(encoder, *query_fn)
        module.halt_keys = nn.Parameter(torch.randn(num_halt_keys, 10, 256))

    logger.info('- # of module parameters: {}'.format(sum(p.numel() for p in module.parameters())))
    return module

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

