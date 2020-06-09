import dgl
import torch
import logging
import torch.nn as nn

from .utils import ResidualLayer
from .graph import Structure2Vec
from .module import GraphModule

from datasets import Molecule

logger = logging.getLogger('module')

def load_encoder(args):
    encoder = Structure2Vec(num_layers=args.num_layers,
                            num_hidden_features=256,
                            num_atom_features=Molecule.atom_feat_size,
                            num_bond_features=Molecule.bond_feat_size)
    logger.info('- # of encoder parameters: {}'.format(sum(p.numel() for p in encoder.parameters())))

    return encoder

def load_module(args):
    logger.info('Loading Module ...')

    encoder = load_encoder(args)
    branches = [ResidualLayer(256) for _ in range(3)]

    module = GraphModule(encoder, *branches, use_label=args.use_label, use_sum=args.use_sum)
    module.halt_keys = nn.Parameter(torch.randn(1, 256))

    logger.info('- # of module parameters: {}'.format(sum(p.numel() for p in module.parameters())))
    return module

