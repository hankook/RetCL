import os, argparse, logging, math, random, time

import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from datasets import load_dataset, build_dataloader, _canonicalize_smiles
from models import Structure2Vec, MultiAttentionQuery, AttentionSimilarity, GraphModule, SimCLRv2
from models import masked_pooling, pad_sequence, graph_parallel
from models.score import PermutationScore
from dgl.data.chem import smiles_to_bigraph, CanonicalBondFeaturizer, CanonicalAtomFeaturizer

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def main(args):
    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='x')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='w')

    encoder = Structure2Vec(num_layers=5,
                            num_hidden_features=256,
                            num_atom_features=atom_featurizer.feat_size('x'),
                            num_bond_features=bond_featurizer.feat_size('w'))

    module = GraphModule(encoder,
                         MultiAttentionQuery(256, args.K),
                         MultiAttentionQuery(256, args.K)).to(device)
    module.halt_keys = nn.Parameter(torch.randn(1, 256).to(device))

    ### LOSS
    sim_fn = AttentionSimilarity()
    loss_fn = SimCLRv2(sim_fn, 0.1).to(device)
    score_fn = PermutationScore(sim_fn, loss_fn)

    module.eval()
    module.load_state_dict(torch.load(args.ckpt)['module'])

    products = []
    reactants = []
    with open(args.input_file) as f_x, open(args.target_file) as f_y:
        for x, y in zip(f_x.readlines(), f_y.readlines()):
            products.append(_canonicalize_smiles(''.join(x.split())))
            reactants.append([_canonicalize_smiles(s) for s in ''.join(y.split()).split('.')])
            
    preds = []
    with open(args.pred_file) as f:
        for pred in f.readlines():
            try:
                preds.append([_canonicalize_smiles(x) for x in ''.join(pred.split()).split(',')[0].split('.')])
            except:
                preds.append([])

    n = len(preds) // len(products)
    num_corrects = torch.zeros(n)
    for i, (x, y) in enumerate(zip(products, reactants)):
        graphs = [smiles_to_bigraph(x,
                                    node_featurizer=atom_featurizer,
                                    edge_featurizer=bond_featurizer)]
        p_indices = []
        r_indices = []
        for j in range(i*n, (i+1)*n):
            p_indices.append(0)
            r_indices.append([])
            try:
                for s in preds[j]:
                    assert len(s) > 0
                    graphs.append(smiles_to_bigraph(
                        s,
                        node_featurizer=atom_featurizer,
                        edge_featurizer=bond_featurizer))
                    r_indices[-1].append(len(graphs)-1)
            except:
                p_indices.pop(-1)
                r_indices.pop(-1)

        if len(p_indices) == 0:
            continue

        batch = dgl.batch(graphs).to(device)
        keys, p_queries, r_queries = module(batch)
        keys = torch.cat([keys, module.halt_keys], 0)

        scores = score_fn.compute_scores(
                (keys, p_queries, r_queries),
                (p_indices, r_indices))

        _, indices = torch.sort(scores, dim=0, descending=True)
        correct = False
        for k, idx in enumerate(indices.tolist()):
            pred = [_canonicalize_smiles(s) for s in preds[i*n+idx]]
            if set(pred) == set(y):
                correct = True
            if correct:
                num_corrects[k] += 1
        if correct:
            for l in range(k+1, n):
                num_corrects[l] += 1
        print('{} / {}'.format(i, len(products)), end='\r')
    print()
    print(num_corrects / len(products))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--target-file', type=str, required=True)
    parser.add_argument('--pred-file', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--K', type=int, default=2)
    args = parser.parse_args()

    main(args)

