import os, argparse, logging, math, random, copy, time, itertools

import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from datasets import load_dataset, build_dataloader
from models import Structure2Vec, MultiAttentionQuery, AttentionSimilarity, GraphModule, SimCLRv2
from models import masked_pooling, pad_sequence, graph_parallel

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def compute_final_scores(product, reactants, score_fn):
    # backward
    scores = []
    for perm in itertools.permutations(list(range(len(reactants)))):
        queries, _ = product
        s = 0
        for i in perm:
            r_queries, r_keys = reactants[i]
            s += score_fn(queries, r_keys, many_to_many=False).div(0.1)
            queries -= r_queries
        scores.append(s)
    scores = torch.stack(scores, 1).logsumexp(1)

    # forward
    queries = 0
    _, p_keys = product
    for r_queries, _ in reactants:
        queries += r_queries
    scores += score_fn(queries, p_keys, many_to_many=False).div(0.1)
    return scores


def main(args):
    ### DATASETS
    datasets = load_dataset(args.dataset, args.datadir)

    ### DATALOADERS
    dataloader = build_dataloader(datasets['test'], batch_size=args.batch_size)
    molloader  = build_dataloader(datasets['known_mol_dict'], batch_size=args.batch_size)

    ### MODELS
    print('Loading model ...')
    encoder = Structure2Vec(num_layers=5,
                            num_hidden_features=256,
                            num_atom_features=datasets['mol_dict'].atom_feat_size,
                            num_bond_features=datasets['mol_dict'].bond_feat_size)

    module = GraphModule(encoder,
                         MultiAttentionQuery(256, args.K),
                         MultiAttentionQuery(256, args.K)).to(device)
    score_fn = AttentionSimilarity()
    print('- # of parameters: {}'.format(sum(p.numel() for p in module.parameters())))

    ckpt = torch.load(args.ckpt, map_location='cpu')
    module.load_state_dict(ckpt['module'])

    module.eval()
    num_corrects = torch.zeros(args.topk)
    with torch.no_grad():
        r_keys = []
        r_queries = []
        for iteration, molecules in enumerate(molloader):
            print('Compute features for reactants ... {} / {}'.format(iteration, len(molloader)), end='\r')
            batch = dgl.batch([m.graph for m in molecules]).to(device)
            keys, _, queries = graph_parallel(module, batch)
            r_keys.append(keys.detach())
            r_queries.append(queries.detach())
        r_keys = torch.cat(r_keys, 0).detach()
        r_queries = torch.cat(r_queries, 0).detach()
        print('Compute features for reactants ... done')

        for iteration, reactions in enumerate(dataloader):
            print('Compute features for reactants ... {} / {}'.format(iteration, len(dataloader)), end='\r')
            N = len(reactions)
            batch = dgl.batch([r.product.graph for r in reactions]).to(device)
            p_keys, p_queries, _ = graph_parallel(module, batch)

            scores = torch.cat([score_fn(p_queries, k).detach() for k in r_keys.split(512)], dim=1)
            scores1, indices1 = scores.topk(args.beam, dim=1)

            r_queries1 = r_queries[indices1.view(-1)]
            r_keys1 = r_keys[indices1.view(-1)]

            final_scores1 = compute_final_scores(
                    (p_queries.repeat_interleave(args.beam, 0), p_keys.repeat_interleave(args.beam, 0)),
                    [(r_queries1, r_keys1)],
                    score_fn)
            final_scores1, final_indices1 = final_scores1.view(N, args.beam).topk(args.topk, dim=1)


            p_queries2 = p_queries.repeat_interleave(args.beam, dim=0) - r_queries1
            scores = torch.cat([score_fn(p_queries2, k).detach() for k in r_keys.split(512)], dim=1)
            scores2, indices2 = scores.topk(args.beam, dim=1)

            r_queries2 = r_queries[indices2.view(-1)]
            r_keys2 = r_keys[indices2.view(-1)]

            final_scores2 = compute_final_scores(
                    (p_queries.repeat_interleave(args.beam**2, 0), p_keys.repeat_interleave(args.beam**2, 0)),
                    [(r_queries1.repeat_interleave(args.beam, 0), r_keys1.repeat_interleave(args.beam, 0)),
                     (r_queries2, r_keys2)],
                    score_fn)
            final_scores2, final_indices2 = final_scores2.view(N, args.beam**2).topk(args.topk, dim=1)

            indices2 = indices2.view(N, -1)
            for i, r in enumerate(reactions):
                correct = False
                for j in range(args.topk):
                    if len(r.reactants) == 1:
                        k = final_indices1[i, j].item()
                        k = indices1[i, k].item()
                        m = datasets['known_mol_dict'][k]
                        if m.smiles == r.reactants[0].smiles:
                            correct = True
                    elif len(r.reactants) == 2:
                        k = final_indices2[i, j].item()
                        k1 = indices1[i, k // args.beam].item()
                        k2 = indices2[i, k].item()
                        m1 = datasets['known_mol_dict'][k1]
                        m2 = datasets['known_mol_dict'][k2]
                        if set([m1, m2]) == set(r.reactants):
                            correct = True

                    if correct:
                        num_corrects[j] += 1
        print('Compute features for reactants ... done')

    print(num_corrects / len(datasets['test']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--datadir', type=str, default='data/uspto50k_coley')
    parser.add_argument('--dataset', type=str, default='uspto50k')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    main(args)


