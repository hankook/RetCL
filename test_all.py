import os, argparse, logging, math, random, copy, time, itertools

import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

import utils
from datasets import load_dataset, build_dataloader
from models import Structure2Vec, MultiAttentionQuery, AttentionSimilarity, GraphModule, SimCLRv2
from models import masked_pooling, pad_sequence, graph_parallel

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')


def select(p_keys, p_queries, r_keys, r_queries, beam=5, best=5, max_reactants=3):
    N = p_queries.shape[0]
    M = r_keys.shape[0]

    final_predictions = [] #[[] for _ in range(N)]
    final_scores = [] #[[] for _ in range(N)]

    curr_predictions = [(i, ) for i in range(N)]
    curr_scores = torch.tensor([0. for _ in range(N)], device=device)
    curr_queries = p_queries
    for n in range(max_reactants+1):
        scores = torch.cat([score_fn(curr_queries, k).detach() for k in r_keys.split(512)], dim=1)
        scores = scores + curr_scores.view(-1, 1)

        new_queries = []
        new_predictions = []
        new_scores = []
        for i in range(N):
            masks = [pred[0] == i for pred in curr_predictions]
            rows = [j for j, mask in enumerate(masks) if mask]
            if len(rows) == 0:
                continue
            ith_scores, ith_indices = scores[rows].view(-1).topk(beam, dim=0)
            for j in range(beam):
                row = rows[ith_indices[j] // M]
                col = ith_indices[j].item() % M

                if col == M-1:
                    if n > 0:
                        final_predictions.append(curr_predictions[row])
                        final_scores.append(ith_scores[j].item() / n)
                else:
                    new_predictions.append(curr_predictions[row] + (col, ))
                    new_queries.append(curr_queries[row] - r_queries[col])
                    new_scores.append(ith_scores[j])

        curr_queries = torch.stack(new_queries, 0)
        curr_predictions = new_predictions
        curr_scores = torch.stack(new_scores, 0)

    # p_keys = p_keys[[pred[0] for pred in final_predictions]]
    # r_queries = torch.stack([r_queries[list(pred[1:])].sum(0) for pred in final_predictions])
    # final_scores = [x+y for x, y in zip(final_scores, score_fn(r_queries, p_keys, many_to_many=False).squeeze(1).tolist())]
    return final_predictions, final_scores

def main(args):
    ### DATASETS
    datasets = load_dataset(args.dataset, args.datadir)
    mol_dict = datasets['test_mol_dict']
    # mol_dict = datasets['known_mol_dict']

    ### DATALOADERS
    dataloader = build_dataloader(datasets['test'], batch_size=args.batch_size)
    molloader  = build_dataloader(mol_dict, batch_size=args.batch_size)

    ### MODELS
    print('Loading model ...')
    encoder = Structure2Vec(num_layers=5,
                            num_hidden_features=256,
                            num_atom_features=datasets['mol_dict'].atom_feat_size,
                            num_bond_features=datasets['mol_dict'].bond_feat_size)

    module = GraphModule(encoder,
                         MultiAttentionQuery(256, args.K),
                         MultiAttentionQuery(256, args.K)).to(device)
    module.halt_keys = nn.Parameter(torch.randn(1, 256).to(device))
    score_fn = AttentionSimilarity()
    print('- # of parameters: {}'.format(sum(p.numel() for p in module.parameters())))

    ### Load from checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    module.load_state_dict(ckpt['module'])
    print('- from ckpt {} @ {}'.format(args.ckpt, ckpt['iteration']))

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
        r_keys.append(module.halt_keys)
        r_keys = torch.cat(r_keys, 0).detach()
        r_queries = torch.cat(r_queries, 0).detach()
        print('Compute features for reactants ... done')

        for iteration, reactions in enumerate(dataloader):
            print('Evaluate retrosynthesis ... {} / {}'.format(iteration, len(dataloader)), end='\r')
            N = len(reactions)
            batch = dgl.batch([r.product.graph for r in reactions]).to(device)
            p_keys, p_queries, _ = graph_parallel(module, batch)

            predictions, scores = select(p_keys, p_queries, r_keys, r_queries, beam=args.beam, best=args.topk, max_reactants=3)
            idx2predictions = defaultdict(list)
            idx2scores = defaultdict(list)
            for pred, score in zip(predictions, scores):
                idx2predictions[pred[0]].append(pred)
                idx2scores[pred[0]].append(score)

            # print(predictions, scores)
            for i, r in enumerate(reactions):
                preds_with_scores = sorted(list(zip(idx2predictions[i], idx2scores[i])), key=lambda x: -x[1])
                # preds_with_scores = list(zip(predictions[i], scores[i]))
                correct = False
                for k, (pred, _) in enumerate(preds_with_scores):
                    mols = set([mol_dict[idx] for idx in pred[1:]])
                    if mols == set(r.reactants):
                        correct = True
                    if correct and k < args.topk:
                        num_corrects[k] += 1

        print('Evaluate retrosynthesis  ... done')

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

