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

def main(args):
    os.makedirs(args.logdir)
    utils.set_logging_options(args.logdir)

    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))
    logger.info(args)

    summary_writer = SummaryWriter(args.logdir)

    ### DATASETS
    datasets = load_dataset(args.dataset, args.datadir)

    ### DATALOADERS
    trainloader = build_dataloader(datasets['train'], batch_size=args.batch_size, num_iterations=args.num_iterations)

    ### MODELS
    logger.info('Loading model ...')
    encoder = Structure2Vec(num_layers=5,
                            num_hidden_features=256,
                            num_atom_features=datasets['mol_dict'].atom_feat_size,
                            num_bond_features=datasets['mol_dict'].bond_feat_size)

    module = GraphModule(encoder,
                         MultiAttentionQuery(256, args.K),
                         MultiAttentionQuery(256, args.K)).to(device)
    module.halt_keys = nn.Parameter(torch.randn(1, 256).to(device))
    score_fn = AttentionSimilarity()

    ### LOSS
    loss_fn = SimCLRv2(score_fn, args.tau).to(device)

    logger.info('- # of parameters: {}'.format(sum(p.numel() for p in module.parameters())))
    if args.optimizer == 'adam':
        optimizer = optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(module.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.SGD(module.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    iteration = 0
    best_acc = 0
    for reactions in trainloader:

        iteration += 1
        module.train()

        # preprocessing
        N = len(reactions)
        batch_molecules = dict()
        graphs = []
        for reaction in reactions:
            for mol in [reaction.product] + reaction.reactants:
                if mol not in batch_molecules:
                    batch_molecules[mol] = len(batch_molecules)
                    graphs.append(mol.graph)

        p_indices = []
        r_indices = []
        num_reactants = []
        for reaction in reactions:
            p_indices.append(batch_molecules[reaction.product])
            r_indices.append([batch_molecules[r] for r in reaction.reactants])
            num_reactants.append(len(reaction.reactants))

        # compute features
        batch = dgl.batch(graphs).to(device)
        keys, p_queries, r_queries = module(batch)

        # forward prediction P(product | reactant)
        queries = [r_queries[r_indices[i]].sum(0) for i in range(N)]
        queries = torch.stack(queries, 0)
        f_loss, f_corrects = loss_fn(queries, keys, p_indices, r_indices)

        # backward prediction P(reactant | product)
        keys = torch.cat([keys, module.halt_keys], 0)
        queries = []
        labels = []
        ignore_indices = []
        lengths = []
        for i, n in enumerate(num_reactants):
            lengths.append(math.factorial(n)*(n+1))
            for perm in itertools.permutations(r_indices[i]):
                perm = perm + (keys.shape[0]-1, )
                queries.append(p_queries[p_indices[i]])
                labels.append(perm[0])
                ignore_indices.append(p_indices[i])
                for j in range(n):
                    queries.append(queries[-1]-r_queries[perm[j]])
                    labels.append(perm[j+1])
                    ignore_indices.append(p_indices[i])
        queries = torch.stack(queries, 0)

        b_loss, b_corrects = [torch.split(x, lengths) for x in loss_fn(queries, keys, labels, ignore_indices)]
        if not args.min:
            b_loss = torch.stack([x.view(-1, n+1).sum(1).mul(-1).logsumexp(0).mul(-1) for n, x in zip(num_reactants, b_loss)])
        else:
            b_loss = torch.stack([x.view(-1, n+1).sum(1).min(0)[0] for n, x in zip(num_reactants, b_loss)])
        b_corrects = torch.stack([x.view(-1, n+1).all(1).any(0) for n, x in zip(num_reactants, b_corrects)])

        loss = (f_loss + b_loss).mean()
        acc = (f_corrects & b_corrects).float().mean().item()

        # optimize
        optimizer.zero_grad()
        loss.backward()
        if args.clip is not None:
            nn.utils.clip_grad_norm_(models.parameters(), args.clip)
        optimizer.step()

        # logging
        logger.info('[Iter {}] [Loss {:.4f}] [BatchAcc {:.4f}]'.format(iteration, loss, acc))
        summary_writer.add_scalar('train/loss', loss.item(), iteration)
        summary_writer.add_scalar('train/batch_acc', acc, iteration)

        if iteration % args.eval_freq == 0:
            acc = evaluate(module, score_fn, datasets['val'], datasets['known_mol_dict'], batch_size=64)
            if best_acc < acc:
                logger.info(f'[Iter {iteration}] [Best! {acc:.4f}]')
                best_acc = acc
                torch.save({
                    'module': module.state_dict(),
                    'optim': optimizer.state_dict(),
                    'iteration': iteration,
                    'best_acc': best_acc,
                    'args': vars(args),
                }, os.path.join(args.logdir, 'best.pth'))

            logger.info(f'[Iter {iteration}] [Val Acc {acc:.4f}]')
            summary_writer.add_scalar('val/acc', acc, iteration)
            summary_writer.add_scalar('val/best', best_acc, iteration)


def evaluate(module, score_fn, dataset, mol_dict, batch_size=64):
    module.eval()
    dataloader = build_dataloader(dataset,  batch_size=batch_size)
    molloader  = build_dataloader(mol_dict, batch_size=batch_size)

    with torch.no_grad():
        r_keys = []
        r_queries = []
        for molecules in molloader:
            batch = dgl.batch([m.graph for m in molecules]).to(device)
            keys, _, queries = graph_parallel(module, batch)

            r_keys.append(keys.detach())
            r_queries.append(queries.detach())

        r_keys.append(module.halt_keys)
        r_queries.append(torch.zeros(1, *r_queries[-1].shape[1:]).to(device))

        r_keys = torch.cat(r_keys, 0).detach()
        r_queries = torch.cat(r_queries, 0).detach()

        num_corrects = 0
        for reactions in dataloader:
            N = len(reactions)
            batch = dgl.batch([r.product.graph for r in reactions]).to(device)
            p_keys, p_queries, _ = graph_parallel(module, batch)

            scores = torch.cat([score_fn(p_queries, k).detach() for k in r_keys.split(512)], dim=1)
            indices1 = scores.argmax(1)

            p_queries = p_queries - r_queries[indices1]
            scores = torch.cat([score_fn(p_queries, k).detach() for k in r_keys.split(512)], dim=1)
            indices2 = scores.argmax(1)

            p_queries = p_queries - r_queries[indices2]
            scores = torch.cat([score_fn(p_queries, k).detach() for k in r_keys.split(512)], dim=1)
            indices3 = scores.argmax(1)

            for i, r in enumerate(reactions):
                if indices2[i].item() == r_keys.shape[0]-1:
                    pred = [indices1[i].item()]
                elif indices3[i].item() == r_keys.shape[0]-1:
                    pred = [indices1[i].item(), indices2[i].item()]
                else:
                    pred = [indices1[i].item(), indices2[i].item(), indices3[i].item()]
                pred = set([mol_dict[x] for x in pred])
                if pred == set(r.reactants):
                    num_corrects += 1

    return num_corrects / len(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--datadir', type=str, default='data/uspto50k_coley')
    parser.add_argument('--dataset', type=str, default='uspto50k')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-iterations', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--eval-freq', type=int, default=1000)
    parser.add_argument('--save-freq', type=int, default=10000)
    parser.add_argument('--K', type=int, default=2)
    parser.add_argument('--clip', type=float, default=None)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--min', action='store_true')
    args = parser.parse_args()

    main(args)

