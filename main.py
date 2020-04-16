import os, argparse, logging, math, random, time, copy
from collections import defaultdict

import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from datasets import load_dataset, build_dataloader
from models import Structure2Vec, MultiAttentionQuery, AttentionSimilarity, GraphModule, SimCLRv2
from models import masked_pooling, pad_sequence, graph_parallel
from models.score import PermutationScore

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def main(args):
    if not args.test_only:
        assert args.logdir is not None
        os.makedirs(args.logdir)
        utils.set_logging_options(args.logdir)
        summary_writer = SummaryWriter(args.logdir)

    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))
    logger.info(args)

    ### DATASETS
    datasets = load_dataset(args.dataset, args.datadir)

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

    ### LOSS
    sim_fn = AttentionSimilarity()
    loss_fn = SimCLRv2(sim_fn, args.tau).to(device)
    score_fn = PermutationScore(sim_fn, loss_fn)

    if args.test_only:
        print('TEST only')
        ckpt = torch.load(os.path.join(args.logdir, 'best.pth'), map_location='cpu')
        module.load_state_dict(ckpt['module'])
        print('Load checkpoint @ {}'.format(ckpt['iteration']))
        acc = evaluate(module, score_fn, datasets['test'], datasets['test_mol_dict'], batch_size=64, best=5, beam=5)
        print(acc)
        return

    ### DATALOADERS
    trainloader = build_dataloader(datasets['train'], batch_size=args.batch_size, num_iterations=args.num_iterations)

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
        batch = dgl.batch(graphs)
        if args.neg_aug > 0:
            neg_batch = copy.deepcopy(batch)
            masks = torch.zeros(neg_batch.number_of_nodes()).uniform_() < args.neg_aug
            n = masks.long().sum().item()
            random_atom_indices = [random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 22, 26, 28, 32]) for _ in range(n)]
            neg_batch.ndata['x'][masks, :43] = F.one_hot(torch.tensor(random_atom_indices), 43).float()
            batch = dgl.batch([batch, neg_batch])

        batch = batch.to(device)
        keys, p_queries, r_queries = module(batch)
        keys = torch.cat([keys, module.halt_keys], 0)

        losses, corrects = score_fn.compute_losses(
                (keys, p_queries, r_queries),
                (p_indices, r_indices))

        loss = losses.mean()
        acc = corrects.float().mean().item()

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
            acc = evaluate(module, score_fn, datasets['val'], datasets['known_mol_dict'], batch_size=64, best=1, beam=1)
            acc = acc[0].item()
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


def evaluate(module, score_fn, dataset, mol_dict, batch_size=64, best=5, beam=5):
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
        r_keys = torch.cat(r_keys, 0).detach()
        r_queries = torch.cat(r_queries, 0).detach()

        num_corrects = torch.zeros(best)
        for reactions in dataloader:
            N = len(reactions)
            batch = dgl.batch([r.product.graph for r in reactions]).to(device)
            p_keys, p_queries, _ = module(batch)

            predictions = score_fn.search(p_queries, (r_keys, r_queries), beam=beam)
            p_indices = []
            r_indices = []
            for pred in predictions:
                p_indices.append(pred[0])
                r_indices.append([x+p_keys.shape[0] for x in pred[1:]])
            scores = score_fn.compute_scores(
                    (torch.cat([p_keys, r_keys], 0), torch.cat([p_queries, r_queries], 0), torch.cat([p_queries, r_queries], 0)),
                    (p_indices, r_indices))
            pred_with_scores = defaultdict(list)
            for pred, score in zip(predictions, scores.tolist()):
                pred_with_scores[pred[0]].append((pred[1:], score))
            for i, r in enumerate(reactions):
                correct = False
                preds = sorted(pred_with_scores[i], key=lambda x: -x[1])
                for k, (pred, score) in enumerate(preds):
                    mols = set([mol_dict[idx] for idx in pred])
                    if mols == set(r.reactants):
                        correct = True
                    if correct and k < best:
                        num_corrects[k] += 1

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
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--neg-aug', type=float, default=0)
    args = parser.parse_args()

    main(args)

