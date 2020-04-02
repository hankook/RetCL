import os, argparse, logging, math, random, copy, time

import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from datasets import load_dataset, build_dataloader
from models import Structure2Vec, MultiAttentionQuery, AttentionSimilarity, GraphModule, SimCLR, MoCo
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
    score_fn = AttentionSimilarity()

    ### LOSS
    if args.contrast == 'simclr':
        loss_fn = SimCLR(score_fn, args.tau).to(device)
    else:
        loss_fn = MoCo(score_fn, args.tau, args.moco_queue, 256).to(device)
        momentum_module = copy.deepcopy(module)

    logger.info('- # of parameters: {}'.format(sum(p.numel() for p in module.parameters())))
    if args.optimizer == 'adam':
        optimizer = optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(module.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.SGD(module.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    iteration = 0
    for reactions in trainloader:
        iteration += 1
        module.train()

        # preprocessing
        products = [r.product.graph for r in reactions]
        reactants = [random.choice(r.reactants).graph for r in reactions]

        N = len(products)
        graphs = products+reactants
        batch = dgl.batch(graphs).to(device)

        # compute features
        keys, p_queries, r_queries = graph_parallel(module, batch)
        if args.contrast == 'moco':
            with torch.no_grad():
                for momentum_param, param in zip(momentum_module.parameters(), module.parameters()):
                    momentum_param.data.mul_(args.moco_momentum).add_(1-args.moco_momentum, param.data)
                keys, _, _ = graph_parallel(momentum_module, batch)

        loss = 0.

        # maximize P(r|p)
        loss1, corrects1 = loss_fn(p_queries[:N], keys,
                                   torch.arange(N, 2*N).to(device),
                                   torch.arange(N).to(device))

        # maximize P(p|r)
        loss2, corrects2 = loss_fn(r_queries[N:], keys,
                                   torch.arange(N).to(device),
                                   torch.arange(N, 2*N).to(device))

        loss = (loss1 + loss2) / 2.
        acc = (corrects1 & corrects2).float().mean().item()

        loss_fn.update(keys)


        optimizer.zero_grad()
        loss.backward()
        if args.clip is not None:
            nn.utils.clip_grad_norm_(models.parameters(), args.clip)
        optimizer.step()


        logger.info('[Iter {}] [Loss {:.4f}] [BatchAcc {:.4f}]'.format(iteration, loss, acc))
        summary_writer.add_scalar('train/loss', loss.item(), iteration)
        summary_writer.add_scalar('train/batch_acc', acc, iteration)


        if iteration % args.eval_freq == 0:
            num_corrects = evaluate(module, score_fn,
                                    datasets['test'], datasets['known_mol_dict'],
                                    beam=args.beam, topk=5, batch_size=64)
            logger.info('[Iter {}] [Top Acc {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}]'.format(
                iteration, *num_corrects))
            summary_writer.add_scalar('eval/acc', num_corrects[0], iteration)


        if iteration % args.save_freq == 0:
            torch.save({
                'module': module.state_dict(),
                'optim': optimizer.state_dict(),
                'iteration': iteration,
                'args': vars(args),
            }, os.path.join(args.logdir, 'ckpt-{}.pth'.format(iteration)))


def evaluate(module, score_fn, dataset, mol_dict, beam=5, topk=5, batch_size=64):
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

        r_keys = torch.cat(r_keys, 0).detach()
        r_queries = torch.cat(r_queries, 0).detach()

        num_corrects = [0] * topk
        for reactions in dataloader:
            N = len(reactions)
            batch = dgl.batch([r.product.graph for r in reactions]).to(device)
            p_keys, p_queries, _ = graph_parallel(module, batch)

            scores = []
            b = 512 // (batch_size // 64)
            for i in range(0, r_keys.shape[0], b):
                scores.append(score_fn(p_queries, r_keys[i:i+b]))
            scores = torch.cat(scores, 1)
            scores, indices = scores.topk(args.beam, dim=1)

            reverse_scores = score_fn(r_queries[indices.view(-1)],
                                      p_keys.repeat_interleave(args.beam, dim=0),
                                      False).view(-1, args.beam)

            total_scores = scores + reverse_scores
            _, indices2 = total_scores.topk(topk, dim=1)

            for i, r in enumerate(reactions):
                smiles = [m.smiles for m in r.reactants]
                correct = 0
                for j in range(topk):
                    k = indices2[i, j].item()
                    k = indices[i, k].item()
                    if mol_dict[k].smiles in smiles:
                        correct += 1

                    if correct > 0:
                        num_corrects[j] += 1

    num_corrects = [c / len(dataset) for c in num_corrects]
    return num_corrects


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--datadir', type=str, default='data/uspto50k_coley')
    parser.add_argument('--dataset', type=str, default='uspto50k')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-iterations', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--eval-freq', type=int, default=1000)
    parser.add_argument('--save-freq', type=int, default=10000)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--clip', type=float, default=None)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--contrast', type=str, default='simclr', choices=['simclr', 'moco'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--attn-mode', type=str, default='basic', choices=['basic', 'sqrt'])
    parser.add_argument('--moco-queue', type=int, default=1024)
    parser.add_argument('--moco-momentum', type=float, default=0.999)
    args = parser.parse_args()

    main(args)


