import os, argparse, logging, math, random, copy, time

import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from datasets import load_dataset, build_dataloader
from models import Structure2Vec, MultiAttentionQuery, AttentionSimilarity
from models import masked_pooling, pad_sequence

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
    datasets = load_dataset(args.dataset, args.datadir, data_type='graph')

    ### DATALOADERS
    trainloader = build_dataloader(datasets['train'],    batch_size=args.batch_size, num_iterations=args.num_iterations)
    valloader   = build_dataloader(datasets['val'],      batch_size=args.batch_size)
    testloader  = build_dataloader(datasets['test'],     batch_size=args.batch_size)
    molloader   = build_dataloader(datasets['mol_dict'], batch_size=args.batch_size)

    ### MODELS
    logger.info('Loading model ...')
    encoder = Structure2Vec(num_layers=5,
                            num_hidden_features=256,
                            num_atom_features=datasets['mol_dict'].atom_feat_size,
                            num_bond_features=datasets['mol_dict'].bond_feat_size).to(device)

    K = args.K
    query_func_o = MultiAttentionQuery(256, K).to(device)
    query_func_r = MultiAttentionQuery(256, K).to(device)
    score_fn = AttentionSimilarity()

    models = nn.ModuleDict({
        'encoder': encoder,
        'query_func_o': query_func_o,
        'query_func_r': query_func_r,
    })
    logger.info('- # of parameters: {}'.format(sum(p.numel() for p in models.parameters())))
    optimizer = optim.Adam(models.parameters(), lr=args.lr, weight_decay=args.wd)

    iteration = 0
    for reactions in trainloaders['train']:
        iteration += 1
        models.train()

        # preprocessing
        products = [r.product.graph for r in reactions]
        reactants = [random.choice(r.reactants).graph for r in reactions]

        N = len(products)
        graphs = products+reactants
        batch = dgl.batch(graphs).to(device)

        # compute features
        features = encoder(batch)
        features = torch.split(features, batch.batch_num_nodes)
        features, masks = pad_sequence(features)
        keys = masked_pooling(features, masks, mode='mean')

        loss = 0.

        # maximize p(r|o)
        queries = query_func_o(features[:N], masks[:N])
        scores = score_fn(queries, keys).div(args.tau)
        score_masks = torch.ones_like(scores, dtype=torch.bool)
        for i in range(N):
            score_masks[i, i] = 0
        scores = scores[score_masks].view(N, -1)
        labels = torch.arange(N-1, 2*N-1).to(device)
        loss += F.cross_entropy(scores, labels) / 2.
        with torch.no_grad():
            acc = (scores.argmax(1) == labels).float().mean().item()

        # maximize p(o|r)
        queries = query_func_r(features[N:], masks[N:])
        scores = score_fn(queries, keys).div(args.tau)
        score_masks = torch.ones_like(scores, dtype=torch.bool)
        for i in range(N):
            score_masks[i, N+i] = 0
        scores = scores[score_masks].view(N, -1)
        labels = torch.arange(N).to(device)
        loss += F.cross_entropy(scores, labels) / 2.


        optimizer.zero_grad()
        loss.backward()
        if args.clip is not None:
            nn.utils.clip_grad_norm_(models.parameters(), args.clip)
        optimizer.step()


        logger.info('[Iter {}] [Loss {:.4f}] [BatchAcc {:.4f}]'.format(iteration, loss, acc))
        summary_writer.add_scalar('train/loss', loss.item(), iteration)
        summary_writer.add_scalar('train/batch_acc', acc, iteration)


        if iteration % args.eval_freq == 0:
            models.eval()
            r_keys = []
            r_queries = []
            with torch.no_grad():
                for molecules in molloader:
                    batch = dgl.batch([m.graph for m in molecules]).to(device)
                    features = encoder(batch)
                    features = torch.split(features, batch.batch_num_nodes)
                    features, masks = pad_sequence(features)

                    r_keys.append(masked_pooling(features, masks, mode='mean').detach())
                    r_queries.append(query_func_r(features, masks).detach())

                r_keys = torch.cat(r_keys, 0)
                r_queries = torch.cat(r_queries, 0)

                num_corrects = [0] * 5
                for reactions in testloader:
                    N = len(reactions)
                    batch = dgl.batch([r.product.graph for r in reactions]).to(device)
                    features = encoder(batch)
                    features = torch.split(features, batch.batch_num_nodes)
                    features, masks = pad_sequence(features)

                    o_queries = query_func_o(features, masks).detach()
                    o_keys = masked_pooling(features, masks, mode='mean').detach()

                    scores = []
                    b = 512 // (args.batch_size // 64)
                    for i in range(0, r_keys.shape[0], b):
                        scores.append(score_fn(o_queries, r_keys[i:i+b]))
                    scores = torch.cat(scores, 1)
                    scores, indices = scores.topk(args.beam, dim=1)

                    reverse_scores = score_fn(r_queries[indices.view(-1)],
                                              o_keys.repeat_interleave(5, dim=0),
                                              False).view(-1, args.beam)

                    total_scores = scores + reverse_scores
                    _, indices2 = total_scores.topk(5, dim=1)

                    for i, rs in enumerate(reactants):
                        smiles = [r.smiles for r in rs]
                        correct = 0
                        for j in range(5):
                            k = indices2[i, j].item()
                            k = indices[i, k].item()
                            if datasets['molset'][k].smiles in smiles:
                                correct += 1

                            if correct > 0:
                                num_corrects[j] += 1

                num_corrects = [c / len(datasets['test']) for c in num_corrects]


            logger.info('[Iter {}] [Top Acc {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}]'.format(
                iteration, *num_corrects))
            summary_writer.add_scalar('eval/acc', num_corrects[0], iteration)


        if iteration % args.save_freq == 0:
            torch.save({
                'models': models.state_dict(),
                'optim': optimizer.state_dict(),
                'iteration': iteration,
                'args': vars(args),
            }, os.path.join(args.logdir, 'ckpt-{}.pth'.format(iteration)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--datadir', type=str, default='data/uspto50k_coley')
    parser.add_argument('--dataset', type=str, default='uspto50k')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-iterations', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--eval-freq', type=int, default=1000)
    parser.add_argument('--save-freq', type=int, default=10000)
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--clip', type=float, default=None)
    praser.add_argument('--beam', type=int, default=5)
    args = parser.parse_args()

    main(args)


