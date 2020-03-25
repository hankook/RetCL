import torch, os, argparse, logging, utils, math, random, copy
from datasets import load_dataset, build_dataloader
from models import Model, Encoder, masked_pooling, ResidualFeedforwardLayer, MaskedAttentionPooling
from utils import SMILESTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
    logger.info('Loading datasets ...')
    datasets = load_dataset(args.dataset, args.datadir, augment=args.augment)
    logger.info('- # of reactions in train/val/test splits: {} / {} / {}'.format(len(datasets['train']),
                                                                                 len(datasets['val']),
                                                                                 len(datasets['test'])))
    logger.info('- # of train molecules: {}'.format(len(datasets['molset_train'])))
    logger.info('- # of known molecules: {}'.format(len(datasets['molset'])))

    trainloaders = {
        'train':     build_dataloader(datasets['train'], batch_size=args.batch_size,
                                      num_iterations=args.num_iterations),
        'molecules': build_dataloader(datasets['molset'], batch_size=args.batch_size,
                                      num_iterations=args.num_iterations),
    }
    testloaders = {
        'train':     build_dataloader(datasets['train'],  batch_size=args.batch_size),
        'val':       build_dataloader(datasets['val'],    batch_size=args.batch_size),
        'test':      build_dataloader(datasets['test'],   batch_size=args.batch_size),
        'molecules': build_dataloader(datasets['molset'], batch_size=args.batch_size),
    }

    ### TOKENIZER
    tokenizer = SMILESTokenizer(os.path.join('.cache', 'data', args.dataset, 'vocab.pth'))

    ### MODELS
    logger.info('Loading model ...')
    encoder = Encoder(vocab_size=tokenizer.vocab_size,
                      hidden_size=256,
                      num_attention_heads=8,
                      intermediate_size=2048,
                      num_layers=4).to(device)

    encoder = nn.DataParallel(encoder)

    K = args.K

    f_o = ResidualFeedforwardLayer(256, intermediate_channels=1024).to(device)
    f_r = ResidualFeedforwardLayer(256, intermediate_channels=1024).to(device)
    map_o = MaskedAttentionPooling(256, K).to(device)
    map_r = MaskedAttentionPooling(256, K).to(device)

    models = nn.ModuleDict({
        'encoder': encoder,
        'f_o': f_o,
        'f_r': f_r,
        'map_o': map_o,
        'map_r': map_r,
    })
    logger.info('- # of parameters: {}'.format(sum(p.numel() for p in models.parameters())))
    optimizer = optim.AdamW(models.parameters(), lr=args.lr)

    iteration = 0
    for reactants, products, labels in trainloaders['train']:
        iteration += 1
        models.train()

        # preprocessing
        products = list(products)
        reactants = [r.split('.') for r in reactants]
        num_reactants = [len(r) for r in reactants]
        offsets = torch.tensor([len(products)] + num_reactants).cumsum(0).to(device)
        num_reactants = torch.tensor(num_reactants).to(device)
        reactants = sum(reactants, [])

        N = len(products)
        M = len(reactants)
        molecules = products+reactants

        # compute features
        input_ids, masks = tokenizer.encode(molecules, device=device)
        features = encoder(input_ids, masks)
        o = F.normalize(map_o(f_o(features), masks).view(N+M, K, 256), dim=-1)
        r = F.normalize(map_r(f_r(features), masks).view(N+M, K, 256), dim=-1)

        loss = 0.
        acc = 0.
        corrects = []

        all_scores = torch.bmm(o.permute(1, 0, 2), r.permute(1, 2, 0)).div(args.tau)
        all_scores = all_scores.logsumexp(0)

        # maximize p(r|o)
        scores = all_scores[:N]
        score_masks = torch.ones_like(scores, dtype=torch.bool)
        for i in range(N):
            score_masks[i, i] = 0
        scores = scores[score_masks].view(N, -1)
        loss = (F.cross_entropy(scores, offsets[:N]-1) + F.cross_entropy(scores, offsets[1:]-2)) / 2.

        with torch.no_grad():
            preds = scores.argmax(1)
            corrects = (preds >= offsets[:N]-1) & (preds < offsets[1:]-1)
            acc = corrects.float().mean().item()

        # maximize log p(o|r)
        scores = all_scores.t()[N:]
        score_masks = torch.ones_like(scores, dtype=torch.bool)
        for i in range(M):
            score_masks[i, N+i] = 0

        labels = []
        for i in range(N):
            for j in range(num_reactants[i].item()):
                labels.append(i)
        labels = torch.tensor(labels).to(device)
        loss += F.cross_entropy(scores[score_masks].view(M, -1), labels)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        logger.info('[Iter {}] [Loss {:.4f}] [BatchAcc {:.4f}]'.format(iteration, loss, acc))
        summary_writer.add_scalar('train/loss', loss.item(), iteration)
        summary_writer.add_scalar('train/batch_acc', acc, iteration)


        if iteration % args.eval_freq == 0:
            models.eval()
            candidates = []
            with torch.no_grad():
                for molecules in testloaders['molecules']:
                    input_ids, masks = tokenizer.encode(molecules, device=device)
                    features = encoder(input_ids, masks)
                    r = F.normalize(map_r(f_r(features), masks).view(-1, K, 256), dim=-1)
                    candidates.append(r.detach())
                candidates = torch.cat(candidates, 0).permute(1, 2, 0)

                num_corrects = [0] * 5
                for reactants, products, labels in testloaders['test']:
                    N = len(products)
                    input_ids, masks = tokenizer.encode(products, device=device)
                    features = encoder(input_ids, masks)
                    o = F.normalize(map_o(f_o(features), masks).view(-1, K, 256), dim=-1)

                    scores = torch.bmm(o.permute(1, 0, 2), candidates).max(0)[0]
                    _, indices = scores.topk(5, dim=1)

                    for i, r in enumerate(reactants):
                        rs = r.split('.')
                        correct = 0
                        for j, k in enumerate(indices[i].tolist()):
                            if datasets['molset'][k] in rs:
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
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--token-dropout', type=float, default=0)
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--diverse', action='store_true')
    args = parser.parse_args()

    main(args)





