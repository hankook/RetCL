import torch, os, argparse, logging, utils, math, random
from datasets import load_dataset, build_dataloader
from models import Model
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
        'val':       build_dataloader(datasets['val'],    batch_size=args.batch_size),
        'test':      build_dataloader(datasets['test'],   batch_size=args.batch_size),
        'molecules': build_dataloader(datasets['molset'], batch_size=args.batch_size),
    }

    ### TOKENIZER
    tokenizer = SMILESTokenizer(os.path.join('.cache', 'data', args.dataset, 'vocab.pth'))

    ### MODELS
    logger.info('Loading model ...')
    model = Model(vocab_size=tokenizer.vocab_size,
                  hidden_size=256,
                  num_attention_heads=8,
                  intermediate_size=2048,
                  num_base_layers=4,
                  num_layers_per_branch=1,
                  num_branches=2).to(device)

    model = nn.DataParallel(model)

    models = nn.ModuleDict({
        'model': model,
    })
    logger.info('- # of parameters: {}'.format(sum(p.numel() for p in models.parameters())))
    optimizer = optim.AdamW(models.parameters(), lr=args.lr)

    iteration = 0
    for reactants, products, labels in trainloaders['train']:
        iteration += 1
        models.train()

        # preprocessing
        products = list(products)
        reactants = [random.choice(r.split('.')) for r in reactants]

        B = len(products)
        # M = len(reactants)

        # compute features
        _, prod_features, reac_features = model(
                *tokenizer.encode(products+reactants, token_dropout=args.token_dropout, device=device),
                pooling='mean')
        prod_features = F.normalize(prod_features)
        reac_features = F.normalize(reac_features)

        scores = torch.matmul(prod_features[:B], reac_features.t()).div(args.tau)
        scores = torch.stack([torch.cat([row[:i], row[i+1:]], 0) for i, row in enumerate(scores)], 0)
        labels = torch.arange(B-1, 2*B-1).to(device)
        loss = F.cross_entropy(scores, labels).mul(0.5)

        with torch.no_grad():
            acc = (scores[:B].argmax(1) == labels).float().mean().item()

        scores = torch.matmul(reac_features[B:], prod_features.t()).div(args.tau)
        scores = torch.stack([torch.cat([row[:B+i], row[B+i+1:]], 0) for i, row in enumerate(scores)], 0)
        labels = torch.arange(B).to(device)
        loss += F.cross_entropy(scores, labels).mul(0.5)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        logger.info('[Iter {}] [Loss {:.4f}] [BatchAcc {:.4f}]'.format(iteration, loss, acc))
        summary_writer.add_scalar('train/loss', loss.item(), iteration)
        summary_writer.add_scalar('train/batch_acc', acc, iteration)


        if iteration % args.eval_freq == 0:
            models.eval()
            candidate_features = []
            with torch.no_grad():
                for molecules in testloaders['molecules']:
                    _, _, features = model(
                            *tokenizer.encode(molecules, device=device),
                            pooling='mean')
                    candidate_features.append(features.detach())
                candidate_features = torch.cat(candidate_features, 0)
                normalized_candidate_features = F.normalize(candidate_features)

                corrects = []
                for reactants, products, labels in testloaders['test']:
                    _, product_features, _ = model(
                            *tokenizer.encode(products, device=device),
                            pooling='mean')
                    product_features = F.normalize(product_features)
                    scores = product_features.matmul(normalized_candidate_features.t())
                    indices = scores.argmax(1)

                    for i, r in enumerate(reactants):
                        if datasets['molset'][indices[i].item()] in r.split('.'):
                            corrects.append(1)
                        else:
                            corrects.append(0)

            acc = sum(corrects) / len(corrects)
            logger.info('[Iter {}] [Top1 Acc {:.4f}]'.format(iteration, acc))
            summary_writer.add_scalar('eval/acc', acc, iteration)


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
    parser.add_argument('--datadir', type=str, default='data/typed_schneider50k')
    parser.add_argument('--dataset', type=str, default='uspto50k')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-iterations', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--eval-freq', type=int, default=1000)
    parser.add_argument('--save-freq', type=int, default=10000)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--token-dropout', type=float, default=0)
    args = parser.parse_args()

    main(args)

