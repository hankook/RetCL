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
    datasets = load_dataset(args.dataset, args.datadir, num_reactants=2)
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
        'val':       build_dataloader(datasets['val'],    batch_size=4),
        'test':      build_dataloader(datasets['test'],   batch_size=4),
        'molecules': build_dataloader(datasets['molset'], batch_size=args.batch_size),
        'molecules_small': build_dataloader(datasets['molset_valtest_r'], batch_size=args.batch_size),
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
        reactants = [r.split('.') for r in reactants]
        for r in reactants:
            random.shuffle(r)
        reactants = sum(reactants, [])

        B = len(products)
        # M = len(reactants)

        # compute features
        _, prod_features, reac_features = model(
                *tokenizer.encode(products+reactants, device=device),
                pooling='mean')
        prod_features = F.normalize(prod_features)
        reac_features = reac_features
        Is, Js, labels = [], [], []
        for i in range(B*3):
            for j in range(i+1, B*3):
                Is.append(i)
                Js.append(j)
                if i >= B and (i-B) % 2 == 0 and j == i+1:
                    labels.append(len(Is)-1)
        reac_features = F.normalize(reac_features[Is]+reac_features[Js])

        scores = torch.matmul(prod_features[:B], reac_features.t()).div(args.tau)
        labels = torch.tensor(labels).to(device)
        loss = F.cross_entropy(scores, labels).mul(0.5)
        with torch.no_grad():
            acc = (scores.argmax(1) == labels).float().mean().item()

        scores = torch.matmul(reac_features[labels], prod_features.t()).div(args.tau)
        loss += F.cross_entropy(scores, torch.arange(B).to(device)).mul(0.5)


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
                # candidate_features = F.normalize(candidate_features[Is] + candidate_features[Js])

                correct = 0
                for reactants, products, labels in testloaders['test']:
                    B = len(products)
                    _, product_features, _ = model(
                            *tokenizer.encode(products, device=device),
                            pooling='mean')
                    product_features = F.normalize(product_features)
                    product_features = product_features.view(B, 1, -1)

                    reactants = [r.split('.') for r in reactants]
                    reactants = sum(reactants, [])

                    indices = datasets['molset'].findall(reactants)
                    indices = torch.tensor(indices).view(-1, 2)


                    reactant_features = candidate_features[indices[:, 0]].view(B, 1, -1) + candidate_features.view(1, candidate_features.shape[0], -1)
                    reactant_features = F.normalize(reactant_features, dim=-1)
                    scores = torch.bmm(product_features, reactant_features.transpose(1, 2)).squeeze(1)
                    print(scores.shape)

                    correct1 = (scores.argmax(1).cpu() == indices[:, 1]).long()

                    reactant_features = candidate_features[indices[:, 1]].view(B, 1, -1) + candidate_features.view(1, candidate_features.shape[0], -1)
                    reactant_features = F.normalize(reactant_features, dim=-1)
                    scores = torch.bmm(product_features, reactant_features.transpose(1, 2)).squeeze(1)
                    print(scores.shape)

                    correct2 = (scores.argmax(1).cpu() == indices[:, 0]).long()

                    correct += ((correct1+correct2) >= 2).float().sum()


            acc = correct / len(datasets['test'])
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
    args = parser.parse_args()

    main(args)


