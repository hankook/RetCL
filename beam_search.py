import torch, os, argparse, logging, utils, math, random
from datasets import load_dataset, build_dataloader
from models import Model
from utils import SMILESTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')


def main(args):

    ### DATASETS
    print('Loading datasets ...')
    datasets = load_dataset(args.dataset, args.datadir)
    print('- # of reactions in train/val/test splits: {} / {} / {}'.format(len(datasets['train']),
                                                                           len(datasets['val']),
                                                                           len(datasets['test'])))
    print('- # of train molecules: {}'.format(len(datasets['molset_train'])))
    print('- # of known molecules: {}'.format(len(datasets['molset'])))

    testloaders = {
        'val':       build_dataloader(datasets['val'],    batch_size=args.batch_size),
        'test':      build_dataloader(datasets['test'],   batch_size=args.batch_size),
        'molecules': build_dataloader(datasets['molset'], batch_size=args.batch_size),
    }

    ### TOKENIZER
    tokenizer = SMILESTokenizer(os.path.join('.cache', 'data', args.dataset, 'vocab.pth'))

    ### MODELS
    print('Loading model ...')
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
    print('- # of parameters: {}'.format(sum(p.numel() for p in models.parameters())))

    ### CHECKPOINTS
    print('Loading checkpoint from {} ...'.format(args.ckpt))
    ckpt = torch.load(args.ckpt)
    models.load_state_dict(ckpt['models'])

    models.eval()
    candidate_features = []
    with torch.no_grad():
        for molecules in testloaders['molecules']:
            _, _, features = model(
                    *tokenizer.encode(molecules, device=device),
                    pooling=args.pooling)
            candidate_features.append(features.detach())
        candidate_features = torch.cat(candidate_features, 0)
        normalized_candidate_features = F.normalize(candidate_features).t()

        num_corrects = 0
        iteration = 0
        for reactants, products, labels in testloaders['test']:
            iteration += 1

            _, product_features, _ = model(
                    *tokenizer.encode(products, device=device),
                    pooling=args.pooling)
            product_features = product_features
            scores1 = torch.matmul(F.normalize(product_features),
                                   normalized_candidate_features)
            scores1, indices1 = scores1.topk(args.beam, dim=1)

            total_indices1 = []
            total_indices2 = []
            total_scores = []

            for k in range(args.beam):
                scores2 = torch.matmul(F.normalize(product_features-candidate_features[indices1[:, k]]),
                                       normalized_candidate_features)
                scores2, indices2 = scores2.topk(args.beam, dim=1)

                for l in range(args.beam):
                    scores3 = torch.mul(
                            F.normalize(product_features),
                            F.normalize(candidate_features[indices1[:, k]]+candidate_features[indices2[:, l]])).sum(1)
                    total_scores.append(scores1[:, k] + scores2[:, l] + scores3)
                    total_indices1.append(indices1[:, k])
                    total_indices2.append(indices2[:, l])
            total_scores = torch.stack(total_scores, 0)
            total_indices1 = torch.stack(total_indices1, 0)
            total_indices2 = torch.stack(total_indices2, 0)

            idx = total_scores.argmax(0)
            for i, r in enumerate(reactants):
                if len(r.split('.')) == 1:
                    m = datasets['molset'][indices1[i, 0].item()]
                    if m == r:
                        num_corrects += 1

                elif len(r.split('.')) == 2:
                    m1 = datasets['molset'][total_indices1[idx[i], i].item()]
                    m2 = datasets['molset'][total_indices2[idx[i], i].item()]
                    if r in [m1+'.'+m2, m2+'.'+m1]:
                        num_corrects += 1

            print('[Iter {}] [Acc {:.4f}]'.format(iteration, num_corrects / ((iteration-1)*args.batch_size+len(products))))

    acc = num_corrects / len(datasets['test'])
    print('[Top1 Acc {:.4f}]'.format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--datadir', type=str, default='data/typed_schneider50k')
    parser.add_argument('--dataset', type=str, default='uspto50k')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--pooling', type=str, default='mean')
    args = parser.parse_args()

    main(args)


