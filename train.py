import os, argparse, logging

import torch, dgl
import torch.optim as optim

import utils
from datasets import load_reaction_dataset, load_molecule_dict, build_dataloader
from trainers.retrosynthesis import create_retrosynthesis_trainer, create_retrosynthesis_evaluator
from models import load_module
from models.similarity import AttentionSimilarity
from models.loss import SimCLR

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def main(args):
    os.makedirs(args.logdir)
    utils.set_logging_options(args.logdir)

    logger = logging.getLogger('train')
    logger.info(' '.join(os.sys.argv))
    logger.info(args)

    ### DATASETS
    datasets = load_reaction_dataset(args.datadir)
    mol_dict = load_molecule_dict(args.datadir)

    ### DATALOADERS
    trainloader = build_dataloader(datasets['train'], batch_size=args.batch_size, num_iterations=args.num_iterations)

    ### MODELS
    module = load_module(name=args.module,
                         num_layers=args.num_layers,
                         num_branches=args.num_branches,
                         K=args.K, num_halt_keys=1).to(device)
    if args.pretrain is not None:
        ckpt = torch.load(args.pretrain, map_location='cpu')
        module.encoder.load_state_dict(ckpt['encoder'], strict=False)

    ### LOSS
    sim_fn = AttentionSimilarity()
    loss_fn = SimCLR(sim_fn, args.tau).to(device)

    ### OPTIMIZER
    if not args.freeze:
        optimizer = optim.SGD(module.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    else:
        params = []
        for name, param in module.named_parameters():
            if not name.startswith('encoder'):
                params.append(param)
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=0.9)

    ### TRAINER
    train_step = create_retrosynthesis_trainer(module, loss_fn, optimizer, device=device)
    evaluate = create_retrosynthesis_evaluator(module, sim_fn, device=device)

    ### TRAINING
    iteration = 0
    best_acc = 0
    for reactions in trainloader:
        iteration += 1

        # TRAINING
        outputs = train_step(reactions)

        # LOGGING
        logger.info('[Iter {}] [Loss {loss:.4f}] [BatchAcc {acc:.4f}]'.format(iteration, **outputs))

        if iteration % args.eval_freq == 0:
            acc = evaluate(mol_dict, datasets['val'])
            acc = acc[0]
            if best_acc < acc:
                logger.info(f'[Iter {iteration}] [BEST {acc:.4f}]')
                best_acc = acc
                torch.save({
                    'module': module.state_dict(),
                    'optim': optimizer.state_dict(),
                    'iteration': iteration,
                    'best_acc': best_acc,
                    'args': vars(args),
                }, os.path.join(args.logdir, 'best.pth'))

            logger.info(f'[Iter {iteration}] [Val Acc {acc:.4f}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--num-layers', type=int, default=5)
    parser.add_argument('--num-branches', type=int, default=2)
    parser.add_argument('--module', type=str, default='v1')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--datadir', type=str, default='/data/uspto50k_coley')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-iterations', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--eval-freq', type=int, default=1000)
    parser.add_argument('--K', type=int, default=2)
    args = parser.parse_args()

    main(args)

