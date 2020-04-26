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
    utils.set_logging_options(None)
    logger = logging.getLogger('eval')

    ### DATASETS
    datasets = load_reaction_dataset(args.datadir)
    mol_dict = load_molecule_dict(args.datadir)

    ### MODELS
    module = load_module(name=args.module,
                         num_layers=args.num_layers,
                         num_branches=args.num_branches,
                         K=args.K, num_halt_keys=1).to(device)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        module.load_state_dict(ckpt['module'])

    ### SIMILARITY FUNCTION
    sim_fn = AttentionSimilarity()

    ### EVALUATOR
    evaluate = create_retrosynthesis_evaluator(module, sim_fn, device=device, verbose=True, best=args.best, beam=args.beam)
    topk_acc = evaluate(mol_dict, datasets['test'])
    logger.info('  K    ACC')
    for k, acc in enumerate(topk_acc):
        logger.info('{:3d}  {:.4f}'.format(k+1, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-layers', type=int, default=5)
    parser.add_argument('--num-branches', type=int, default=2)
    parser.add_argument('--module', type=str, default='v1')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--datadir', type=str, default='/data/uspto50k_coley')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--best', type=int, default=5)
    parser.add_argument('--K', type=int, default=2)
    args = parser.parse_args()

    main(args)

