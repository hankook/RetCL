import os, argparse, logging

import torch, dgl
import torch.optim as optim

import utils
from datasets import load_reaction_dataset, load_molecule_dict, MoleculeDict
from trainers.retrosynthesis import create_retrosynthesis_trainer, create_retrosynthesis_evaluator
from models import load_module
from models.similarity import *
from models.loss import SimCLR
from options import add_model_arguments

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def main(args):
    utils.set_logging_options(None)
    logger = logging.getLogger('eval')

    ### DATASETS
    datasets = load_reaction_dataset(args.datadir)
    if args.mol_dict is not None:
        mol_dict = MoleculeDict.load(args.mol_dict)
        max_idx = len(mol_dict)
        for m in load_molecule_dict(args.datadir):
            mol_dict.add(m)
    else:
        mol_dict = load_molecule_dict(args.datadir)
        max_idx = len(mol_dict)

    ### MODELS
    module = load_module(args).to(device)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        module.load_state_dict(ckpt['module'])

    ### SIMILARITY FUNCTION
    if args.module == 'v5':
        sim_fn = MultiAttentionSimilarity()
    else:
        sim_fn = AttentionSimilarity()

    ### EVALUATOR
    evaluate = create_retrosynthesis_evaluator(module, sim_fn, device=device, verbose=True, best=args.best, beam=args.beam, cpu=True, chunk_size=5000, max_idx=max_idx, forward=not args.backward_only)
    topk_acc, _ = evaluate(mol_dict, datasets['test'])
    logger.info('  K    ACC')
    for k, acc in enumerate(topk_acc):
        logger.info('{:3d}  {:.4f}'.format(k+1, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--datadir', type=str, default='/data/uspto/uspto50k_coley')
    parser.add_argument('--mol-dict', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--backward-only', action='store_true')
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--best', type=int, default=5)
    args = parser.parse_args()

    main(args)

