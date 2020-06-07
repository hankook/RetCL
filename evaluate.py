import os, argparse, logging

import torch, dgl
import torch.optim as optim

import utils
from datasets import load_reaction_dataset, load_molecule_dict, MoleculeDict, check_molecule_dict
from trainers.retrosynthesis import create_retrosynthesis_evaluator, create_retrosynthesis_score_evaluator
from trainers.utils import collect_embeddings
from models import load_module
from models.similarity import *
from models.loss import SimCLR
from options import add_model_arguments

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def main(args):
    utils.set_logging_options(None)
    logger = logging.getLogger('eval')
    logger.info(' '.join(os.sys.argv))
    logger.info(args)

    ### DATASETS
    datasets = load_reaction_dataset(args.datadir, ['test'])
    mol_dict = load_molecule_dict(args.mol_dict)
    try:
        check_molecule_dict(mol_dict, datasets)
    except:
        max_idx = len(mol_dict)
        for d in datasets.values():
            for rxn in d:
                mol_dict.add(rxn.product)
                for reactant in rxn.reactants:
                    mol_dict.add(reactant)

    ### MODELS
    module = load_module(args).to(device)
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        module.load_state_dict(ckpt['module'])
        logger.info('- load checkpoint @ {}'.format(ckpt['iteration']))

    ### SIMILARITY FUNCTION
    if args.module == 'v5':
        sim_fn = MultiAttentionSimilarity()
    elif args.module == 'v0':
        sim_fn = CosineSimilarity()
    else:
        sim_fn = AttentionSimilarity()

    ### EVALUATOR
    if args.use_score:
        score_fn = create_retrosynthesis_score_evaluator(module,
                                                         sim_fn,
                                                         device=device,
                                                         forward=not args.backward_only)
    else:
        score_fn = None
    evaluate = create_retrosynthesis_evaluator(module,
                                               sim_fn,
                                               device=device,
                                               verbose=True,
                                               best=args.best, beam=args.beam,
                                               score_fn=score_fn,
                                               remove_duplicate=args.remove_duplicate)
    embeddings = collect_embeddings(module, mol_dict, device=device)
    topk_acc, predictions = evaluate(mol_dict, datasets['test'], embeddings)
    logger.info('  K    ACC')
    for k, acc in enumerate(topk_acc):
        logger.info('{:3d}  {:.3f}'.format(k+1, acc))

    logger.info(' '.join(os.sys.argv))
    logger.info(args)

    if args.output_file is not None:
        if os.path.exists(args.output_file):
            logger.info(f'{args.output_file} exists ...')
        else:
            logger.info(f'writing predictions into {args.output_file} ...')
            with open(args.output_file, 'w') as f:
                for rxn, preds in zip(datasets['test'], predictions):
                    f.write('UNK {}>>{} {}\n'.format('.'.join([m.smiles for m in rxn.reactants]),
                                                     rxn.product.smiles,
                                                     len(preds)))
                    for p in preds:
                        f.write('UNK {}\n'.format('.'.join([mol_dict[i].smiles for i in p[0]])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--datadir', type=str, default='/data/uspto/uspto50k_coley')
    parser.add_argument('--mol-dict', type=str, default='/data/uspto/uspto50k_coley')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--backward-only', action='store_true')
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument('--best', type=int, default=5)
    parser.add_argument('--use-score', action='store_true')
    parser.add_argument('--output-file', type=str, default=None)
    parser.add_argument('--remove-duplicate', action='store_true')
    args = parser.parse_args()

    main(args)

