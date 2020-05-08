import os, argparse, logging, random

import torch, dgl
import torch.optim as optim

import utils
from datasets import Reaction, Molecule
from trainers.retrosynthesis import create_retrosynthesis_score_evaluator
from models import load_module
from models.similarity import *
from models.loss import SimCLR
from options import add_model_arguments

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def compute_accuracy(true_reaction, pred_reactions):
    correct = False
    corrects = []
    target = set([mol.smiles for mol in true_reaction.reactants])
    for rxn in pred_reactions:
        pred = set([mol.smiles for mol in rxn.reactants])
        if pred == target:
            correct = True
        corrects.append(correct)
    return corrects

def main(args):
    utils.set_logging_options(None)
    logger = logging.getLogger('eval')

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
    evaluate = create_retrosynthesis_score_evaluator(module, sim_fn, device=device)

    original_corrects = torch.zeros(args.max_preds)
    resorted_corrects = torch.zeros(args.max_preds)
    random_corrects   = torch.zeros(args.max_preds)
    with open(args.pred) as f:
        reader = iter(f)
        counter = 0
        while True:
            try:
                _, rxn, n = next(reader).split()
            except StopIteration:
                break

            counter += 1

            product = Molecule.from_smiles(rxn.split('>')[-1])
            true_reactants = [Molecule.from_smiles(mol) for mol in rxn.split('>')[0].split('.')]
            true_reaction = Reaction(
                product=product,
                reactants=true_reactants,
                label=None)

            reactions = []
            for _ in range(int(n)):
                line = next(reader)
                if len(line.split()) == 1 or len(reactions) >= args.max_preds:
                    continue
                _, pred = line.split()

                try:
                    reactants = [Molecule.from_smiles(mol) for mol in pred.split('.')]
                except:
                    continue

                reactions.append(Reaction(
                    product=product,
                    reactants=reactants,
                    label=None))

            original_corrects[:len(reactions)] += torch.tensor(compute_accuracy(true_reaction, reactions)).float()

            scores = evaluate(reactions)
            resorted_reactions = sorted(list(zip(reactions, scores.tolist())), key=lambda x: -x[1][0])
            resorted_reactions = list(zip(*resorted_reactions))[0]
            resorted_corrects[:len(reactions)] += torch.tensor(compute_accuracy(true_reaction, resorted_reactions)).float()

            random_reactions = random.sample(reactions, len(reactions))
            random_corrects[:len(reactions)] += torch.tensor(compute_accuracy(true_reaction, random_reactions)).float()

            logger.info('[# of reactions: {}] [original {:.4f}] [resorted {:.4f}] [random {:.4f}]'.format(
                counter,
                original_corrects[0] / counter,
                resorted_corrects[0] / counter,
                random_corrects[0] / counter))

    logger.info('# of reactions', counter)
    logger.info(' K   original   resorted |')
    for k, (acc1, acc2) in enumerate(zip(original_corrects.div(counter).tolist(),
                                         resorted_corrects.div(counter).tolist())):
        logger.info('{:2d}   {:8.4f}   {:8.4f}'.format(k, acc1, acc2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--max-preds', type=int, default=50)
    args = parser.parse_args()

    main(args)

