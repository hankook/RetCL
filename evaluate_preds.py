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

def compute_accuracy(true_reaction, pred_reactions, max_len=None):
    correct = False
    corrects = []
    target = set([mol.smiles for mol in true_reaction.reactants])
    for rxn in pred_reactions:
        pred = set([mol.smiles for mol in rxn.reactants])
        if pred == target:
            correct = True
        corrects.append(correct)
    if max_len is not None and max_len > len(corrects):
        if len(corrects) > 0:
            corrects = corrects + [corrects[-1]] * (max_len-len(corrects))
        else:
            corrects = [False] * max_len

    if max_len is not None:
        return corrects[:max_len]
    else:
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
    elif args.module == 'v0':
        sim_fn = CosineSimilarity()
    else:
        sim_fn = AttentionSimilarity()

    ### EVALUATOR
    evaluate = create_retrosynthesis_score_evaluator(module, sim_fn, device=device)

    original_corrects = torch.zeros(args.max_preds)
    resorted_corrects = torch.zeros(args.max_preds)
    filtered_corrects = torch.zeros(args.max_preds)
    filtered_corrects2 = torch.zeros(args.max_preds)
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

            original_corrects += torch.tensor(compute_accuracy(true_reaction, reactions, max_len=args.max_preds)).float()

            if len(reactions) > 0:
                scores = evaluate(reactions)
                resorted_reactions = sorted(list(zip(reactions, scores)), key=lambda x: -x[1])
                resorted_reactions = list(zip(*resorted_reactions))[0]
                resorted_corrects += torch.tensor(compute_accuracy(true_reaction, resorted_reactions, max_len=args.max_preds)).float()
            else:
                scores = []

            filtered_reactions = [rxn for rxn, score in zip(reactions, scores) if score > 0.7]
            filtered_corrects += torch.tensor(compute_accuracy(true_reaction, filtered_reactions, max_len=args.max_preds)).float()

            filtered_reactions2 = [rxn for rxn, score in zip(reactions, scores) if score > 0.5]
            filtered_corrects2 += torch.tensor(compute_accuracy(true_reaction, filtered_reactions2, max_len=args.max_preds)).float()

            random_reactions = random.sample(reactions, len(reactions))
            random_corrects += torch.tensor(compute_accuracy(true_reaction, random_reactions, max_len=args.max_preds)).float()

            logger.info('[# of reactions: {}] [original {:.4f}] [resorted {:.4f}] [filtered>0.7 {:.4f}] [filtered>0.5 {:.4f}] [random {:.4f}]'.format(
                counter,
                original_corrects[0] / counter,
                resorted_corrects[0] / counter,
                filtered_corrects[0] / counter,
                filtered_corrects2[0] / counter,
                random_corrects[0] / counter))

    logger.info('# of reactions', counter)
    logger.info(' K   original   resorted    filtered (0.7, 0.5)')
    for k, (acc1, acc2, acc3, acc4) in enumerate(zip(original_corrects.div(counter).tolist(),
                                                     resorted_corrects.div(counter).tolist(),
                                                     filtered_corrects.div(counter).tolist(),
                                                     filtered_corrects2.div(counter).tolist(),
                                                     )):
        logger.info('{:2d}   {:8.3f}   {:8.3f}   {:8.3f}   {:8.3f}'.format(k, acc1, acc2, acc3, acc4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--max-preds', type=int, default=50)
    args = parser.parse_args()

    main(args)

