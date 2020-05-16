import dgl
import math
import torch
import torch.nn.functional as F
import itertools
import logging
import random
from datasets import Molecule, MoleculeDict, Reaction, build_dataloader
from .utils import prepare_reactions, prepare_molecules, collect_embeddings, convert_tensor

def permute_reactions(reactions):
    permuted_reactions = []
    for r in reactions:
        for perm in itertools.permutations(r.reactants):
            permuted_reactions.append(Reaction(
                product=r.product,
                reactants=list(perm),
                label=r.label))
    return permuted_reactions


def create_retrosynthesis_trainer(
        module,
        loss_fn,
        forward=True,
        nearest_neighbors=None,
        num_neighbors=None,
        mol_dict=None,
        device=None):

    def step(batch):
        module.train()
        reactions, graphs = prepare_reactions(batch,
                                              nearest_neighbors=nearest_neighbors,
                                              num_neighbors=num_neighbors,
                                              device=device)
        n = len(reactions)
        lengths = [len(r.reactants) for r in reactions]
        permuted_reactions = permute_reactions(reactions)

        products, reactants = [], []
        positive_indices, ignore_indices = [], []
        if forward:
            for r in reactions:
                products.append(None)
                reactants.append(r.reactants)
                positive_indices.append(r.product)
                ignore_indices.append(r.reactants)

        for r in permuted_reactions:
            for i in range(len(r.reactants)+1):
                products.append(r.product)
                reactants.append(r.reactants[:i])
                if i < len(r.reactants):
                    positive_indices.append(r.reactants[i])
                else:
                    positive_indices.append(-1)
                ignore_indices.append(r.product)

        embeddings = module(graphs)
        queries = module.construct_queries(products, reactants, embeddings)
        keys = module.construct_keys(embeddings)
        keys = torch.cat([keys, module.halt_keys], 0)

        losses, corrects = loss_fn(queries, keys, positive_indices, ignore_indices)

        if forward:
            f_losses, f_corrects = losses[:n], corrects[:n]
            b_losses, b_corrects = losses[n:], corrects[n:]
        else:
            b_losses, b_corrects = losses, corrects
        b_losses   = torch.split(b_losses,   [(x+1)*math.factorial(x) for x in lengths])
        b_corrects = torch.split(b_corrects, [(x+1)*math.factorial(x) for x in lengths])
        b_losses   = torch.stack([x.view(-1, l+1).sum(1).min(0)[0] for x, l in zip(b_losses,   lengths)], 0)
        b_corrects = torch.stack([x.view(-1, l+1).all(1).any(0)    for x, l in zip(b_corrects, lengths)], 0)

        if forward:
            loss = (f_losses + b_losses).mean()
            acc = (f_corrects & b_corrects).float().mean()
        else:
            loss = b_losses.mean()
            acc = b_corrects.float().mean()

        return { 'loss': loss, 'acc': acc }

    return step


def create_retrosynthesis_evaluator(
        module,
        sim_fn,
        best=1,
        beam=1,
        device=None,
        verbose=False,
        chunk_size=200000,
        max_idx=-1,
        forward=True):

    if verbose:
        logger = logging.getLogger('eval')

    def evaluate(mol_dict, dataset, embeddings):
        module.eval()
        num_corrects = [0] * best
        all_predictions = []
        with torch.no_grad():
            embeddings = convert_tensor(embeddings, device=device)
            keys = module.construct_keys(embeddings)
            keys = torch.cat([keys, module.halt_keys], 0)
            halt_idx = keys.shape[0]-1

            for n, reaction in enumerate(dataset):
                targets = [mol_dict.index(r) for r in reaction.reactants]
                p_idx = mol_dict.index(reaction.product)

                final_predictions = []
                predictions = [[]]
                scores = [0.]
                for _ in range(4):
                    queries = module.construct_queries([p_idx]*len(predictions), predictions, embeddings)
                    similarities = []
                    for i in range(0, keys.shape[0], chunk_size):
                        similarities.append(sim_fn(queries, keys[i:i+chunk_size]).detach())
                    similarities = torch.cat(similarities, 1)
                    similarities[:, p_idx] = float('-inf')
                    similarities[:, max_idx:-1] = float('-inf')

                    topk_scores, topk_indices = similarities.topk(beam, dim=1)
                    topk_scores = topk_scores.tolist()
                    topk_indices = topk_indices.tolist()
                    new_predictions = []
                    for i, (pred, score) in enumerate(zip(predictions, scores)):
                        for j in range(beam):
                            if topk_indices[i][j] == halt_idx:
                                final_predictions.append((pred, score + topk_scores[i][j]))
                            else:
                                new_predictions.append((pred + [topk_indices[i][j]], score + topk_scores[i][j]))
                    if len(new_predictions) == 0:
                        break
                    predictions, scores = zip(*sorted(new_predictions, key=lambda x: -x[1]))
                    predictions, scores = predictions[:beam], scores[:beam]

                final_predictions = sorted(final_predictions, key=lambda x: -x[1]/(len(x[0])+1))
                correct = False
                all_predictions.append(final_predictions)
                for k, (pred, score) in enumerate(final_predictions[:best]):
                    if set(pred) == set(targets):
                        correct = True
                    if correct:
                        num_corrects[k] += 1

                if verbose:
                    logger.info('Evaluate reactions ... {} / {}'.format(n, len(dataset)))

        return [c / len(dataset) for c in num_corrects], all_predictions

    return evaluate

def create_retrosynthesis_score_evaluator(
        module,
        sim_fn,
        forward=True,
        device=None):

    def evaluate(reactions):
        module.eval()
        with torch.no_grad():
            reactions, graphs = prepare_reactions(reactions, device=device)
            n = len(reactions)
            lengths = [len(r.reactants) for r in reactions]
            permuted_reactions = permute_reactions(reactions)

            products, reactants = [], []
            positive_indices, ignore_indices = [], []
            if forward:
                for r in reactions:
                    products.append(None)
                    reactants.append(r.reactants)
                    positive_indices.append(r.product)
                    ignore_indices.append(r.reactants)

            for r in permuted_reactions:
                for i in range(len(r.reactants)+1):
                    products.append(r.product)
                    reactants.append(r.reactants[:i])
                    if i < len(r.reactants):
                        positive_indices.append(r.reactants[i])
                    else:
                        positive_indices.append(-1)
                    ignore_indices.append(r.product)

            embeddings = module(graphs)
            queries = module.construct_queries(products, reactants, embeddings)
            keys = module.construct_keys(embeddings)
            keys = torch.cat([keys, module.halt_keys], 0)

            scores = sim_fn(queries, keys[positive_indices], many_to_many=False)
            if forward:
                f_scores, b_scores = scores[:n].squeeze(1), scores[n:]
            else:
                f_scores, b_scores = 0, scores

            b_scores = torch.split(b_scores, [(x+1)*math.factorial(x) for x in lengths])
            b_scores = torch.stack([x.view(-1, l+1).sum(1).max(0)[0] for x, l in zip(b_scores, lengths)], 0)

            return f_scores + b_scores

    return evaluate

