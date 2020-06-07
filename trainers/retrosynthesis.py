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
        reduction=None,
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
        labels = []
        if forward:
            for r in reactions:
                products.append(None)
                reactants.append(r.reactants)
                positive_indices.append(r.product)
                ignore_indices.append(r.reactants)
                labels.append(r.label)

        for r in permuted_reactions:
            for i in range(len(r.reactants)+1):
                products.append(r.product)
                reactants.append(r.reactants[:i])
                if i < len(r.reactants):
                    positive_indices.append(r.reactants[i])
                else:
                    positive_indices.append(-1)
                ignore_indices.append(r.product)
                labels.append(r.label+10)

        labels = torch.tensor(labels).to(device)

        embeddings = module(graphs)
        queries = module.construct_queries(products, reactants, embeddings, labels=labels)
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
        if reduction == 'sum':
            b_losses   = torch.stack([x.view(-1, l+1).sum(1).mul(-1).logsumexp(0).mul(-1) for x, l in zip(b_losses,   lengths)], 0)
        else:
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
        score_fn=None,
        remove_duplicate=False,
        max_idx=-1):

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
                    labels = torch.tensor([reaction.label+10]*len(predictions)).to(device)
                    queries = module.construct_queries([p_idx]*len(predictions), predictions, embeddings, labels=labels)
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
                    if remove_duplicate:
                        check = set()
                        _predictions = []
                        _scores = []
                        for i in range(len(predictions)):
                            k = '.'.join([str(x) for x in sorted(predictions[i])])
                            if k not in check:
                                check.add(k)
                                _predictions.append(predictions[i])
                                _scores.append(scores[i])
                        predictions, scores = _predictions, _scores
                    predictions, scores = predictions[:beam], scores[:beam]

                if score_fn is not None and len(final_predictions) > 0 and beam > 1:
                    final_reactions = [Reaction(
                        product=reaction.product,
                        reactants=[mol_dict[i] for i in pred],
                        label=reaction.label) for pred, _ in final_predictions]
                    scores = score_fn(final_reactions)
                    final_predictions = [(pred, score) for (pred, _), score in zip(final_predictions, scores)]
                    final_predictions = sorted(final_predictions, key=lambda x: -x[1])
                else:
                    final_predictions = sorted(final_predictions,
                                               key=lambda x: -x[1]/(len(x[0])+1))
                correct = False
                all_predictions.append([])
                prev = None
                k2 = 0
                for k, (pred, score) in enumerate(final_predictions[:best]):
                    if remove_duplicate and sorted(pred) == prev:
                        continue

                    all_predictions[-1].append((pred, score))
                    if sorted(pred) == sorted(targets):
                        correct = True
                    if correct:
                        num_corrects[k2] += 1
                    k2 += 1

                    prev = sorted(pred)

                for k in range(k2, best):
                    if correct:
                        num_corrects[k] += 1

                if verbose:
                    logger.info('Evaluate reactions ... {} / {} (Top1: {:.3f})'.format(n, len(dataset), num_corrects[0] / (n+1)))

        return [c / len(dataset) for c in num_corrects], all_predictions

    return evaluate

def create_retrosynthesis_score_evaluator(
        module,
        sim_fn,
        forward=True,
        reduction=None,
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
            labels = []
            if forward:
                for r in reactions:
                    products.append(None)
                    reactants.append(r.reactants)
                    positive_indices.append(r.product)
                    ignore_indices.append(r.reactants)
                    labels.append(r.label)

            for r in permuted_reactions:
                for i in range(len(r.reactants)+1):
                    products.append(r.product)
                    reactants.append(r.reactants[:i])
                    if i < len(r.reactants):
                        positive_indices.append(r.reactants[i])
                    else:
                        positive_indices.append(-1)
                    ignore_indices.append(r.product)
                    labels.append(r.label+10)

            labels = torch.tensor(labels).to(device)

            embeddings = module(graphs)
            queries = module.construct_queries(products, reactants, embeddings, labels=labels)
            keys = module.construct_keys(embeddings)
            keys = torch.cat([keys, module.halt_keys], 0)

            scores = sim_fn(queries, keys[positive_indices], many_to_many=False)
            if forward:
                f_scores, b_scores = scores[:n].squeeze(1), scores[n:]
            else:
                f_scores, b_scores = 0, scores

            b_scores = torch.split(b_scores, [(x+1)*math.factorial(x) for x in lengths])
            if reduction == 'sum':
                b_scores = torch.stack([x.view(-1, l+1).sum(1).logsumexp(0) for x, l in zip(b_scores, lengths)], 0)
            else:
                b_scores = torch.stack([x.view(-1, l+1).sum(1).max(0)[0] for x, l in zip(b_scores, lengths)], 0)

            if forward:
                return [s / (l+2) for s, l in zip((f_scores+b_scores).tolist(), lengths)]
            else:
                return [s / (l+1) for s, l in zip(b_scores.tolist(), lengths)]

    return evaluate

