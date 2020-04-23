import dgl
import math
import torch
import itertools
from datasets import MoleculeDict, Reaction, build_dataloader
from .utils import prepare_reactions, prepare_molecules

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
        optimizer,
        device=None):

    def step(batch):
        module.train()
        reactions, graphs = prepare_reactions(batch, device=device)
        n = len(reactions)
        lengths = [len(r.reactants) for r in reactions]
        permuted_reactions = permute_reactions(reactions)

        products, reactants = [], []
        positive_indices, ignore_indices = [], []
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

        f_losses, f_corrects = losses[:n], corrects[:n]
        b_losses, b_corrects = losses[n:], corrects[n:]
        b_losses   = torch.split(b_losses,   [(x+1)*math.factorial(x) for x in lengths])
        b_corrects = torch.split(b_corrects, [(x+1)*math.factorial(x) for x in lengths])
        b_losses   = torch.stack([x.view(-1, l+1).sum(1).min(0)[0] for x, l in zip(b_losses,   lengths)], 0)
        b_corrects = torch.stack([x.view(-1, l+1).all(1).any(0)    for x, l in zip(b_corrects, lengths)], 0)

        loss = (f_losses + b_losses).mean()
        acc = (f_corrects & b_corrects).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return { 'loss': loss.item(), 'acc': acc.item() }

    return step


def create_retrosynthesis_evaluator(
        module,
        sim_fn,
        batch_size=128,
        best=1,
        device=None):

    def evaluate(mol_dict, dataset):
        module.eval()
        num_corrects = [0] * best
        with torch.no_grad():
            embeddings = []
            for molecules in build_dataloader(mol_dict, batch_size=batch_size):
                embeddings.append(module(prepare_molecules(molecules, device=device)))
            embeddings = [torch.cat(e, 0) for e in zip(*embeddings)]
            keys = module.construct_keys(embeddings)
            keys = torch.cat([keys, module.halt_keys], 0)
            halt_idx = keys.shape[0]-1

            for reaction in dataset:
                targets = [mol_dict.index(r) for r in reaction.reactants]
                p_idx = mol_dict.index(reaction.product)
                predictions = [[]]
                for _ in range(4):
                    queries = module.construct_queries([p_idx]*len(predictions), predictions, embeddings)
                    similarities = sim_fn(queries, keys)
                    similarities[:, p_idx] = float('-inf')
                    index = similarities.argmax(1).item()
                    if index == halt_idx:
                        break
                    predictions[0].append(index)

                correct = False
                for k, pred in enumerate(predictions):
                    if set(pred) == set(targets):
                        correct = True
                    if correct:
                        num_corrects[k] += 1
        return [c / len(dataset) for c in num_corrects]

    return evaluate

