import dgl
import torch
import random
import logging
from datasets import MoleculeDict, Reaction, build_dataloader

logger = logging.getLogger('trainers.utils')

def convert_tensor(inputs, device=None, detach=True):
    if isinstance(inputs, list):
        return list(convert_tensor(x, device=device, detach=detach) for x in inputs)
    elif isinstance(inputs, tuple):
        return tuple(convert_tensor(x, device=device, detach=detach) for x in inputs)
    elif isinstance(inputs, torch.Tensor):
        if device is not None:
            inputs = inputs.to(device)
        if detach:
            inputs = inputs.detach()
        return inputs
    else:
        return inputs


def collect_embeddings(module, mol_dict, batch_size=512, cpu=True, device=None):
    logger.info('Collect embeddings ...')
    module.eval()
    with torch.no_grad():
        dataloader = build_dataloader(mol_dict, batch_size=batch_size)
        out_device = torch.device('cpu') if cpu else device
        with torch.no_grad():
            embeddings = []
            for i, molecules in enumerate(dataloader):
                es = module(prepare_molecules(molecules, device=device))
                es = convert_tensor(es, device=out_device)
                embeddings.append(es)

            if not isinstance(embeddings[0], torch.Tensor):
                embeddings = [torch.cat(e, 0) for e in zip(*embeddings)]
            else:
                embeddings = torch.cat(embeddings, 0)
    logger.info('Collect embeddings ... done')
    return embeddings


def prepare_reactions(reactions, nearest_neighbors=None, num_neighbors=None, device=None):
    molecule_dict = MoleculeDict()
    for r in reactions:
        for m in [r.product] + r.reactants:
            molecule_dict.add(m)

    if nearest_neighbors is not None:
        for i in range(len(molecule_dict)):
            for mol in nearest_neighbors[molecule_dict[i].smiles][:num_neighbors]:
                molecule_dict.add(mol)

    graphs = dgl.batch([m.graph for m in molecule_dict])
    if device is not None:
        graphs = graphs.to(device)
    converted_reactions = []
    for r in reactions:
        converted_reactions.append(Reaction(
            product=molecule_dict.index(r.product),
            reactants=[molecule_dict.index(x) for x in r.reactants],
            label=r.label))

    return converted_reactions, graphs


def prepare_molecules(molecules, device=None):
    graphs = dgl.batch([m.graph for m in molecules])
    if device is not None:
        graphs = graphs.to(device)
    return graphs

