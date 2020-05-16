import copy, random
import dgl
import torch
import torch.nn.functional as F
from .utils import prepare_molecules
from datasets import Molecule, MoleculeDict

def fill_zero_randomly(tensor, p):
    tensor[torch.rand(tensor.shape[0]) < p] = 0

def create_pretrainer(
        module,
        loss_fn,
        optimizer,
        mol_dict=None,
        all_keys=None,
        num_neighbors=None,
        momentum=None,
        clip=None,
        p=0.1,
        device=None):

    def step(batch):
        batch_mols = MoleculeDict()
        for mol in batch:
            batch_mols.add(mol)

        if num_neighbors > 0:
            batch_mol_indices = [mol_dict.index(mol) for mol in batch_mols]
            with torch.no_grad():
                batch_keys = all_keys[batch_mol_indices]
                _, nearest_neighbors = torch.einsum('ik, jk -> ij', batch_keys, all_keys).topk(num_neighbors+1, dim=1)
                for nns in nearest_neighbors.tolist():
                    for idx in nns[1:]:
                        batch_mols.add(mol_dict[idx])

            batch_mol_indices = [mol_dict.index(mol) for mol in batch_mols]

            graphs1 = dgl.batch([mol.graph for mol in batch_mols]).to(device)
            with torch.no_grad():
                module.eval()
                embeddings = module(graphs1)
                keys = module.construct_keys(embeddings)
                all_keys[batch_mol_indices].mul_(momentum).add_(1-momentum, F.normalize(keys))
        else:
            graphs1 = dgl.batch([mol.graph for mol in batch_mols]).to(device)

        n = len(batch_mols)
        module.train()
        fill_zero_randomly(graphs1.ndata['x'], p)
        fill_zero_randomly(graphs1.edata['w'], p)
        graphs2 = dgl.batch([mol.graph for mol in batch_mols]).to(device)
        fill_zero_randomly(graphs2.ndata['x'], p)
        fill_zero_randomly(graphs2.edata['w'], p)

        graphs = dgl.batch([graphs1, graphs2])
        embeddings = module(graphs)
        keys = module.construct_keys(embeddings)
        positive_indices = list(range(n, 2*n)) + list(range(n))
        ignore_indices = list(range(2*n))
        losses, corrects = loss_fn(keys, keys, positive_indices, ignore_indices)

        loss = losses.mean()
        acc = corrects.float().mean()

        optimizer.zero_grad()
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(sum([list(pg['params']) for pg in optimizer.param_groups], []),
                                           clip)
        optimizer.step()

        return { 'loss': loss.item(), 'acc': acc.item() }

    return step

