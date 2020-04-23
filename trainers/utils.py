import dgl
from datasets import MoleculeDict, Reaction

def prepare_reactions(reactions, device=None):
    molecule_dict = MoleculeDict()
    for r in reactions:
        for m in [r.product] + r.reactants:
            molecule_dict.add(m)

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

