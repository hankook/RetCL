import dgl
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from .molecule import Molecule


Reaction = namedtuple('Reaction', ['reactants', 'product', 'label'])
class ReactionDataset(Dataset):

    def __init__(self, reactions=None):
        if reactions is None:
            self._reactions = []
        else:
            self._reactions = reactions

    def __getitem__(self, idx):
        return self._reactions[idx]

    def __len__(self):
        return len(self._reactions)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def add(self, reaction):
        assert isinstance(reaction, Reaction)
        self._reactions.append(reaction)

    def save(self, path):
        labels = [r.label for r in self]
        lengths = [len(r.reactants) for r in self]
        molecules = [r.product for r in self]
        molecules = sum([r.reactants for r in self], molecules)
        graphs = [m.graph for m in molecules]
        smiles = [m.smiles for m in molecules]

        dgl.data.utils.save_graphs(path+'.graphs.bin', graphs)
        torch.save((smiles, lengths, labels), path+'.smiles.pth')

    @staticmethod
    def load(path):
        graphs, _ = dgl.data.utils.load_graphs(path+'.graphs.bin')
        smiles, lengths, labels = torch.load(path+'.smiles.pth')
        reactions = []
        offset = len(labels)
        for i, (label, length) in enumerate(zip(labels, lengths)):
            reactions.append(Reaction(
                reactants=[Molecule(s, g) for s, g in zip(smiles[offset:offset+length], graphs[offset:offset+length])],
                product=Molecule(smiles[i], graphs[i]),
                label=label))
            offset += length
        return ReactionDataset(reactions)

    @classmethod
    def collate(cls, batch):
        return batch

