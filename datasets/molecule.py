import dgl
import torch
from torch.utils.data import Dataset
from dgl.data.chem.utils import (
        CanonicalAtomFeaturizer,
        CanonicalBondFeaturizer,
        smiles_to_bigraph )
from rdkit import Chem

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    smiles = Chem.MolToSmiles(mol)
    return smiles


class Molecule(object):

    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='x')
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='w')

    atom_feat_size = atom_featurizer.feat_size('x')
    bond_feat_size = bond_featurizer.feat_size('w')
    
    def __init__(self, smiles, graph):
        self.smiles = smiles
        self.graph = graph

    def __hash__(self):
        return self.smiles.__hash__()

    @classmethod
    def from_smiles(cls, smiles, canonicalize=True):
        if canonicalize:
            smiles = canonicalize_smiles(smiles)
        graph = smiles_to_bigraph(smiles,
                                  node_featurizer=cls.atom_featurizer,
                                  edge_featurizer=cls.bond_featurizer)
        return Molecule(smiles, graph)


class MoleculeDict(Dataset):

    def __init__(self, molecules=None):
        if molecules is None:
            self._molecules = []
        else:
            self._molecules = molecules
        self._smiles2idx = { mol.smiles: i for i, mol in enumerate(self._molecules) }

    def __len__(self):
        return len(self._molecules)

    def __getitem__(self, idx):
        if type(idx) is str:
            idx = self._smiles2idx[idx]

        return self._molecules[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def index(self, mol):
        if isinstance(mol, str):
            return self._smiles2idx[mol]
        elif isinstance(mol, Molecule):
            return self._smiles2idx[mol.smiles]
        else:
            raise Exception

    def __contains__(self, mol):
        return mol.smiles in self._smiles2idx

    def add(self, mol):
        if isinstance(mol, str):
            return self.add(Molecule.from_smiles(mol))

        assert isinstance(mol, Molecule)
        if mol not in self:
            self._molecules.append(mol)
            self._smiles2idx[mol.smiles] = len(self)-1
        return mol

    def save(self, path):
        graphs = [mol.graph for mol in self]
        smiles = [mol.smiles for mol in self]
        dgl.data.utils.save_graphs(path+'.graphs.bin', graphs)
        torch.save(smiles, path+'.smiles.pth')

    @staticmethod
    def load(path):
        graphs, _ = dgl.data.utils.load_graphs(path+'.graphs.bin')
        smiles = torch.load(path+'.smiles.pth')
        molecules = [Molecule(s, g) for s, g in zip(smiles, graphs)]
        return MoleculeDict(molecules)

    @classmethod
    def collate(cls, batch):
        return batch

