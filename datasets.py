import os, csv, random
import torch, numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from collections import defaultdict, OrderedDict
from torch.utils.data import DataLoader, RandomSampler


def _get_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    smiles = Chem.MolToSmiles(mol)
    return smiles

def _random_augment(smiles):
    """https://github.com/EBjerrum/SMILES-enumeration"""
    if random.random() < 0.5:
        m = Chem.MolFromSmiles(smiles)
        ans = list(reversed(range(m.GetNumAtoms())))
        nm = Chem.RenumberAtoms(m, ans)
        return Chem.MolToSmiles(nm, canonical=False)
    else:
        return smiles

def _random_strong_augment(smiles):
    """https://github.com/EBjerrum/SMILES-enumeration"""
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    return Chem.MolToSmiles(nm, canonical=False)


class MoleculeSet(Dataset):

    def __init__(self, molecules, augment=False):
        _set = set()
        for m in molecules:
            _set |= set(m.split('.'))
        self.molecules = sorted(list(_set))
        self.mol2idx = { mol: i for i, mol in enumerate(self.molecules) }
        self.augment = augment

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        if self.augment:
            return _random_augment(self.molecules[idx])
        else:
            return self.molecules[idx]

    def find(self, mol):
        return self.mol2idx[mol]

    def findall(self, mols):
        return [self.find(mol) for mol in mols]


class RetroSynthesisDataset(Dataset):

    def __init__(self, reactants, products, labels, augment=False):
        self.reactants = reactants
        self.products = products
        self.labels = labels
        self.augment = augment

        assert len(self.reactants) == len(self.products)
        if self.labels is not None:
            assert len(self.labels) == len(self.products)

    def __getitem__(self, idx):
        if self.labels is None:
            if self.augment:
                return _random_augment(self.reactants[idx]), _random_augment(self.products[idx])
            else:
                return self.reactants[idx], self.products[idx]
        else:
            if self.augment:
                return _random_augment(self.reactants[idx]), _random_augment(self.products[idx]), self.labels[idx]
            else:
                return self.reactants[idx], self.products[idx], self.labels[idx]

    def __len__(self):
        return len(self.products)


class AugmentedMoleculeSet(Dataset):

    def __init__(self, molset):
        self.molset = molset

    def __len__(self):
        return len(self.molset)

    def __getitem__(self, idx):
        m = self.molset[idx]
        m1 = _random_strong_augment(m)
        m2 = _random_strong_augment(m)
        return m1, m2


def load_dataset(dataset, datadir, augment=False, num_reactants=None):
    if dataset == 'uspto50k':

        def _load(split):
            filename = os.path.join(datadir, 'raw_{}.csv'.format(split))
            savefile = os.path.join('.cache', datadir, '{}.pth'.format(split))
            os.makedirs(os.path.dirname(savefile), exist_ok=True)
            if os.path.isfile(savefile):
                return torch.load(savefile)

            reactants = []
            products = []
            labels = []
            with open(filename) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r, _, p = row['reactants>reagents>production'].split('>')
                    reactants.append(_get_canonical_smiles(r))
                    products.append(_get_canonical_smiles(p))
                    labels.append(int(row['class']))

            torch.save((reactants, products, labels), savefile)
            return reactants, products, labels

        sets = {}
        for split in ['train', 'val', 'test']:
            sets[split] = RetroSynthesisDataset(*_load(split), augment=(augment and split == 'train'))

        known_molecules = set(sets['train'].products)
        train_molecules = set(sets['train'].products)
        for split, data in sets.items():
            for rs in data.reactants:
                known_molecules |= set(rs.split('.'))
                if split == 'train':
                    train_molecules |= set(rs.split('.'))
        sets['molset_train'] = MoleculeSet(train_molecules)
        sets['molset'] = MoleculeSet(known_molecules)
        sets['molset_valtest_r'] = MoleculeSet(sets['val'].reactants+sets['test'].reactants)

        if num_reactants is not None:
            for split in ['train', 'val', 'test']:
                d = sets[split]
                indices = [i for i in range(len(d)) if len(d[i][0].split('.')) == num_reactants]
                d.products = [d.products[i] for i in indices]
                d.reactants = [d.reactants[i] for i in indices]
                d.labels = [d.labels[i] for i in indices]

        return sets


class BatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, batch_size, num_iterations):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __len__(self):
        return self.num_iterations

    def __iter__(self):
        indices = []
        for _ in range(self.num_iterations):
            if len(indices) < self.batch_size:
                indices = torch.randperm(len(self.dataset)).tolist()

            yield indices[:self.batch_size]

            indices = indices[self.batch_size:]


def build_dataloader(dataset, batch_size, num_iterations=None, replacement=False):
    if num_iterations is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        if not replacement:
            return DataLoader(dataset, batch_size=batch_size,
                              sampler=RandomSampler(dataset, replacement=True, num_samples=num_iterations*batch_size))
        else:
            return DataLoader(dataset,
                              batch_sampler=BatchSampler(dataset, batch_size, num_iterations))

