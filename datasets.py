import os, csv, dgl, logging, torch, numpy as np, re
from collections import OrderedDict, namedtuple
from torch.utils.data import Dataset, Subset, DataLoader, RandomSampler
from rdkit import Chem
from dgl.data.chem import smiles_to_bigraph, CanonicalBondFeaturizer, CanonicalAtomFeaturizer


def _canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    smiles = Chem.MolToSmiles(mol)
    return smiles

def _tokenize_smiles(smiles):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    return list(re.compile(pattern).findall(smiles))


Molecule = namedtuple('Molecule', 'smiles graph token_ids')
class MoleculeDictionary(Dataset):
    @staticmethod
    def save(cachedir, smiles, known_indices):
        atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='x')
        bond_featurizer = CanonicalBondFeaturizer(bond_data_field='w')
        graphs = []
        tokens = set()
        for i, s in enumerate(smiles):
            graphs.append(smiles_to_bigraph(s,
                                            node_featurizer=atom_featurizer,
                                            edge_featurizer=bond_featurizer))
            tokens |= set(_tokenize_smiles(s))
            print('{} / {}'.format(i+1, len(smiles)), end='\r', flush=True)

        tokens = tokens - set(['.'])
        tokens = [(tok, i+4) for i, tok in enumerate(sorted(list(tokens)))]
        vocab = OrderedDict([('<pad>', 0), ('<bos>', 1), ('<eos>', 2), ('<unk>', 3)] + tokens)

        torch.save(vocab, 'vocab.pth')
        dgl.data.utils.save_graphs(os.path.join(cachedir, 'mol_dict_graphs.bin'), graphs)
        torch.save((smiles, known_indices), os.path.join(cachedir, 'mol_dict.pth'))

    @staticmethod
    def load(cachedir):
        graphs, _ = dgl.data.utils.load_graphs(os.path.join(cachedir, 'mol_dict_graphs.bin'))
        vocab = torch.load(os.path.join(cachedir, 'vocab.pth'))
        smiles, known_indices = torch.load(os.path.join(cachedir, 'mol_dict.pth'))
        return MoleculeDictionary(smiles, graphs, vocab), known_indices

    def __init__(self, smiles, graphs, vocab):
        self._smiles = smiles
        self._graphs = graphs
        self._vocab = vocab
        self._smiles2idx = { s: i for i, s in enumerate(self._smiles) }

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def atom_feat_size(self):
        return self._graphs[0].ndata['x'].shape[1]

    @property
    def bond_feat_size(self):
        return self._graphs[0].edata['w'].shape[1]

    def __len__(self):
        return len(self._smiles)

    def __getitem__(self, idx):
        if type(idx) is str:
            idx = self._smiles2idx[idx]

        smiles = self._smiles[idx]
        graph  = self._graphs[idx]
        tokens = ['<bos>'] + _tokenize_smiles(smiles) + ['<eos>']
        token_ids = [self._vocab[t] for t in tokens]

        return Molecule(smiles=self._smiles[idx], graph=self._graphs[idx], token_ids=token_ids)


Reaction = namedtuple('Reaction', 'reactants product label')
class ReactionDataset(Dataset):
    def __init__(self, reactants, products, labels, mol_dict):
        self.reactants = [[mol_dict[r] for r in rs.split('.')] for rs in reactants]
        self.products = [mol_dict[p] for p in products]
        self.labels = labels

    def __getitem__(self, idx):
        return Reaction(product=self.products[idx],
                        reactants=self.reactants[idx],
                        label=self.labels[idx])

    def __len__(self):
        return len(self.products)


def load_dataset(dataset, datadir):
    logger = logging.getLogger('data')
    logger.info('Loading datasets ...')
    if dataset == 'uspto50k':

        cachedir = os.path.join('.cache', datadir)
        os.makedirs(cachedir, exist_ok=True)

        data = {}
        for split in ['train', 'val', 'test']:
            cachefile = os.path.join(cachedir, f'{split}.pth')
            if os.path.isfile(cachefile):
                data[split] = torch.load(cachefile)
                continue

            logger.info(f'- Canonicalize SMILES from raw data ({split} split) ...')
            reactants, products, labels = [], [], []
            with open(os.path.join(datadir, 'raw_{}.csv'.format(split))) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r, _, p = row['reactants>reagents>production'].split('>')
                    reactants.append(_canonicalize_smiles(r))
                    products.append(_canonicalize_smiles(p))
                    labels.append(int(row['class']))
            torch.save((reactants, products, labels), cachefile)
            data[split] = (reactants, products, labels)

        cachefile = os.path.join(cachedir, 'mol_dict.pth')
        if not os.path.isfile(cachefile):
            logger.info('- Construct DGL Graphs from SMILES ...')
            smiles = []
            for i in range(2):
                for split in ['train', 'val', 'test']:
                    new_smiles = set(sum([x.split('.') for x in data[split][i]], []))
                    new_smiles = new_smiles - set(smiles)
                    smiles += list(new_smiles)
                    if i == 1 and split == 'train':
                        known_indices = list(range(len(smiles)))
            MoleculeDictionary.save(cachedir, smiles, known_indices)
        mol_dict, known_indices = MoleculeDictionary.load(cachedir)

        datasets = { split: ReactionDataset(*data[split], mol_dict) for split in ['train', 'val', 'test'] }

        datasets['known_mol_dict'] = Subset(mol_dict, known_indices)
        datasets['mol_dict'] = mol_dict

    logger.info('- # of reactions in train/val/test splits: {} / {} / {}'.format(len(datasets['train']),
                                                                                 len(datasets['val']),
                                                                                 len(datasets['test'])))
    logger.info('- # of known/all molecules: {} / {}'.format(len(datasets['known_mol_dict']),
                                                             len(datasets['mol_dict'])))

    return datasets


def _collate(batch):
    return batch

def build_dataloader(dataset, batch_size, num_iterations=None):
    if num_iterations is None:
        return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate, shuffle=False)
    else:
        return DataLoader(dataset, batch_size=batch_size, collate_fn=_collate,
                          sampler=RandomSampler(dataset, replacement=True, num_samples=num_iterations*batch_size))

