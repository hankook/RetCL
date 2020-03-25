import torch, os, re, logging, random
import torch.nn.functional as F
from collections import OrderedDict
from rdkit import Chem
import dgl


def set_logging_options(logdir):
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s - %(filename)s] %(message)s")
    fh = logging.FileHandler(os.path.join(logdir, 'log.txt'))
    sh = logging.StreamHandler(os.sys.stdout)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False


def read_candidate_file(filename, n):
    all_reactants = []
    with open(filename) as f:
        for line in f.readlines():
            reactants = ''.join(line.split(',')[0].split()).split('.')
            all_reactants.append(reactants)
    assert len(all_reactants) % n == 0
    k = len(all_reactants) // n

    all_reactants = [all_reactants[i*k:(i+1)*k] for i in range(n)]
    return all_reactants


def _tokenize_smiles(text):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    return list(re.compile(pattern).findall(text))


def _dropout_token(seq, dropout=0):
    if dropout < 1e-4:
        return seq

    seq = [token if random.random() > dropout else '???' for token in seq]
    return seq


class SMILESTokenizer(object):
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    @classmethod
    def build_vocab(cls, molecules):
        vocab = OrderedDict()
        def _find_tokens(x):
            tokens = set()
            if type(x) is list:
                for y in x:
                    tokens |= _find_tokens(y)
            elif type(x) is str:
                tokens = set(_tokenize_smiles(x))
            return tokens
        tokens = [(tok, i+4) for i, tok in enumerate(sorted(list(_find_tokens(molecules))))]
        vocab = OrderedDict([(cls.PAD_TOKEN, 0), (cls.BOS_TOKEN, 1),
                             (cls.EOS_TOKEN, 2), (cls.UNK_TOKEN, 3)] + tokens)
        return vocab

    def __init__(self, vocab_file):
        self.vocab = torch.load(vocab_file)
        self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    def save_vocabulary(self, save_dir):
        torch.save(self.vocab, os.path.join(save_dir, 'vocab.pth'))

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_token_id(self):
        return self.vocab[self.PAD_TOKEN]

    @property
    def unk_token_id(self):
        return self.vocab[self.UNK_TOKEN]

    def get_token_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, seqs, token_dropout=0, device=None):
        seqs = [[self.BOS_TOKEN] + _dropout_token(_tokenize_smiles(seq), token_dropout) + [self.EOS_TOKEN] for seq in seqs]

        batch_size = len(seqs)
        max_length = max(len(seq) for seq in seqs)

        input_ids = torch.zeros(batch_size, max_length, dtype=torch.long).fill_(self.pad_token_id)
        masks = torch.zeros(batch_size, max_length, dtype=torch.bool)

        for i, seq in enumerate(seqs):
            masks[i, :len(seq)] = True
            for j, token in enumerate(seq):
                input_ids[i, j] = self.get_token_id(token)

        if device is not None:
            input_ids, masks = input_ids.to(device), masks.to(device)

        return input_ids, masks

def _convert_smiles_to_graph(smiles):
    m = Chem.MolFromSmiles(smiles)
    atom_features = []
    for i, atom in enumerate(m.GetAtoms()):
        atom_features.append((
            atom.GetAtomicNum(),
            atom.GetTotalNumHs()))

    bond_features = []
    for bond in mol.GetBonds():
        bond_features.append((
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond.GetBondType()))

    return atom_features, bond_features

class SMILESGraph(object):

    @classmethod
    def build_vocab(cls, molecules):
        atoms = set()
        max_num_Hs = 0
        for m in molecules:
            graph = _convert_smiles_to_graph(m)
            for atom in graph[0]:
                atoms.add(atom[0])
                max_num_Hs = max(max_num_Hs, atom[1])

        vocab = {
            'atoms': OrderedDict([(atomic_num, i) for i, atomic_num in sorted(list(atoms))]),
            'max_num_Hs': max_num_Hs,
            'bonds': {
                Chem.rdchem.BondType.SINGLE: 0,
                Chem.rdchem.BondType.DOUBLE: 1,
                Chem.rdchem.BondType.TRIPLE: 2,
                Chem.rdchem.BondType.AROMATIC: 3, },
        }
        return vocab

    def __init__(self, vocab_file):
        self.vocab = torch.load(vocab_file)

    def encode(self, molecules, device=None):
        graphs = []
        for m in molecules:
            atom_features, bond_features = _convert_smiles_to_graph(m)
            atom_features = [(self.vocab['atoms'][atom[0]], atom[1]) for atom in atom_features]
            atom_features = torch.tensor(atom_features, dtype=torch.long)
            bond_features = [(i, j, self.vocab['bonds'][bond[2]]) for bond in bond_features]
            bond_features = torch.tensor(bond_features, dtype=torch.long)

            g = dgl.DGLGraph()
            g.add_nodes(atom_features.shape[0])
            g.add_edges(bond_features[:, 0], bond_features[:, 1])
            g.add_edges(bond_features[:, 1], bond_features[:, 0])

            g.ndata['x'] = torch.cat([
                F.one_hot(atom_features[:, 0], len(self.vocab['atoms'])),
                F.one_hot(atom_features[:, 1], self.vocab['max_num_Hs']), 1])
            g.edata['w'] = F.one_hot(bond_features[:, 2], len(self.vocab['bonds']))

            graphs.append(g)

        return dgl.batch(graphs)

