import torch, os, re, logging, random
from collections import OrderedDict


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
        vocab = OrderedDict([(self.PAD_TOKEN, 0), (self.BOS_TOKEN, 1),
                             (self.EOS_TOKEN, 2), (self.UNK_TOKEN, 3)] + tokens)
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

