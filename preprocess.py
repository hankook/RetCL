import os, torch, argparse, shutil
from datasets import load_dataset
from utils import SMILESTokenizer, SMILESGraph

def preprocess_uspto50k(datadir):
    dataset = 'uspto50k'
    os.makedirs(os.path.join('.cache', datadir))

    sets = load_dataset(dataset, datadir)
    for key in list(sets.keys()):
        if key not in ['train', 'val', 'test']:
            del sets[key]

    molecules = []
    for data in sets.values():
        for attr in ['reactants', 'products']:
            molecules += getattr(data, attr)

    vocabfile = os.path.join('.cache', datadir, 'vocab.pth')
    if not os.path.isfile(vocabfile):
        vocab = SMILESTokenizer.build_vocab(molecules)
        torch.save(vocab, vocabfile)

    vocabfile = os.path.join('.cache', datadir, 'graph_vocab.pth')
    if not os.path.isfile(vocabfile):
        vocab = SMILESGraph.build_vocab(molecules)
        torch.save(vocab, vocabfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data/typed_schneider50k')
    parser.add_argument('--dataset', type=str, default='uspto50k')
    args = parser.parse_args()

    if args.dataset == 'uspto50k':
        preprocess_uspto50k(args.datadir)

