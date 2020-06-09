import os, csv, argparse, time, torch
from datasets import Molecule, MoleculeDict
from datasets import Reaction, ReactionDataset
from datasets.molecule import canonicalize_smiles

def uspto(datadir, splits=['train', 'val', 'test']):
    for split in splits:
        reaction_dataset = ReactionDataset()
        with open(os.path.join(datadir, 'raw_{}.csv'.format(split))) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                r, _, p = row['reactants>reagents>production'].split('>')
                reactants = [Molecule.from_smiles(x) for x in r.split('.')]
                product = Molecule.from_smiles(p)
                try:
                    label = int(row['class'])
                except:
                    label = 0
                reaction_dataset.add(Reaction(reactants, product, label))

                print(split, i+1, 'reactions ...', end='\r')
            print()

        cachefile = os.path.join(datadir, f'cache.{split}')
        reaction_dataset.save(cachefile)

def uspto_50k(datadir):
    print('Preprocessing USPTO-50k ...')
    uspto(datadir)

def uspto_full(datadir):
    print('Preprocessing USPTO-full ...')
    uspto(datadir, splits='test')

def uspto_candidates(datadir):
    molecule_dict = MoleculeDict()
    print('Preprocessing USPTO-Candidates ...')
    with open(os.path.join(datadir, 'candidates.txt')) as f:
        t0 = time.time()
        for i, line in enumerate(f):
            mol = Molecule.from_smiles(line.strip(), canonicalize=False)
            molecule_dict.add(mol)
            print('{} molecules ({:.4f}s/mol)'.format(i+1, (time.time()-t0)/(i+1)), end='\r')
        print()

    molecule_dict.save(os.path.join(datadir, 'cache.molecule_dict'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)
    args = parser.parse_args()

    if args.dataset == 'uspto_50k':
        uspto_50k(args.datadir)
    elif args.dataset == 'uspto_candidates':
        uspto_candidates(args.datadir)
    else:
        raise Exception(f'Unknown dataset: {args.dataset}')

