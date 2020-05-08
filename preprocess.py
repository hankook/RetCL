import os, csv, argparse, time, torch
from datasets import Molecule, MoleculeDict
from datasets import Reaction, ReactionDataset
from datasets.molecule import canonicalize_smiles


def uspto50k(datadir):
    molecule_dict = MoleculeDict()
    print('preprocessing USPTO-50k ...')
    for split in ['train', 'val', 'test']:
        reaction_dataset = ReactionDataset()
        with open(os.path.join(datadir, 'raw_{}.csv'.format(split))) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                r, _, p = row['reactants>reagents>production'].split('>')
                reactants = [molecule_dict.add(x) for x in r.split('.')]
                product = molecule_dict.add(p)
                label = int(row['class'])
                reaction_dataset.add(Reaction(reactants, product, label))

                print(split, i+1, 'reactions ...', end='\r')
            print()

        cachefile = os.path.join(datadir, f'cache.{split}')
        reaction_dataset.save(cachefile)

    print(len(molecule_dict), 'unique molecules ...')
    molecule_dict.save(os.path.join(datadir, 'cache.molecule_dict'))

def uspto_large(datadir):
    molecule_dict = MoleculeDict()
    print('preprocessing USPTO-Large ...')
    with open(os.path.join(datadir, '1976_Sep2016_USPTOgrants_smiles.rsmi')) as f:
        t0 = time.time()
        for i, line in enumerate(f.readlines()):
            if i == 0: continue
            reaction = line.split()[0]
            reactants, _, product = reaction.split('>')
            for mol in [product] + reactants.split('.'):
                try:
                    molecule_dict.add(mol)
                except:
                    pass
            print('{} ({:.4f}s/mol)'.format(i+1, (time.time()-t0)/(i+1)), end='\r')
        print('\n- # of unique molecules:', len(molecule_dict))

    molecule_dict.save(os.path.join(datadir, 'cache.molecule_dict'))

def uspto_large2(datadir):
    molecule_dict = MoleculeDict()
    print('preprocessing USPTO-Large2 ...')
    with open(os.path.join(datadir, 'candidates_single.txt')) as f:
        t0 = time.time()
        for i, line in enumerate(f.readlines()):
            mol = Molecule.from_smiles(line.strip(), canonicalize=True)
            molecule_dict.add(mol)
            print('{} ({:.4f}s/mol)'.format(i+1, (time.time()-t0)/(i+1)), end='\r')
        print('\n- # of unique molecules:', len(molecule_dict))

    molecule_dict.save(os.path.join(datadir, 'cache.molecule_dict'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)
    args = parser.parse_args()

    if args.dataset == 'uspto50k':
        uspto50k(args.datadir)
    elif args.dataset == 'uspto-large':
        uspto_large(args.datadir)
    elif args.dataset == 'uspto-large2':
        uspto_large2(args.datadir)
    else:
        raise Exception(f'Unknown dataset: {args.dataset}')

