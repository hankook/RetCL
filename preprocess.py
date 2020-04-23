import os, csv, argparse
from datasets import Molecule, MoleculeDict
from datasets import Reaction, ReactionDataset


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)
    args = parser.parse_args()

    if args.dataset == 'uspto50k':
        uspto50k(args.datadir)
    else:
        raise Exception(f'Unknown dataset: {args.dataset}')

