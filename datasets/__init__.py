import os
import logging
from torch.utils.data import DataLoader, RandomSampler
from .molecule import Molecule, MoleculeDict
from .reaction import Reaction, ReactionDataset

logger = logging.getLogger('data')

def load_reaction_dataset(datadir, splits=['train', 'val', 'test']):
    datasets = {}
    logger.info('Loading ReactionDataset ...')
    for split in splits:
        datasets[split] = ReactionDataset.load(os.path.join(datadir, f'cache.{split}'))
        logger.info('- {} reactions in {} split'.format(len(datasets[split]), split))
    return datasets

def load_molecule_dict(datadir):
    logger = logging.getLogger('data')
    logger.info('Loading MoleculeDict ...')
    mol_dict = MoleculeDict.load(os.path.join(datadir, 'cache.molecule_dict'))
    logger.info('- {} molecules'.format(len(mol_dict)))
    return mol_dict

def check_molecule_dict(mol_dict, datasets):
    for split in ['train', 'val', 'test']:
        for rxn in datasets[split]:
            assert rxn.product in mol_dict
            for reactant in rxn.reactants:
                assert reactant in mol_dict

def collate(batch):
    return batch

def build_dataloader(dataset, batch_size=64, num_iterations=None):
    if num_iterations is None:
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=False)
    else:
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate,
                          sampler=RandomSampler(dataset, replacement=True, num_samples=num_iterations*batch_size))

