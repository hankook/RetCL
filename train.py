import os, argparse, logging, random
from collections import defaultdict

import torch, dgl, faiss
import torch.optim as optim

import utils
from datasets import load_reaction_dataset, load_molecule_dict, build_dataloader
from trainers.retrosynthesis import create_retrosynthesis_trainer, create_retrosynthesis_evaluator
from trainers.utils import collect_embeddings, knn_search
from models import load_module
from models.similarity import *
from models.loss import SimCLR
from options import add_model_arguments

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')
faiss_res = faiss.StandardGpuResources()

def main(args):
    if not args.resume:
        os.makedirs(args.logdir)
    utils.set_logging_options(args.logdir)

    logger = logging.getLogger('train')
    logger.info(' '.join(os.sys.argv))
    logger.info(args)

    ### DATASETS
    datasets = load_reaction_dataset(args.datadir)
    mol_dict = load_molecule_dict(args.mol_dict)
    for mol in datasets['mol_dict']:
        mol_dict.add(mol)

    ### DATALOADERS
    trainloader = build_dataloader(datasets['train'], batch_size=args.batch_size, num_iterations=args.num_iterations)

    ### MODELS
    module = load_module(args).to(device)
    if args.pretrain is not None:
        ckpt = torch.load(args.pretrain, map_location='cpu')
        module.load_state_dict(ckpt['module'], strict=False)

    ### LOSS
    if args.module in ['v1', 'v2', 'v3']:
        sim_fn = AttentionSimilarity()
    elif args.module == 'v4':
        sim_fn = MaxSimilarity()
    elif args.module == 'v5':
        sim_fn = MultiAttentionSimilarity()
    loss_fn = SimCLR(sim_fn, args.tau).to(device)

    ### OPTIMIZER
    if not args.freeze:
        params = list(module.parameters())
    else:
        params = []
        for name, param in module.named_parameters():
            if not name.startswith('encoder'):
                params.append(param)

    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

    if args.schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_iterations)

    nearest_neighbors = defaultdict(list)

    ### TRAINER
    train_step = create_retrosynthesis_trainer(module, loss_fn, forward=not args.backward_only, clip=args.clip,
                                               nearest_neighbors=nearest_neighbors,
                                               num_neighbors=args.num_neighbors,
                                               alpha=args.alpha,
                                               device=device)
    evaluate = create_retrosynthesis_evaluator(module, sim_fn, device=device)

    ### TRAINING
    if args.resume:
        ckpt = torch.load(os.path.join(args.logdir, 'best.pth'), map_location='cpu')
        module.load_state_dict(ckpt['module'])
        optimizer.load_state_dict(ckpt['optim'])
        iteration = ckpt['iteration']
        best_acc = ckpt['best_acc']
    else:
        iteration = 0
        best_acc = 0

    def save(name='last.pth'):
        torch.save({
            'module': module.state_dict(),
            'optim': optimizer.state_dict(),
            'iteration': iteration,
            'best_acc': best_acc,
            'args': vars(args),
        }, os.path.join(args.logdir, name))

    product2reactions = defaultdict(list)
    for i, rxn in enumerate(datasets['train']):
        product2reactions[rxn.product.smiles].append(i)

    for reactions in trainloader:
        iteration += 1
        if iteration > args.num_iterations:
            break

        if (iteration-1) % args.update_freq == 0:
            logger.info('Update nearest_neighbors ...')
            with torch.no_grad():
                embeddings = collect_embeddings(module, mol_dict, device=device)
                logger.info('Update nearest_neighbors ... collecting embeddings is done')
                keys = module.construct_keys(embeddings)
                keys = F.normalize(keys).to(device)
                for i in range(0, keys.shape[0], 512):
                    _, indices = torch.einsum('ik, jk -> ij', keys[i:i+512], keys).topk(65, dim=1)
                    for j, neighbors in enumerate(indices.tolist()):
                        nearest_neighbors[mol_dict[i+j].smiles] = [mol_dict[k] for k in neighbors[1:]]

                #_, indices = knn_search(faiss_res, keys, keys, args.num_neighbors+1)
                #for i, nns in enumerate(indices.tolist()):
                #    nearest_neighbors[mol_dict[i].smiles] = [mol_dict[j] for j in nns[1:]]
            logger.info('Update nearest_neighbors ... done')

        if args.hard_batch:
            hard_reactions = []
            for rxn in reactions:
                hard_reactions.append(rxn)
                for mol in nearest_neighbors[reactions[0].product.smiles]:
                    if len(product2reactions[mol.smiles]) > 0:
                        hard_reactions.append(datasets['train'][random.choice(product2reactions[mol.smiles])])
                if len(hard_reactions) > args.batch_size:
                    break
            reactions = hard_reactions

        # TRAINING
        optimizer.zero_grad()
        outputs = train_step(reactions)
        outputs['loss'].backward()
        if args.clip is not None:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        if args.schedule is not None:
            scheduler.step()

        # LOGGING
        logger.info('[Iter {}] [BatchSize {}] [Loss {:.4f}] [BatchAcc {:.4f}]'.format(
            iteration, len(reactions), outputs['loss'].item(), outputs['acc'].item()))

        if iteration % args.eval_freq == 0:
            acc, _ = evaluate(datasets['mol_dict'], datasets['val'])
            acc = acc[0]
            if best_acc < acc:
                logger.info(f'[Iter {iteration}] [BEST {acc:.4f}]')
                best_acc = acc
                save('best.pth')
            save()
            logger.info(f'[Iter {iteration}] [Val Acc {acc:.4f}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--backward-only', action='store_true')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--datadir', type=str, default='/data/uspto/uspto50k_coley')
    parser.add_argument('--mol-dict', type=str, default='/data/uspto/uspto50k_coley')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-iterations', type=int, default=200000)
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--schedule', type=str, default=None)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--eval-freq', type=int, default=1000)
    parser.add_argument('--num-neighbors', type=int, default=0)
    parser.add_argument('--update-freq', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--hard-batch', action='store_true')
    args = parser.parse_args()

    main(args)

