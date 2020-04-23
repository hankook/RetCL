import copy, random
import torch.nn.functional as F
from .utils import prepare_molecules
from datasets import Molecule

def create_pretrainer(
        encoder,
        classifier,
        optimizer,
        p=0.1,
        device=None):

    outputs = Molecule.atom_feat_size

    def step(batch):
        encoder.train()
        graphs = prepare_molecules(batch, device=device)
        graphs = copy.deepcopy(graphs)

        indices = []
        for i in range(graphs.number_of_nodes()):
            if random.random() < p:
                indices.append(i)
        targets = copy.deepcopy(graphs.ndata['x'][indices])[:, :43]
        graphs.ndata['x'][indices] = 0.

        node_embeddings = encoder(graphs)
        node_embeddings = node_embeddings[indices]

        preds = classifier(node_embeddings)[:, :43]

        loss = F.kl_div(F.log_softmax(preds, 1), targets, reduction='batchmean')
        acc = (preds.argmax(1) == targets.argmax(1)).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return { 'loss': loss.item(), 'acc': acc.item() }

    return step

