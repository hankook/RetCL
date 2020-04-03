import torch, math
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter_logsumexp

class SimCLR(nn.Module):
    def __init__(self, score_fn, temperature):
        super(SimCLR, self).__init__()
        self.score_fn = score_fn
        self.temperature = temperature

    def forward(self, queries, keys, positive_indices, self_indices):
        N = queries.shape[0]
        scores = self.score_fn(queries, keys).div(self.temperature)
        self_masks = F.one_hot(self_indices, num_classes=keys.shape[0]).bool()
        scores = scores[~self_masks].view(N, -1)
        labels = positive_indices - (positive_indices > self_indices).long()
        loss = F.cross_entropy(scores, labels)
        corrects = scores.argmax(1) == labels
        return loss, corrects

    def update(self, keys):
        pass


class SimCLRv2(nn.Module):
    def __init__(self, score_fn, temperature):
        super(SimCLRv2, self).__init__()
        self.score_fn = score_fn
        self.temperature = temperature

    def forward(self, queries, keys, positive_indices, ignore_indices):
        N = queries.shape[0]
        scores = self.score_fn(queries, keys).div(self.temperature)
        indices = torch.zeros_like(scores, dtype=torch.long)
        positive_masks = torch.zeros_like(scores, dtype=torch.bool)
        ignore_masks = torch.zeros_like(scores, dtype=torch.bool)
        for i in range(N):
            indices[i, :] = i
            for j in positive_indices[i]:
                positive_masks[i, j] = True
            for j in ignore_indices[i]:
                ignore_masks[i, j] = True
        scores = scores.masked_fill(ignore_masks, float('-inf'))
        logp = scatter_logsumexp(scores[positive_masks], indices[positive_masks]) - scores.logsumexp(1)
        loss = logp.mul(-1).sum()

        preds = scores.argmax(1).tolist()
        corrects = [p in positive_indices[i] for i, p in enumerate(preds)]
        return loss, corrects


class MoCo(nn.Module):
    def __init__(self, score_fn, temperature, queue_size, num_hidden_features):
        super(MoCo, self).__init__()
        self.score_fn = score_fn
        self.temperature = temperature

        self.register_buffer("queue", torch.randn(queue_size, num_hidden_features))
        self.queue.requires_grad = False
        self.queue_head = 0
        self.queue_num_elements = 0

    def forward(self, queries, keys, positive_indices, self_indices):
        positive_keys = keys[positive_indices]
        if self.queue_num_elements == 0:
            return torch.tensor(0., requires_grad=True), torch.tensor([False])
        negative_keys = self.queue[:self.queue_num_elements]

        positive_scores = self.score_fn(queries, positive_keys, False)
        negative_scores = self.score_fn(queries, negative_keys)

        scores = torch.cat([positive_scores, negative_scores], 1).div(self.temperature)
        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        loss = F.cross_entropy(scores, labels)
        corrects = scores.argmax(1) == labels
        return loss, corrects

    def update(self, keys):
        n = keys.shape[0]
        N = self.queue.shape[0]
        offset = self.queue_head
        if offset + n < N:
            self.queue.data[offset:offset+n] = keys.data
        else:
            self.queue.data[offset:] = keys.data[:N-offset]
            self.queue.data[:n-(N-offset)] = keys.data[N-offset:]
        self.queue_head = (self.queue_head + n) % N
        self.queue_num_elements += n

