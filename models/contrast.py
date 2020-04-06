import torch, math
import torch.nn as nn
import torch.nn.functional as F


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

    def generate_masks(self, inputs, indices, value):
        masks = torch.full_like(inputs, not value, dtype=torch.bool)
        for i, row in enumerate(indices):
            row = [row] if type(row) is int else row
            for j in row:
                masks[i, j] = value
        return masks

    def forward(self, queries, keys, positive_indices, ignore_indices):
        """
        queries: N x ...
        keys:    M x ...
        positive_indices: a list (length N) of (integers or lists of integers)
        ignore_indices:   a list (length N) of (integers or lists of integers)
        """
        N = queries.shape[0]
        M = keys.shape[0]
        scores = self.score_fn(queries, keys).div(self.temperature)
        positive_masks = self.generate_masks(scores, positive_indices, True)
        negative_masks = self.generate_masks(scores, ignore_indices, False)
        positive_scores = scores.masked_fill(~positive_masks, float('-inf'))
        negative_scores = scores.masked_fill(~negative_masks, float('-inf'))
        loss = negative_scores.logsumexp(1) - positive_scores.logsumexp(1)
        with torch.no_grad():
            preds = negative_scores.argmax(1)
            corrects = (F.one_hot(preds, num_classes=M).bool() & positive_masks).any(1)
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

