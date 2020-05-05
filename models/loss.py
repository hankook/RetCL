import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def forward(self, queries, keys, positive_indices, ignore_indices):
        raise NotImplementedError


class SimCLR(ContrastiveLoss):

    def __init__(self, sim_fn, temperature):
        super(SimCLR, self).__init__()
        self.sim_fn = sim_fn
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
        scores = self.sim_fn(queries, keys).div(self.temperature)
        positive_masks = self.generate_masks(scores, positive_indices, True)
        negative_masks = self.generate_masks(scores, ignore_indices, False)
        positive_scores = scores.masked_fill(~positive_masks, float('-inf'))
        negative_scores = scores.masked_fill(~negative_masks, float('-inf'))
        loss = negative_scores.logsumexp(1) - positive_scores.logsumexp(1)
        with torch.no_grad():
            preds = negative_scores.argmax(1)
            corrects = (F.one_hot(preds, num_classes=M).bool() & positive_masks).any(1)
        return loss, corrects

