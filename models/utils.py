import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiAttentionQuery(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 num_candidates=2,
                 mode='basic'):
        super(MultiAttentionQuery, self).__init__()
        self.hidden_size = hidden_size
        self.num_candidates = num_candidates
        self.projection = nn.Linear(hidden_size, num_candidates)
        self.mode = mode

    def forward(self, inputs, masks):
        """
        inputs: N x T x d
        masks:  N x T
        return: N x C x d
        """
        scores = self.projection(inputs)
        if self.mode == 'sqrt':
            scores = scores.div(self.hidden_size ** 0.5)
        scores = scores.masked_fill(~masks.unsqueeze(2), float('-inf')).softmax(1)
        scores = scores.unsqueeze(3)
        inputs = inputs.unsqueeze(2)
        return inputs.mul(scores).sum(1)


def pad_sequence(inputs):
    if not isinstance(inputs[0], torch.Tensor):
        inputs = [torch.tenosr(x) for x in inputs]
    outputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    masks = torch.zeros(*outputs.shape[:2], dtype=torch.bool, device=outputs.device)
    for i, x in enumerate(inputs):
        masks[i, :x.shape[0]] = True
    return outputs, masks


def masked_pooling(inputs, masks, mode='sum'):
    """
    inputs: N x T x d
    masks:  N x T     (boolean)
    return: N x d
    """

    masks = masks.unsqueeze(-1).float()
    if mode == 'sum':
        return inputs.mul(masks).sum(1)
    elif mode == 'mean':
        return inputs.mul(masks).mean(1)
    else:
        raise Exception('Unknown mode: {}'.format(mode))

