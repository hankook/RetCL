import torch
import torch.nn.functional as F

class Similarity:

    def __call__(self, queries, keys, many_to_many=True):
        raise NotImplementedError


class CosineSimilarity(Similarity):

    def __call__(self, queries, keys, many_to_many=True):
        if many_to_many:
            return self.compute_all(queries, keys)
        else:
            return self.compute(queries, keys)

    def compute(self, queries, keys):
        """
        queries: N x d
        keys:    N x d
        return:  N x 1
        """

        queries = F.normalize(queries, dim=-1)
        keys    = F.normalize(keys,    dim=-1)
        return queries.mul(keys).sum(1, keepdim=True)

    def compute_all(self, queries, keys):
        """
        queries: N x d
        keys:    M x d
        return:  N x M
        """

        queries = F.normalize(queries, dim=-1)
        keys    = F.normalize(keys,    dim=-1)
        return queries.matmul(keys.t())

