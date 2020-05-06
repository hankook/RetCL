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


class AttentionSimilarity(Similarity):

    def __call__(self, queries, keys, many_to_many=True):
        if many_to_many:
            return self.compute_all(queries, keys)
        else:
            return self.compute(queries, keys)

    def compute(self, queries, keys):
        """
        queries: N x C x d
        keys:    N x d
        return:  N x 1
        """

        scores = torch.bmm(queries, keys.unsqueeze(-1))
        weights = scores.softmax(1)
        weighted_queries = queries.mul(weights).sum(1)

        weighted_queries = F.normalize(weighted_queries, dim=-1)
        keys = F.normalize(keys, dim=-1)
        return weighted_queries.mul(keys).sum(-1, keepdim=True)

    def compute_all(self, queries, keys):
        """
        queries: N x C x d
        keys:    M x d
        return:  N x M
        """

        N, C, d = queries.shape
        M, d = keys.shape

        scores = queries.view(N*C, d).matmul(keys.t()).view(N, C, M)
        weights = scores.softmax(1)
        weighted_queries = queries.unsqueeze(2).mul(weights.unsqueeze(3)).sum(1) # N x M x d

        weighted_queries = F.normalize(weighted_queries, dim=-1)
        keys = F.normalize(keys, dim=-1)
        return weighted_queries.mul(keys.unsqueeze(0)).sum(2) # N x M


class MaxSimilarity(Similarity):

    def __call__(self, queries, keys, many_to_many=True):
        if many_to_many:
            return self.compute_all(queries, keys)
        else:
            return self.compute(queries, keys)

    def compute(self, queries, keys):
        """
        queries: N x K x d
        keys:    N x d
        return:  N x 1
        """

        N, K, d = queries.shape
        N, d = keys.shape

        queries = F.normalize(queries, dim=-1)
        keys    = F.normalize(keys,    dim=-1)

        scores = queries.mul(keys.unsqueeze(1)).sum(2).max(1, keepdim=True)[0]
        return scores

    def compute_all(self, queries, keys):
        """
        queries: N x K x d
        keys:    M x d
        return:  N x M
        """

        N, K, d = queries.shape
        M, d = keys.shape
        
        queries = F.normalize(queries, dim=-1)
        keys    = F.normalize(keys,    dim=-1)

        scores = queries.view(N*K, d).matmul(keys.t()).view(N, K, M).max(1)[0]
        return scores

class MultiAttentionSimilarity(AttentionSimilarity):

    def __call__(self, queries, keys, many_to_many=True):
        if many_to_many:
            return self.compute_all(queries, keys)
        else:
            return self.compute(queries, keys)

    def compute(self, queries, keys):
        """
        queries: N x C x d
        keys:    N x K x d
        return:  N x 1
        """

        N, C, d = queries.shape
        N, K, d = keys.shape
        queries = queries.repeat_interleave(K, dim=0)
        keys = keys.view(N*K, d)
        scores = super(MultiAttentionSimilarity, self).compute(queries, keys)
        scores = scores.view(N, K).max(1, keepdim=True)[0]
        return scores

    def compute_all(self, queries, keys):
        """
        queries: N x C x d
        keys:    M x K x d
        return:  N x M
        """

        N, C, d = queries.shape
        M, K, d = keys.shape
        keys = keys.view(M*K, d)
        scores = super(MultiAttentionSimilarity, self).compute_all(queries, keys)
        scores = scores.view(N, M, K)
        scores = scores.max(2)[0]
        return scores


