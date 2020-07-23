#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement unmasked linear attention."""

import torch
import torch.nn as nn
from torch.nn import Module


class LinearSoftmaxAttention(Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we make use of a feature map function Φ(.) and perform the following
    computation

        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).

    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, alpha=None, eps=1e-6):
        super(LinearSoftmaxAttention, self).__init__()
        self.alpha = alpha or 3
        self.eps = eps
        self.norm_key = nn.LayerNorm(out_channels, elementwise_affine=False)
        self.norm_query = nn.LayerNorm(out_channels, elementwise_affine=False)


    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the normalization to the queries and keys
        Q = self.norm_key(queries)
        K = self.norm_query(keys)/(self.alpha*math.sqrt(keys.shape[-1]))

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        # The softmax Taylor approximation to
        # the second order makes the complexity l*d^3 instead of l*d^2

        # order0 = values.sum(2, keepdims=True).repeat(1, query.shape[2], 1, 1)
        order0 = 0 # Substract one trick to allow 0 correlation values
        KV = torch.einsum("nshd, nshm->nhmd", K, V)
        order1 = torch.einsum("nlhd, nhmd -> nlhm", Q, KV)
        QQ = torch.einsum("nlhd,nlhe->nlhde", Q, Q)
        KKV = torch.einsum("nshd,nshe,nshm->nshdem", K, K, values)
        order2 = 0.5 * torch.einsum("nlhde, nshdem -> nshm", QQ, KKV)  # N, T, L, h, d

        # norm = Q.shape[2]
        norm = 0 # Substract one trick to allow 0 correlation values
        norm = norm + torch.einsum("nlhd, nshd->nlh", Q, K.sum(2))
        KK = torch.einsum("nshd, nshe->nhde", K, K)
        norm = norm + 0.5 * torch.einsum("nlhde, nhde->nlh", QQ, KK)
        V = (order0 + order1 + order2) / norm.unsqueeze(-1)

        return V.contiguous()
