##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Functions for Encoding Layer (Pure PyTorch Fallback)"""
import torch
import torch.nn.functional as F

__all__ = ['aggregate', 'scaled_l2', 'pairwise_cosine']

def aggregate(A, X, C):
    r""" Aggregate operation, aggregate the residuals of inputs (:math:`X`) with repect
    to the codewords (:math:`C`) with assignment weights (:math:`A`).

    .. math::

        e_{k} = \sum_{i=1}^{N} a_{ik} (x_i - d_k)

    Shape:
        - Input: :math:`A\in\mathcal{R}^{B\times N\times K}`
          :math:`X\in\mathcal{R}^{B\times N\times D}` :math:`C\in\mathcal{R}^{K\times D}`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`
    """
    # A: B x N x K
    # X: B x N x D
    # C: K x D
    # Result: B x K x D
    # e_k = sum_i a_ik (x_i - c_k) = (sum_i a_ik x_i) - (sum_i a_ik) c_k
    # sum_i a_ik x_i: B x K x D (via matmul)
    # sum_i a_ik: B x K (via sum)
    AX = torch.matmul(A.transpose(1, 2), X) # B x K x D
    sum_A = torch.sum(A, dim=1, keepdim=True).transpose(1, 2) # B x K x 1
    return AX - sum_A * C

def scaled_l2(X, C, S):
    r""" scaled_l2 distance

    .. math::
        sl_{ik} = s_k \|x_i-c_k\|^2

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}`
          :math:`C\in\mathcal{R}^{K\times D}` :math:`S\in \mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times N\times K}`
    """
    # X: B x N x D
    # C: K x D
    # S: K
    # Result: B x N x K
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x, c>
    X2 = torch.sum(X * X, dim=2, keepdim=True) # B x N x 1
    C2 = torch.sum(C * C, dim=1) # K
    XC = torch.matmul(X, C.t()) # B x N x K
    dist2 = X2 + C2 - 2 * XC # B x N x K
    return S * dist2

# Experimental
def pairwise_cosine(X, C, normalize=False):
    r"""Pairwise Cosine Similarity or Dot-product Similarity
    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}`
          :math:`C\in\mathcal{R}^{K\times D}` :math:`S\in \mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times N\times K}`
    """
    if normalize:
        X = F.normalize(X, dim=2, eps=1e-8)
        C = F.normalize(C, dim=1, eps=1e-8)
    return torch.matmul(X, C.t())
