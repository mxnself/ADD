# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import operator
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torch.nn import Sequential, ModuleList, Linear

from fairseq.modules.fairseq_dropout import FairseqDropout


# from fairseq.modules.fairseq_dropout import FairseqDropout
# from fairseq.modules.quant_noise import quant_noise

def log_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)


class AdaptiveSoftmax(nn.Module):
    """
    This is an implementation of the efficient softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309).
    """
    in_features: int
    n_classes: int
    cutoffs: List[int]
    div_value: float
    head_bias: bool
    head: Linear
    tail: ModuleList

    def __init__(
            self,
            vocab_size,
            input_dim,
            cutoffs,
            dropout,
            factor=4.0,
            adaptive_inputs=None,
            tie_proj=False,
            q_noise=0,
            qn_block_size=8,
            head_bias=False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.cutoffs = cutoffs
        self.onnx_trace = True

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) > (vocab_size - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):
            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.in_features = input_dim
        self.n_classes = vocab_size
        self.cutoffs = cutoffs + [vocab_size]
        self.div_value = factor
        self.head_bias = head_bias

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(self.in_features, self.head_size, bias=self.head_bias)
        self.tail = ModuleList()

        self.dropout_module = FairseqDropout(eval(dropout), module_name=self.__class__.__name__)

        for i in range(self.n_clusters):
            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = Sequential(
                Linear(self.in_features, hsz, bias=False),
                nn.Dropout(self.dropout_module.p),
                Linear(hsz, osz, bias=False)
            )

            self.tail.append(projection)

    def reset_parameters(self) -> None:
        self.head.reset_parameters()
        for i2h, h2o in self.tail:
            i2h.reset_parameters()
            h2o.reset_parameters()


    def predict(self, input: Tensor) -> Tensor:
        r""" This is similar to `self.get_log_prob(input)`,
        but is more efficient in some cases.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            output (Tensor): a class with the highest probability for each example

        Shape:
            - Input: [bsz,len,d]
            - Output: [bsz,len,vocab_size]
        """
        bsz,seqlen,dim=input.size()
        input=input.contiguous().view(-1,dim)

        # [bsz,len,d]=>[bsz*len,d]
        vocab_size=self.cutoffs[-1]



        head_output = self.head(input)
        head_max_prob_idx = torch.argmax(head_output, dim=1)
        not_in_shortlist = (head_max_prob_idx >= self.shortlist_size)
        all_in_shortlist = not (not_in_shortlist.any())


        vocab_log_prob = torch.full((bsz*seqlen,vocab_size), 0).to(input).float()
        

        if all_in_shortlist:
            vocab_log_prob[:,:self.cutoffs[0]] = log_softmax(head_output[:,:self.cutoffs[0]], -1)


        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(input, head_output)
            vocab_log_prob = log_softmax(log_prob, -1)

        else:
            log_prob = self._get_full_log_prob(input[not_in_shortlist],
                                               head_output[not_in_shortlist])


            vocab_log_prob[~not_in_shortlist,:self.cutoffs[0]] = log_softmax(head_output[~not_in_shortlist,:self.cutoffs[0]], -1)
            vocab_log_prob[not_in_shortlist,:] = log_softmax(log_prob, -1)

        # [bsz*len,d]=>[bsz,len,d]
        vocab_log_prob=vocab_log_prob.view(bsz,seqlen,vocab_size)
        return vocab_log_prob

    def _get_full_log_prob(self, input, head_output):
        """ Given input tensor, and output of `self.head`,
        compute the log of the full distribution """

        out = input.new_empty((head_output.size(0), self.n_classes))
        head_logprob = log_softmax(head_output, dim=-1, onnx_trace=self.onnx_trace)

        out[:, :self.shortlist_size] = head_logprob[:, :self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
            cluster_output = self.tail[i](input)
            cluster_logprob = log_softmax(cluster_output, dim=-1, onnx_trace=self.onnx_trace)
            output_logprob = cluster_logprob + head_logprob[:, self.shortlist_size + i].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_logprob

        return out


    def log_prob(self, input: Tensor) -> Tensor:
        """ Given input tensor, and output of `self.head`,
               compute the log of the full distribution """
        head_output = self.head(input)
        head_output=self.dropout_module(head_output)
        return self._get_full_log_prob(input,head_output)

    def get_log_prob(self, input, target):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """

        if not self.training:
            return self.predict(input)

        bsz, length, dim = input.size()
        adaptive_input = input.contiguous().view(-1, dim)
        out = self.log_prob(adaptive_input)  # pytorch's adaptive_softmax need input shape (BxT,D)
        out = out.view(bsz, length, -1)
        out = out.float()  # convert to float32 to avoid loss==inf
        return out
