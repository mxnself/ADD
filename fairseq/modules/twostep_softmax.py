

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

class TwoStepSoftmax(nn.Module):
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
            enable_gumbel_softmax=False,
            return_head_loss=False
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

        print("cutoffs:{}".format(cutoffs))
        print("dropout:{}".format(dropout))
        print("enable_gumbel_softmax:{}".format(enable_gumbel_softmax))
        print("head loss:{}".format(return_head_loss))

        self.in_features = input_dim
        self.n_classes = vocab_size
        self.cutoffs = cutoffs + [vocab_size]
        self.cutoffs_t=None
        self.div_value = factor
        self.head_bias = head_bias

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs)
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(self.in_features, len(self.cutoffs), bias=self.head_bias)
        self.tail = ModuleList()
        self.enable_gumbel_softmax=enable_gumbel_softmax
        self.enable_head_loss = return_head_loss

        self.dropout_module =None
        if dropout is not None and eval(dropout)>0.0:
            self.dropout_module = FairseqDropout(eval(dropout), module_name=self.__class__.__name__)

        for i in range(self.n_clusters):
            if i == 0:
                cluster_vocab_size = self.cutoffs[0]
            else:
                cluster_vocab_size = self.cutoffs[i] - self.cutoffs[i - 1]
            self.tail.append(Linear(input_dim, cluster_vocab_size, bias=False))

    def reset_parameters(self) -> None:
        self.head.reset_parameters()
        for liner in self.tail:
            liner.reset_parameters()

    def predict(self, input: Tensor):
        bsz, seqlen, dim = input.size()
        input = input.contiguous().view(-1, dim)
        head_output = self.head(input)
        head_logprob = log_softmax(head_output, dim=-1, onnx_trace=self.onnx_trace)
        max_cluster_prob_idx = head_output.argmax(dim=-1)

        tmp_index = [0] + self.cutoffs
        max_batch_size = max([tmp_index[i + 1] - tmp_index[i] for i in range(len(tmp_index) - 1)])

        vocab_log_prob = input.new_full((bsz * seqlen, max_batch_size), 0, dtype=torch.float32)

        cluster_idxs, counts = torch.unique(max_cluster_prob_idx, return_counts=True)
        tail_bias = input.new_empty((bsz * seqlen), dtype=torch.int32)
        for i, count in zip(cluster_idxs, counts):

            start_idx = tmp_index[i]
            stop_idx = tmp_index[i + 1]
            select_idx = (max_cluster_prob_idx == i)
            tail_bias[select_idx] = start_idx
            vocab_log_prob[select_idx, : stop_idx - start_idx]=log_softmax(self.tail[i](input[select_idx,:]), -1) + \
                                                                                        head_logprob[select_idx, i].unsqueeze(1)

        #[bsz*len,d]=>[bsz,len,d]
        vocab_log_prob = vocab_log_prob.view(bsz, seqlen, -1)

        return vocab_log_prob, tail_bias,None



    def log_prob(self, input: Tensor,target=None):
        """ Given input tensor, and output of `self.head`,
               compute the log of the full distribution """
        head_output = self.head(input)
        if self.dropout_module is not None:
            head_output = self.dropout_module(head_output)

        out = input.new_empty((head_output.size(0), self.n_classes))
        if self.enable_gumbel_softmax and False:
            head_logprob = F.gumbel_softmax(head_output, tau=1.0, hard=True, dim=-1)
        else:
            head_logprob = log_softmax(head_output, dim=-1, onnx_trace=self.onnx_trace)
        tmp_index = [0] + self.cutoffs
        for i in range(self.n_clusters):
            start_idx = tmp_index[i]
            stop_idx = tmp_index[i + 1]
            # if self.dropout_module is not None:
            #     cluster_output = self.dropout_module(self.tail[i](input))
            # else:
            cluster_output = self.tail[i](input)
            cluster_logprob = log_softmax(cluster_output, dim=-1, onnx_trace=self.onnx_trace)
            if self.enable_gumbel_softmax:
                output_logprob = cluster_logprob * head_logprob[:, i].unsqueeze(1)
            else:
                output_logprob = cluster_logprob + head_logprob[:, i].unsqueeze(1)
            out[:, start_idx:stop_idx] = output_logprob

        head_loss = None
        if self.enable_head_loss and (target is not None) and self.training:
            if self.cutoffs_t is None:
                self.cutoffs_t = torch.tensor(self.cutoffs, dtype=torch.int32).to(input.device)
            bsz,seq_len=target.size()
            offset = torch.bucketize(target.view(bsz*seq_len,1), self.cutoffs_t, right=True)
            head_loss= -head_logprob.gather(-1,offset).sum()
        return out, head_loss

    def get_log_prob(self, input, target=None):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """
        # if not self.training:
        #     return self.predict(input)

        bsz, length, dim = input.size()
        adaptive_input = input.contiguous().view(-1, dim)
        out,head_loss = self.log_prob(adaptive_input,target)  # pytorch's adaptive_softmax need input shape (BxT,D)
        out = out.view(bsz, length, -1)
        out = out.float()  # convert to float32 to avoid loss==inf
        out={"prob":out,"tail_bias":0,"head_loss":head_loss}
        return out
