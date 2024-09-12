import time

import torch

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


# from fairseq.modules.fairseq_dropout import FairseqDropout
# from fairseq.modules.quant_noise import quant_noise

def log_softmax(x, dim: int, onnx_trace: bool = False):
    #return F.log_softmax(x,dim=dim)
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)


def set_random_values(x, value_to_set,ratio=0.33):
    # 获取 Tensor 的总元素数量
    total_elements = x.numel()

    # 计算需要设置为特定值的元素数量（10%）
    num_elements_to_set = int(ratio * total_elements)

    # 生成随机索引，表示要设置为特定值的位置
    random_indices = torch.randperm(total_elements)[:num_elements_to_set]

    # 将选定的位置设置为特定值
    x.view(-1)[random_indices] = value_to_set


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

        # self.dropout_module = FairseqDropout(eval(dropout), module_name=self.__class__.__name__)

        for i in range(self.n_clusters):
            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = Sequential(
                Linear(self.in_features, hsz, bias=False),
                # nn.Dropout(self.dropout_module.p),
                Linear(hsz, osz, bias=False)
            )

            self.tail.append(projection)

    def reset_parameters(self) -> None:
        self.head.reset_parameters()
        for i2h, h2o in self.tail:
            i2h.reset_parameters()
            h2o.reset_parameters()

    # v1.0 解码使用方法
    # 目前v1.0在all_4k_7k_d01上使用bsz=1解码，dev-clean得分3.38，耗时21:29s (get_log_prob获取完整概率分布得分3.36,差距不大)
    # 当前计算其他cluster时，仅根据head中概率最大的选项来计算（使用argmax来获取概率最大的索引, 类似于greedy - search），
    # 是否要变为topk(防止出现这种情况：概率最大的在head，而第2、3大的在其他cluster中，以适配beam-search),
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
        bsz, seqlen, dim = input.size()
        input = input.view(-1, dim)

        # [bsz,len,d]=>[bsz*len,d]
        vocab_size = self.cutoffs[-1]

        head_output = self.head(input)

        # 以下3行代码耗时大
        head_max_prob_idx = head_output.argmax(dim=-1)
        not_in_shortlist = (head_max_prob_idx >= self.shortlist_size)
        all_in_shortlist = not (not_in_shortlist.any())

        # 初始化全部词的概率值(log_softmax值)为nan
        # 返回的vocab_log_prob会在 解码时的 sequence_generator.py的343行处理: 将nan替换为-math.inf(对应概率为0)，训练时的验证集不会调用该位置
        # vocab_log_prob = torch.full((bsz*seqlen,vocab_size), float('nan')).to(input).float（）# 后续log_softmax都是float32，用nan\-inf会导致dev的loss和ppl变成nan\inf，暂用-10000
        # vocab_log_prob = torch.full((bsz*seqlen,vocab_size), -10000).to(input).float()
        #
        vocab_log_prob = input.new_zeros((bsz * seqlen, vocab_size))  # 此方法耗时小
        # vocab_log_prob = torch.full((bsz * seqlen, vocab_size), 0).to(input).float() # 此方法耗时非常大!
        # return head_output

        # 全部词都在head中,仅计算head中的概率分布，其他的为nan
        if all_in_shortlist:
            vocab_log_prob[:, :self.cutoffs[0]] = log_softmax(head_output[:, :self.cutoffs[0]], -1)

        # 全部词都不在head中,则计算完整的词表概率分布
        elif not_in_shortlist.all():
            #print("11")
            vocab_log_prob = self._get_full_log_prob(input, head_output)


        # 部分词不在head中,这部分需要二次
        else:
            #print("22")
            log_prob = self._get_full_log_prob(input[not_in_shortlist],
                                               head_output[not_in_shortlist])

            # vocab_log_prob[:,:self.cutoffs[0]] = log_softmax(head_output[:,:self.cutoffs[0]], -1) # 此处应该对应各自变换调整的部分做相应log_softmax，
            # 在head里的调整cutoffs[0]的部分，不在head的计算完整词表概率分布
            vocab_log_prob[~not_in_shortlist, :self.cutoffs[0]] = log_softmax(
                head_output[~not_in_shortlist, :self.cutoffs[0]], -1)
            vocab_log_prob[not_in_shortlist, :] = log_softmax(log_prob, -1)

        # [bsz*len,d]=>[bsz,len,d]
        vocab_log_prob = vocab_log_prob.view(bsz, seqlen, vocab_size)
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
        head_output = self.dropout_module(head_output)
        return self._get_full_log_prob(input, head_output)

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


class LinearSoftmax(nn.Module):
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
            input_dim
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.in_features = input_dim
        self.head_bias = False
        self.head = Linear(self.in_features, self.vocab_size, bias=self.head_bias)

    def reset_parameters(self) -> None:
        self.head.reset_parameters()

    def predict(self, input: Tensor) -> Tensor:
        bsz, seqlen, dim = input.size()
        input = input.contiguous().view(-1, dim)
        head_output = self.head(input)
        return log_softmax(head_output, dim=-1).view(bsz, seqlen, self.vocab_size)


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
        self.n_clusters = len(self.cutoffs)
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(self.in_features, len(self.cutoffs), bias=self.head_bias)
        self.tail = ModuleList()

        self.dropout_module =None
        #self.dropout_module = FairseqDropout(eval(dropout), module_name=self.__class__.__name__)

        for i in range(self.n_clusters):
            if i == 0:
                cluster_vocab_size = self.cutoffs[0]
            else:
                cluster_vocab_size = self.cutoffs[i] - self.cutoffs[i - 1]
            self.tail.append(Linear(hidden_size, cluster_vocab_size, bias=False))

    def reset_parameters(self) -> None:
        self.head.reset_parameters()
        for liner in self.tail:
            liner.reset_parameters()

    def predict(self, input: Tensor,fake_cluster_idx=None) -> Tensor:
        bsz, seqlen, dim = input.size()
        input = input.view(-1, dim)

        head_output = self.head(input)
        # 3090: 基线的73%
        # cpu: 基线的2%

        max_cluster_prob_idx = head_output.argmax(dim=-1)
        if fake_cluster_idx is not None:
            max_cluster_prob_idx=fake_cluster_idx


        # 获取需要计算的cluster_idx
        assert bsz*seqlen==1
        tail_bias = input.new_empty((bsz * seqlen), dtype=torch.int32)
        i=max_cluster_prob_idx[0]
        tail_bias[0] = i
        vocab_log_prob= log_softmax(self.tail[i](input), -1)+log_softmax(head_output[:,i].unsqueeze(0),-1)

        #[bsz*len,d]=>[bsz,len,d]
        vocab_log_prob = vocab_log_prob.view(bsz, seqlen, -1)
        return vocab_log_prob

    def _get_full_log_prob(self, input, head_output):
        """ Given input tensor, and output of `self.head`,
        compute the log of the full distribution """

        out = input.new_empty((head_output.size(0), self.n_classes))
        head_logprob = log_softmax(head_output, dim=-1, onnx_trace=self.onnx_trace)

        tmp_index = [0] + self.cutoffs
        for i in range(self.n_clusters):
            start_idx = tmp_index[i]
            stop_idx = tmp_index[i + 1]
            cluster_output = self.tail[i](input)
            cluster_logprob = log_softmax(cluster_output, dim=-1, onnx_trace=self.onnx_trace)
            output_logprob = cluster_logprob + head_logprob[:,i].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_logprob

        return out

    def log_prob(self, input: Tensor) -> Tensor:
        """ Given input tensor, and output of `self.head`,
               compute the log of the full distribution """
        head_output = self.head(input)
        if self.dropout_module is not None:
            head_output = self.dropout_module(head_output)
        return self._get_full_log_prob(input, head_output)

    def get_log_prob(self, input, target=None):
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


if __name__ == '__main__':
    # 1*5：2.25， 2*5：1.46， 5*5：0.95， 10*5：0.8，20*5：0.63
    bsz = 1*1
    seq_len = 1
    hidden_size = 256
    vocab_size = 10000
    cutoffs = [1000, 2000,3000,4000,5000,6000,7000,8000,9000]
    device = "cpu"
    loop = 1000

    with torch.no_grad():

        embedding_base = nn.Embedding(vocab_size, hidden_size)
        embedding_1k = nn.Embedding(vocab_size//10, hidden_size)
        model_twostep = TwoStepSoftmax(vocab_size, hidden_size, cutoffs, "0.0", 2).to(device)
        model_base = LinearSoftmax(vocab_size, hidden_size).to(device)
        model_base_1k = LinearSoftmax(vocab_size//10, hidden_size).to(device)
        x = torch.rand((bsz, seq_len, hidden_size)).to(device)
        test_id=torch.randint(0,vocab_size,(bsz,seq_len))
        test_id_1k = torch.randint(0, vocab_size//10, (bsz, seq_len))
        fake_cluster_idx =  torch.tensor([0] * bsz*seq_len, dtype=torch.int32).to(device)
        for i in range(len(cutoffs)+1):
            set_random_values(fake_cluster_idx, i, 0.1)
        print(fake_cluster_idx)

        if device != "cpu": torch.cuda.synchronize(device=device)
        # 预热
        start_time = time.time()
        for i in range(loop * 2):
            output = model_twostep.predict(x, fake_cluster_idx)
            # print(output.size())
            if device != "cpu": torch.cuda.synchronize(device=device)
        end_time = time.time()
        print("warm-up time={}s".format(end_time - start_time))

        start_time = time.time()
        for i in range(loop):
            embedding_data = embedding_base(test_id)
            output = model_twostep.predict(x, fake_cluster_idx)
            # print(output.size())
            if device != "cpu": torch.cuda.synchronize(device=device)
        end_time = time.time()
        new_time=end_time - start_time

        print("new func time={}s".format(new_time))

        start_time = time.time()
        for i in range(loop):
            embedding_data=embedding_base(test_id)
            output = model_base.predict(x)
            # print(output.size())
            if device != "cpu": torch.cuda.synchronize(device=device)
        end_time = time.time()
        base_time=end_time - start_time
        print("base func time={}s".format(base_time))

        start_time = time.time()
        for i in range(loop):
            embedding_data = embedding_1k(test_id_1k)
            output = model_base_1k.predict(x)
            if device != "cpu": torch.cuda.synchronize(device=device)
        end_time = time.time()
        base_1k_time = end_time - start_time
        print("base_1k func time={}s".format(base_1k_time))

        print("与base的ratio={:.2f}".format(new_time/base_time))
        print("与理论base_1k的ratio={:.2f}".format(new_time / base_1k_time))

