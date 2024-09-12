import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.modules import LayerNorm

logger = logging.getLogger(__name__)


class CTCCompressStrategy:
    @staticmethod
    def avg(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = 1.0 / same[1]
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix.to(device)

    @staticmethod
    def weighted(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix

    @staticmethod
    def softmax(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = F.softmax(
                    prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]], dtype=torch.float32
                )
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix


class Adapter(nn.Module):
    def __init__(self, dim, adapter_type, dictionary_size, embed_tokens=None, strategy=None):
        super().__init__()

        dim = dim

        self.adapter_type = adapter_type
        if self.adapter_type in ["linear", "league", "gated_league", "gated_league2"]:
            self.linear_adapter = nn.Sequential(
                nn.Linear(dim, dim),
                LayerNorm(dim),
                nn.ReLU(),
            )

        if self.adapter_type in ["context", "league", "gated_league", "gated_league2", "inter_league"]:
            self.embed_adapter = nn.Linear(dim, dictionary_size, bias=False)    # reverse for initialization
            if embed_tokens is not None:
                self.embed_adapter.weight = embed_tokens.weight

        if self.adapter_type == "gated_league":
            self.gate_linear = nn.Linear(2 * dim, dim)
        elif self.adapter_type == "gated_league2":
            self.gate_linear1 = nn.Linear(dim, dim)
            self.gate_linear2 = nn.Linear(dim, dim)

        if self.adapter_type == "shrink":
            assert strategy is not None
            self.ctc_compress = getattr(CTCCompressStrategy, strategy)
            logger.info("CTC Compress Strategy: %s" % strategy)
        elif self.adapter_type == "league":
            self.distribution_cutoff = strategy
            if self.distribution_cutoff is not None:
                logger.info("Distribution cutoff: %d" % int(strategy))

    def forward(self, x, padding=None):

        representation, distribution = x
        distribution = distribution.type_as(representation)
        seq_len, bsz, dim = representation.size()
        org_distribution = distribution
        distribution = distribution.contiguous().view(-1, distribution.size(-1))

        if self.adapter_type == "linear":
            out = self.linear_adapter(representation)

        elif self.adapter_type == "context":
            out = torch.mm(distribution, self.embed_adapter.weight.t()).view(seq_len, bsz, -1)

        elif self.adapter_type == "league":
            linear_out = self.linear_adapter(representation)
            if self.distribution_cutoff is not None:
                cutoff = min(int(self.distribution_cutoff), org_distribution.size(-1) - 1)
                threshold = org_distribution.sort(dim=-1, descending=True)[0][:, :, cutoff:cutoff+1]
                distribution = torch.where(
                    org_distribution > threshold, org_distribution, torch.zeros_like(org_distribution)
                )
                distribution = distribution.view(-1, distribution.size(-1))

            soft_out = torch.mm(distribution, self.embed_adapter.weight).view(seq_len, bsz, -1)
            out = linear_out + soft_out

        elif self.adapter_type == "gated_league":
            linear_out = self.linear_adapter(representation)
            soft_out = torch.mm(distribution, self.embed_adapter.weight.t()).view(seq_len, bsz, -1)

            coef = (self.gate_linear(torch.cat([linear_out, soft_out], dim=-1))).sigmoid()
            out = coef * linear_out + (1 - coef) * soft_out

        elif self.adapter_type == "inter_league":
            soft_out = torch.mm(distribution, self.embed_adapter.weight).view(seq_len, bsz, -1)
            out = representation + soft_out

        elif self.adapter_type == "none":
            out = representation

        elif self.adapter_type == "shrink":
            from itertools import groupby

            lengths = (~padding).long().sum(-1)
            with torch.no_grad():
                batch_predicted = []
                prob_ctc = org_distribution.transpose(0, 1)  # T x B x D -> B x T x D
                for b in range(prob_ctc.shape[0]):
                    predicted = prob_ctc[b][: lengths[b]].argmax(-1).tolist()
                    batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted)])

                new_lengths = [len(p) for p in batch_predicted]
                weights_matrix = self.ctc_compress(prob_ctc, batch_predicted, new_lengths,
                                                   prob_ctc.dtype, prob_ctc.device)

            # x is T x B x C -> B x C x T; weights_matrix is B x T x T'
            representation = representation.permute(1, 2, 0)
            compressed_output = representation.bmm(weights_matrix).type_as(representation)  # B x C x T'
            out = compressed_output.permute(2, 0, 1)

            out_lengths = lengths.new(new_lengths)
            padding = lengths_to_padding_mask(out_lengths)

        else:
            out = None
            logging.error("Unsupported adapter type: {}.".format(self.adapter_type))

        return out, padding
