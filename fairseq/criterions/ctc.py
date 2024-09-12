# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional
import numpy as np
import logging

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round

logger = logging.getLogger(__name__)


@dataclass
class CtcCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=True,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="sentencepiece",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
                    "wordpiece, BPE symbols, etc. "
                    "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    ctc_entropy: float = field(
        default=0.0,
        metadata={"help": "weight of CTC entropy"},
    )
    intermedia_ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of intermedia CTC loss"},
    )
    target_ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of CTC loss for target sentence"},
    )
    target_intermedia_ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of intermedia CTC loss for target sentence"},
    )
    ctc_self_distill_weight: float = field(
        default=0.0,
        metadata={"help": "weight of the self distillation CTC loss"},
    )

    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )


@register_criterion("ctc", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask, ctc_weight=1.0):
        super().__init__(task)

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process
        self.sentence_avg = cfg.sentence_avg

        self.ctc_weight = ctc_weight
        self.intermedia_ctc_weight = cfg.intermedia_ctc_weight
        self.target_ctc_weight = cfg.target_ctc_weight
        self.target_intermedia_ctc_weight = cfg.target_intermedia_ctc_weight
        self.ctc_self_distill_weight = cfg.ctc_self_distill_weight
        self.ctc_entropy = cfg.ctc_entropy
        self.all_ctc_weight = self.ctc_weight + self.intermedia_ctc_weight + \
                              self.target_ctc_weight + self.target_intermedia_ctc_weight + \
                              self.ctc_self_distill_weight + self.ctc_entropy

        if self.all_ctc_weight > 0:
            # assert getattr(task, "src_dict", None) is not None, "CTC need a source dictionary."
            self.ctc_loss = torch.nn.CTCLoss(blank=self.blank_idx, reduction="sum", zero_infinity=True)

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])

        ntokens = sample["ntokens"]

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        loss, logging_output = self.compute_ctc_loss(model, sample, net_output, logging_output)
        return loss, sample_size, logging_output

    def get_loss(self, lprobs, targets_flat, input_lengths, transcript_lengths):
        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = self.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                transcript_lengths,
            )
        return ctc_loss

    def compute_ctc_loss(self, model, sample, net_output, logging_output):
        if "transcript" in sample:
            tokens = sample["transcript"]["tokens"]
        else:
            tokens = sample["target"]
        if "ctc_padding_mask" in net_output:
            non_padding_mask = ~net_output["ctc_padding_mask"][0]
        else:
            non_padding_mask = ~net_output["encoder_padding_mask"][0]
        # non_padding_mask = ~net_output["encoder_padding_mask"][0]

        mixup = False
        if "mixup" in net_output and net_output["mixup"] is not None:
            mixup = True
            mixup_coef = net_output["mixup"]["coef"]
            mixup_idx1 = net_output["mixup"]["index1"]
            mixup_idx2 = net_output["mixup"]["index2"]

        input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (tokens != self.pad_idx) & (
                tokens != self.eos_idx
        )
        if mixup:
            mask1 = pad_mask[mixup_idx1]
            mask2 = pad_mask[mixup_idx2]
            transcript_flat1 = tokens[[mixup_idx1]].masked_select(mask1)
            transcript_flat2 = tokens[mixup_idx2].masked_select(mask2)
            transcript_lengths1 = mask1.sum(-1)
            transcript_lengths2 = mask2.sum(-1)
            transcript_flat = [transcript_flat1, transcript_flat2]
            transcript_lengths = [transcript_lengths1, transcript_lengths2]
            loss_coef = [mixup_coef, 1 - mixup_coef]
        else:
            transcript_flat = [tokens.masked_select(pad_mask)]
            transcript_lengths = [pad_mask.sum(-1)]
            loss_coef = [1]

        ctc_loss = 0
        ctc_entropy = 0
        lprobs = None
        if self.ctc_weight > 0 and "ctc_logit" in net_output and len(net_output["ctc_logit"]) > 0:
            ctc_logit = net_output["ctc_logit"][0]
            lprobs = model.get_normalized_probs(
                [ctc_logit], log_probs=True
            ).contiguous()  # (T, B, C) from the encoder
            lprobs.batch_first = False

            for flat, lengths, coef in zip(transcript_flat, transcript_lengths, loss_coef):
                ctc_loss += self.get_loss(lprobs, flat, input_lengths, lengths) * coef

            if self.ctc_entropy > 0:
                from torch.distributions import Categorical
                # ctc_logit = ctc_logit.sort(dim=-1, descending=True)[0][:, :, 0:100]
                # ctc_logit = ctc_logit / ctc_logit.sum(dim=-1, keepdim=True)
                # cut_ctc_logit = ctc_logit.sort(dim=-1, descending=True)[0][:, :, 0:100]
                # ctc_entropy = Categorical(logits=cut_ctc_logit).entropy().sum()
                ctc_entropy = Categorical(logits=ctc_logit).entropy().sum()
                logging_output["ctc_entropy"] = utils.item(ctc_entropy.data)
            logging_output["ctc_loss"] = utils.item(ctc_loss.data)

        intermedia_ctc_num = 0
        intermedia_ctc_loss = 0
        if "intermedia_ctc_logits" in net_output:
            intermedia_ctc_num = len(net_output["intermedia_ctc_logits"])

        # calculate the intermedia CTC loss
        if self.intermedia_ctc_weight > 0 and intermedia_ctc_num > 0:
            for i in range(intermedia_ctc_num):
                out = net_output["intermedia_ctc_logits"][i]
                if type(out) == list:
                    inter_ctc_logit = out[0]
                    padding = ~out[1]
                    inter_input_lengths = padding.long().sum(-1)
                else:
                    inter_ctc_logit = out
                    inter_input_lengths = input_lengths

                inter_lprobs = model.get_normalized_probs(
                    [inter_ctc_logit], log_probs=True
                ).contiguous()  # (T, B, C) from the encoder
                inter_lprobs.batch_first = False

                for flat, lengths, coef in zip(transcript_flat, transcript_lengths, loss_coef):
                    intermedia_ctc_loss += self.get_loss(inter_lprobs, flat, inter_input_lengths, lengths) * coef

            intermedia_ctc_loss /= intermedia_ctc_num
            logging_output["intermedia_ctc_loss"] = utils.item(intermedia_ctc_loss.data)

            if lprobs is None:
                lprobs = inter_lprobs

        target_ctc_loss = 0
        target_intermedia_ctc_loss = 0

        # calculate the target CTC loss
        if self.target_ctc_weight > 0 or self.target_intermedia_ctc_weight:
            target = sample["target"]
            pad_mask = (target != self.pad_idx) & (target != self.eos_idx)

            if mixup:
                mask1 = pad_mask[mixup_idx1]
                mask2 = pad_mask[mixup_idx2]
                target_flat1 = target.masked_select(mask1)
                target_flat2 = target.masked_select(mask2)
                transcript_lengths1 = mask1.sum(-1)
                transcript_lengths2 = mask2.sum(-1)
                target_flat = [target_flat1, target_flat2]
                target_length = [transcript_lengths1, transcript_lengths2]
                loss_coef = [mixup_coef, 1 - mixup_coef]
            else:
                target_flat = [target.masked_select(pad_mask)]
                target_length = [pad_mask.sum(-1)]
                loss_coef = [1]

            if self.target_ctc_weight > 0:
                assert "target_ctc_logit" in net_output
                target_ctc_logit = net_output["target_ctc_logit"]

                tgt_lprobs = model.get_normalized_probs(
                    [target_ctc_logit], log_probs=True
                ).contiguous()  # (T, B, C) from the encoder
                tgt_lprobs.batch_first = False

                for flat, lengths, coef in zip(target_flat, target_length, loss_coef):
                    target_ctc_loss += self.get_loss(tgt_lprobs, flat, input_lengths, lengths) * coef

            target_intermedia_ctc_num = 0
            if "target_intermedia_ctc_logits" in net_output:
                target_intermedia_ctc_num = len(net_output["target_intermedia_ctc_logits"])

            for i in range(target_intermedia_ctc_num):
                out = net_output["target_intermedia_ctc_logits"][i]
                if type(out) == list:
                    inter_ctc_logit = out[0]
                    padding = ~out[1]
                    tgt_input_lengths = padding.long().sum(-1)
                else:
                    inter_ctc_logit = out
                    tgt_input_lengths = input_lengths

                tgt_inter_lprobs = model.get_normalized_probs(
                    [inter_ctc_logit], log_probs=True
                ).contiguous()  # (T, B, C) from the encoder
                tgt_inter_lprobs.batch_first = False

                for flat, lengths, coef in zip(target_flat, target_length, loss_coef):
                    target_intermedia_ctc_loss += self.get_loss(tgt_inter_lprobs, flat, tgt_input_lengths, lengths) * coef

            target_intermedia_ctc_loss /= target_intermedia_ctc_num
            logging_output["target_intermedia_ctc_loss"] = utils.item(target_intermedia_ctc_loss.data)

        # calculate the self distillation CTC loss
        ctc_self_distill_loss = 0
        ctc_self_distill_num = 0
        if self.ctc_weight > 0 and self.ctc_self_distill_weight > 0 and intermedia_ctc_num > 0:
            for i in range(intermedia_ctc_num):
                out = net_output["intermedia_ctc_logits"][i]
                if type(out) == list:
                    inter_ctc_logit = out[0]
                    padding = ~out[1]
                else:
                    inter_ctc_logit = out

                if inter_ctc_logit.size() != ctc_logit.size():
                    continue

                ctc_self_distill_num += 1
                loss = F.kl_div(
                    F.log_softmax(inter_ctc_logit, dim=-1, dtype=torch.float32),
                    F.softmax(ctc_logit, dim=-1, dtype=torch.float32),
                    reduction="none",
                )
                loss = loss.sum(-1).transpose(0, 1).masked_fill_(~non_padding_mask, 0.0)
                loss = loss.sum()
                ctc_self_distill_loss += loss

            ctc_self_distill_loss /= ctc_self_distill_num
            logging_output["ctc_self_distill_loss"] = utils.item(ctc_self_distill_loss.data)

        loss = \
            self.ctc_weight * ctc_loss + \
            self.intermedia_ctc_weight * intermedia_ctc_loss + \
            self.target_ctc_weight * target_ctc_loss + \
            self.target_intermedia_ctc_weight * target_intermedia_ctc_loss + \
            self.ctc_self_distill_weight * ctc_self_distill_loss + \
            self.ctc_entropy * ctc_entropy

        logging_output["all_ctc_loss"] = utils.item(loss.data)

        if torch.isnan(loss) or torch.isinf(loss) or utils.item(loss.data) < 0:
            logger.warning("Illegal loss %f!" % loss)
            if self.ctc_weight != 0:
                logger.warning("CTC loss %f!" % ctc_loss)
            if self.intermedia_ctc_weight != 0:
                logger.warning("Intermedia CTC loss %f!" % intermedia_ctc_loss)
            if self.target_ctc_weight != 0:
                logger.warning("Target CTC loss %f!" % target_ctc_loss)

        if not model.training and self.ctc_weight > 0:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()
                target = tokens
                if mixup:
                    idx = mixup_idx1
                    if mixup_coef < 0.5:
                        idx = mixup_idx2
                    target = target[idx]

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                        lprobs_t,
                        target,
                        input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                            t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        ctc_loss_sum = utils.item(
            sum(log.get("ctc_loss", 0) for log in logging_outputs)
        )
        ctc_entropy_sum = utils.item(
            sum(log.get("ctc_entropy", 0) for log in logging_outputs)
        )
        inter_ctc_loss_sum = utils.item(
            sum(log.get("intermedia_ctc_loss", 0) for log in logging_outputs)
        )
        target_ctc_loss_sum = utils.item(
            sum(log.get("target_ctc_loss", 0) for log in logging_outputs)
        )
        target_intermedia_ctc_loss_sum = utils.item(
            sum(log.get("target_intermedia_ctc_loss", 0) for log in logging_outputs)
        )
        ctc_self_distill_loss_sum = utils.item(
            sum(log.get("ctc_self_distill_loss", 0) for log in logging_outputs)
        )
        all_ctc_loss_sum = utils.item(
            sum(log.get("all_ctc_loss", 0) for log in logging_outputs)
        )
        # loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        if np.isnan(all_ctc_loss_sum) or np.isinf(all_ctc_loss_sum) or all_ctc_loss_sum < 0:
            logger.warning("Illegal loss %f!" % all_ctc_loss_sum)
        if all_ctc_loss_sum > 0:
            if "loss" not in logging_outputs[0]:
                metrics.log_scalar(
                    "loss",
                    all_ctc_loss_sum / sample_size / math.log(2),
                    sample_size,
                    round=3,
                )
            else:
                if all_ctc_loss_sum != ctc_loss_sum:
                    metrics.log_scalar(
                        "all_ctc_loss",
                        all_ctc_loss_sum / sample_size / math.log(2),
                        sample_size,
                        round=3,
                    )
        if ctc_loss_sum > 0:
            metrics.log_scalar(
                "ctc_loss",
                ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if ctc_entropy_sum > 0:
            metrics.log_scalar(
                "ctc_entropy",
                ctc_entropy_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if inter_ctc_loss_sum > 0:
            metrics.log_scalar(
                "intermedia_ctc_loss",
                inter_ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if target_ctc_loss_sum > 0:
            metrics.log_scalar(
                "target_ctc_loss",
                target_ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if target_intermedia_ctc_loss_sum > 0:
            metrics.log_scalar(
                "target_intermedia_ctc_loss",
                target_intermedia_ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )

        if ctc_self_distill_loss_sum > 0:
            metrics.log_scalar(
                "ctc_self_distill_loss",
                ctc_self_distill_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", ctc_loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        # wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        # metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "cer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            # metrics.log_derived(
            #     "raw_wer",
            #     lambda meters: safe_round(
            #         meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
            #     )
            #     if meters["_w_total"].sum > 0
            #     else float("nan"),
            # )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
