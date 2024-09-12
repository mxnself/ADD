import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)


from torch import Tensor

logger = logging.getLogger(__name__)


@register_model("s2t_ctc")
class S2TCTCModel(FairseqEncoderModel):

    def __init__(self, encoder):
        super().__init__(encoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # subsampling
        parser.add_argument(
            "--subsampling-type",
            type=str,
            help="subsampling type, like conv1d and conv2d",
        )
        parser.add_argument(
            "--subsampling-layers",
            type=int,
            help="subsampling layers",
        )
        parser.add_argument(
            "--subsampling-filter",
            type=int,
            help="subsampling filter",
        )
        parser.add_argument(
            "--subsampling-kernel",
            type=int,
            help="subsampling kernel",
        )
        parser.add_argument(
            "--subsampling-stride",
            type=int,
            help="subsampling stride",
        )
        parser.add_argument(
            "--subsampling-norm",
            type=str,
            default="none",
            help="subsampling normalization type",
        )
        parser.add_argument(
            "--subsampling-activation",
            type=str,
            default="none",
            help="subsampling activation function type",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-type",
            type=str,
            default="selfattn",
            choices=[
                "local",
                "selfattn",
                "reduced",
                "rel_selfattn",
                "relative",
                "rel_pos",
                "rope",
                "abs",
                "transfer",
                "reduced_rel_pos",
            ],
            help="transformer encoder self-attention layer type"
        )
        parser.add_argument(
            "--relative-pos-enc",
            action="store_true",
            help="use relative position encoding for attention",
        )
        parser.add_argument(
            "--linear-att",
            action="store_true",
            help="use linear attention",
        )

        parser.add_argument(
            "--attention-reduced-method",
            type=str,
            default="conv",
            help="reduction method for attention",
        )
        parser.add_argument(
            "--attention-reduced-q",
            action="store_true",
            help="use reduction for query or not",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-type",
            type=str,
            default="selfattn",
            choices=[
                "selfattn",
                "rel_selfattn",
                "relative",
                "local",
            ],
            help="transformer decoder self-attention layer type"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument('--share-all-embeddings',
                            action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--max-encoder-relative-length', type=int, default=-1,
                            help='the max relative length')
        parser.add_argument('--max-decoder-relative-length', type=int, default=-1,
                            help='the max relative length')
        parser.add_argument('--k-only', default=False, action='store_true',
                            help='select the relative mode to map relative position information')
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--encoder-freeze-module",
            type=str,
            metavar="STR",
            help="freeze the module of the encoder",
        )
        parser.add_argument(
            "--decoder-freeze-module",
            type=str,
            metavar="STR",
            help="freeze the module of the decoder",
        )
        parser.add_argument(
            "--use-enc-dlcl",
            default=False,
            action='store_true',
            help="use dlcl encoder",
        )
        parser.add_argument(
            "--use-dec-dlcl",
            default=False,
            action='store_true',
            help="use dlcl encoder",
        )
        parser.add_argument('--init-value', type=str, default='avg', choices=['avg', 'one'],
                            help='how to init the learned weight matrix')
        parser.add_argument('--weight-type', type=str, default='scalar',
                            help='type of learned weight [scalar, scalar_n(n>1), vector]')
        parser.add_argument('--encoder-learnable', type=eval, default='True',
                            help='enable to learn weights for encoder')
        parser.add_argument('--decoder-learnable', type=eval, default='True',
                            help='enable to learn weights for decoder')
        parser.add_argument('--normalize-learned-weight', type=eval, default='False',
                            help='normalize learned weight by softmax')
        parser.add_argument('--normalize-embedding', type=eval, default='False',
                            help='normalize the input of embedding')
        parser.add_argument('--history-dropout', type=float, default=0.0, metavar='D',
                            help='dropout for history output')
        parser.add_argument('--history-window-size', type=int, default='-1',
                            help='how many past layers are considered. -1 means all')
        # CTC
        parser.add_argument(
            "--ctc-layer",
            default=0,
            type=int,
            help="the position of the ctc loss",
        )

        # local modeling
        parser.add_argument(
            '--hard-mask-window',
            type=float,
            metavar="D",
            default=0,
            help='window size of local mask'
        )
        parser.add_argument(
            '--gauss-mask-sigma',
            type=float,
            metavar="D",
            default=0,
            help='standard deviation of the gauss mask'
        )
        parser.add_argument(
            '--init-mask-weight',
            type=float,
            metavar="D",
            default=0.5,
            help='initialized weight for local mask'
        )

        # Conformer setting
        parser.add_argument(
            "--encoder-activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--macaron-style",
            default=False,
            type=bool,
            help="Whether to use macaron style for positionwise layer",
        )
        # Attention
        parser.add_argument(
            "--zero-triu",
            default=False,
            type=bool,
            help="If true, zero the upper triangular part of attention matrix.",
        )
        # Relative positional encoding
        parser.add_argument(
            "--rel-pos-type",
            type=str,
            default="legacy",
            choices=["legacy", "latest"],
            help="Whether to use the latest relative positional encoding or the legacy one."
                 "The legacy relative positional encoding will be deprecated in the future."
                 "More Details can be found in https://github.com/espnet/espnet/pull/2816.",
        )
        # CNN module
        parser.add_argument(
            "--use-cnn-module",
            default=False,
            type=bool,
            help="Use convolution module or not",
        )
        parser.add_argument(
            "--cnn-module-kernel",
            default=31,
            type=int,
            help="Kernel size of convolution module.",
        )

        # Simultaneous speech translation
        parser.add_argument(
            "--simul",
            default=False,
            action="store_true",
            help="Simultaneous speech translation or not",
        )
        # interleaved dropout
        parser.add_argument('--interleave-dropout', type=int,
                            help='interleaved dropout probability')
        parser.add_argument('--cl-dropout',
                            action="store_true",
                            default=False,
                            help='interleaved dropout probability')
        parser.add_argument('--cl-dropout-epoch',
                            type=int,
                            default=None,
                            help='interleaved dropout probability')
        parser.add_argument('--cl-dropout-strategy',
                            type=str,
                            help='interleaved dropout probability')

        # pds setting
        parser.add_argument(
            "--pds-stages",
            type=int,
            help="the number of the stage",
        )
        parser.add_argument(
            "--pds-layers",
            type=str,
            help="the number of the encoder layers in each stage",
        )
        parser.add_argument(
            "--pds-ratios",
            type=str,
            help="the ratio of the down-sampling in each stage",
        )
        parser.add_argument(
            "--pds-ds-method",
            type=str,
            choices=["glu", "conv", "proj", "fusion"],
            help="the down-sampling method",
        )
        parser.add_argument(
            "--pds-embed-dims",
            type=str,
            help="the embedding dimension in each stage",
        )
        parser.add_argument(
            "--pds-kernel-sizes",
            type=str,
            help="the kernel size of the down-sampling module in each stage",
        )
        parser.add_argument(
            "--pds-embed-norm",
            action="store_true",
            help="use layer norm in the down-sampling module",
        )
        parser.add_argument(
            "--pds-position-embed",
            type=str,
            help="use the position embedding or not before each encoding",
        )
        parser.add_argument(
            "--pds-attn-heads",
            type=str,
            help="the number of the attention heads in each stage",
        )
        parser.add_argument(
            "--pds-attn-ds-ratios",
            type=str,
            help="the ratio of the down-sampling in the self attention module",
        )
        parser.add_argument(
            "--pds-ffn-ratios",
            type=str,
            help="the ratio of the ffn  in each stage",
        )
        parser.add_argument(
            "--pds-conv-strides",
            type=str,
            help="the strides of the convolutional module (conformer) in each stage",
        )
        parser.add_argument(
            "--pds-attn-strides",
            type=str,
            help="the strides of the attention module (conformer) in each stage",
        )
        parser.add_argument(
            "--pds-fusion",
            action="store_true",
            help="use the representation fusion method",
        )
        parser.add_argument(
            "--pds-fusion-method",
            type=str,
            help="the fusion method",
        )
        parser.add_argument(
            "--pds-dropout",
            type=float,
            help="dropout in each stage",
        )
        parser.add_argument(
            "--pds-ctc",
            type=str,
            help="use the ctc after each stage",
        )

        # intermedia CTC loss
        parser.add_argument(
            "--intermedia-ctc-layers",
            default=None,
            type=str,
            help="the position of the ctc loss, separated by comma ",
        )
        parser.add_argument(
            "--intermedia-adapter",
            default="none",
            type=str,
            help="type of intermedia adapter",
        )
        parser.add_argument(
            "--intermedia-distribution-cutoff",
            default=None,
            type=int,
            help="cutoff of the distribution",
        )
        parser.add_argument(
            "--intermedia-drop-prob",
            default=0,
            type=float,
            help="probability of dropping the followed layers",
        )

        # encoder
        parser.add_argument(
            "--encoder-type",
            default="transformer",
            type=str,
            help="encoder type",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2TCTCEncoder(args, task)
        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )

        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        encoder = cls.build_encoder(args, task)
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logging.info("freeze the encoder module: {}".format(args.encoder_freeze_module))

        return cls(encoder)

    def get_normalized_probs(
            self,
            net_output,
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (T, B, D) tensor
        if isinstance(net_output, list):
            logits = net_output[0]
        else:
            logits = net_output["ctc_logit"][0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

    def forward(self, src_tokens, src_lengths, prev_output_tokens=None):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

        return encoder_out


class S2TCTCEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None):
        super().__init__(None)

        setattr(args, "ctc_weight", 1.0)
        encoder_type = getattr(args, "encoder_type", "transformer")
        if encoder_type == "transformer":
            from .s2t_transformer import S2TTransformerEncoder
            self.encoder = S2TTransformerEncoder(args, task)
        elif encoder_type == "pds":
            from .pdss2t_transformer import PDSS2TTransformerEncoder
            self.encoder = PDSS2TTransformerEncoder(args, task)
        else:
            logger.error("Unsupported architecture: %s." % encoder_type)

        return

    def forward(self, src_tokens, src_lengths, **kwargs):

        return self.encoder(src_tokens, src_lengths, **kwargs)

    def reorder_encoder_out(self, encoder_out, new_order):
        return self.encoder.reorder_encoder_out(encoder_out, new_order)


class CTCDecoder(object):

    def __init__(self, models, args, dictionary, blank_idx):
        self.dict = dictionary
        self.vocab_size = len(dictionary)

        self.blank = blank_idx
        self.pad = dictionary.pad()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.vocab_size = len(dictionary)
        self.beam_size = args.beam
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(self.beam_size, self.vocab_size - 1)

        # from fairseq.sequence_generator import EnsembleModel
        from fairseq.sequence_generator import EnsembleModel
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.model = models[0]
        self.model.eval()

        self.lm_model = getattr(args, "kenlm_model", None)
        self.lm_weight = getattr(args, "lm_weight", 0)
        if self.lm_model is not None:
            self.lm_model.eval()

        from ctcdecode import CTCBeamDecoder
        self.ctc_decoder = CTCBeamDecoder(
            dictionary.symbols,
            model_path=self.lm_model,
            alpha=self.lm_weight,
            beta=0,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=self.beam_size,
            num_processes=20,
            blank_id=self.blank,
            log_probs_input=False
        )

    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):

        net_input = sample["net_input"]

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        src_tokens = net_input["src_tokens"]
        src_lengths = net_input["src_lengths"]
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        encoder_outs = self.model(src_tokens=src_tokens,
                                  src_lengths=src_lengths)

        ctc_logit = encoder_outs["ctc_logit"][0].transpose(0, 1)
        logit_length = (~encoder_outs["encoder_padding_mask"][0]).long().sum(-1)
        beam_results, beam_scores, time_steps, out_lens = self.ctc_decoder.decode(
            utils.softmax(ctc_logit, -1), logit_length
        )

        finalized = []
        for idx in range(bsz):
            hypos = []
            for beam_idx in range(beam_size):
                hypo = dict()
                length = out_lens[idx][beam_idx]
                scores = beam_scores[idx, beam_idx]
                hypo["tokens"] = beam_results[idx, beam_idx, : length]
                hypo["score"] = scores
                hypo["attention"] = None
                hypo["alignment"] = None
                hypo["positional_scores"] = torch.Tensor([scores / length] * length)
                hypos.append(hypo)
            finalized.append(hypos)
        return finalized


@register_model_architecture(model_name="s2t_ctc", arch_name="s2t_ctc")
def base_architecture(args):
    # Convolutional subsampler
    args.subsampling_type = getattr(args, "subsampling_type", "conv1d")
    args.subsampling_layers = getattr(args, "subsampling_layers", 2)
    args.subsampling_filter = getattr(args, "subsampling_filter", 1024)
    args.subsampling_kernel = getattr(args, "subsampling_kernel", 5)
    args.subsampling_stride = getattr(args, "subsampling_stride", 2)
    args.subsampling_norm = getattr(args, "subsampling_norm", "none")
    args.subsampling_activation = getattr(args, "subsampling_activation", "glu")

    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_type = getattr(args, "encoder_attention_type", "selfattn")
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)

    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    # CTC
    args.ctc_layer = getattr(args, "ctc_layer", 0)

    # Conformer
    args.macaron_style = getattr(args, "macaron_style", False)
    args.use_cnn_module = getattr(args, "use_cnn_module", False)
    args.cnn_module_kernel = getattr(args, "cnn_module_kernel", 31)

    # settings for DLCL
    args.use_enc_dlcl = getattr(args, "use_enc_dlcl", False)
    args.use_dec_dlcl = getattr(args, "use_dec_dlcl", False)
    args.init_value = getattr(args, 'init_value', 'avg')
    args.weight_type = getattr(args, 'weight_type', 'scalar')
    args.encoder_learnable = getattr(args, 'encoder_learnable', True)
    args.normalize_embed = getattr(args, 'normalize_embed', False)
    args.history_dropout = getattr(args, 'history_dropout', 0.0)
    args.history_window_size = getattr(args, 'history_window_size', -1)

    # Relative position encoding
    args.max_encoder_relative_length = getattr(args, 'max_encoder_relative_length', -1)
    args.k_only = getattr(args, 'k_only', True)

    # local modeling
    args.hard_mask_window = getattr(args, 'hard_mask_window', 0)
    args.gauss_mask_sigma = getattr(args, 'gauss_mask_sigma', 0)
    args.init_mask_weight = getattr(args, 'init_mask_weight', 0)

    # interleaved dropout
    args.interleave_dropout = getattr(args, "interleave_dropout", None)
    args.cl_dropout = getattr(args, "cl_dropout", False)
    args.cl_dropout_epoch = getattr(args, "cl_dropout_epoch", None)
    args.cl_dropout_strategy = getattr(args, "cl_dropout_strategy", "linear")

    # PDS
    args.pds_stages = getattr(args, "pds_stages", None)
    args.pds_layers = getattr(args, "pds_layers", None)
    args.pds_ratios = getattr(args, "pds_ratios", None)

    args.pds_ds_method = getattr(args, "pds_ds_method", "conv")
    args.pds_embed_dims = getattr(args, "pds_embed_dims", None)
    args.pds_embed_norm = getattr(args, "pds_embed_norm", True)
    args.pds_position_embed = getattr(args, "pds_position_embed", None)

    args.pds_attn_heads = getattr(args, "pds_attn_heads", None)
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", None)
    args.pds_cnn_kernel_sizes = getattr(args, "pds_cnn_kernel_sizes", None)

    args.pds_attn_ds_ratios = getattr(args, "pds_attn_ds_ratios", "1_1_1_1")
    args.pds_conv_strides = getattr(args, "pds_conv_strides", "1_1_1_1")
    args.pds_attn_strides = getattr(args, "pds_attn_strides", "1_1_1_1")

    args.ctc_layer = getattr(args, "ctc_layer", 0)
    args.pds_dropout = getattr(args, "pds_dropout", args.dropout)

    args.pds_fusion = getattr(args, "pds_fusion", False)
    args.pds_fusion_method = getattr(args, "pds_fusion_method", "all_conv")

    # intermedia CTC
    args.intermedia_ctc_layers = getattr(args, "intermedia_ctc_layers", None)
    args.intermedia_adapter = getattr(args, "intermedia_adapter", None)


@register_model_architecture("s2t_ctc", "s2t_ctc_s")
def s2t_ctc_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_s_relative")
def s2t_ctc_s_relative(args):
    args.max_encoder_relative_length = 100
    args.k_only = True
    s2t_ctc_s(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_xs")
def s2t_ctc_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_ctc_s(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_sp")
def s2t_ctc_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_ctc_s(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_m")
def s2t_ctc_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_mp")
def s2t_ctc_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_ctc_m(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_l")
def s2t_ctc_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_ctc", "s2t_ctc_lp")
def s2t_ctc_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_ctc_l(args)
