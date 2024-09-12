import logging
import math
from functools import reduce
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text import S2TTransformerModel
from fairseq.modules.speech_to_text import CTC, Adapter

from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    RelPositionalEncoding,
    LegacyRelPositionalEncoding,
    PDSTransformerEncoderLayer,
    DownSampleConvolutionModule
)
from fairseq.modules.speech_to_text import (
    subsampling
)

logger = logging.getLogger(__name__)


def lengths_to_padding_mask_with_maxlen(lens, max_length):
    bsz = lens.size(0)
    mask = torch.arange(max_length).to(lens.device).view(1, max_length)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_length)
    return mask


class Permute120(nn.Module):

    @staticmethod
    def forward(x):
        return x.permute(1, 2, 0)


class Permute201(nn.Module):

    @staticmethod
    def forward(x):
        return x.permute(2, 0, 1)


class Downsampling(nn.Module):
    # down-sampling module
    def __init__(
            self,
            reduced_way: str,
            embed_norm: bool,
            in_channels: int,
            out_channels: int,
            kernel_sizes: int,
            stride: int,
            padding: int,
    ):
        super().__init__()

        self.stride = stride
        self.reduced_way = reduced_way

        if stride == 0:
            return
        # default conv
        if self.reduced_way == "conv":
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_sizes, stride=stride, padding=padding),
            )
        elif self.reduced_way == "proj":
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_sizes, stride=stride, padding=padding),
                nn.ReLU()
            )
        else:
            logger.error("Unsupported reduced way!")

        self.embed_norm = embed_norm
        if self.embed_norm:
            self.norm = LayerNorm(out_channels)

    def forward(self, x, lengths):
        if self.stride == 0:
            return x, lengths

        seq_len, bsz, dim = x.size()
        assert seq_len % self.stride == 0, "The sequence length %d must be a multiple of %d." % (seq_len, self.stride)

        # mask batch padding
        if not torch.all(lengths == seq_len):
            padding_mask = lengths_to_padding_mask_with_maxlen(lengths, seq_len)  # bsz, seq_len
            mask_pad = padding_mask.unsqueeze(2)
            if mask_pad is not None:
                x = x.transpose(0, 1)
                x.masked_fill_(mask_pad, 0.0)
                x = x.transpose(0, 1)

        lengths = ((lengths.float() - 1) / self.stride + 1).floor().long()
        out_seq_len = max(lengths).item()
        if self.reduced_way == "proj":
            x = x.permute(1, 2, 0)  # bsz, dim, seq_len
            x = nn.functional.adaptive_avg_pool1d(x, out_seq_len)
            x = self.conv(self.in_norm(x))
            x = x.permute(2, 0, 1)  # seq_len, bsz, dim
        else:
            x = x.permute(1, 2, 0)  # B * D * T
            x = self.conv(x)
            x = x.permute(2, 0, 1)  # T * B * D
        if self.embed_norm:
            x = self.norm(x)

        # mask batch padding
        if not torch.all(lengths == x.size(-1)):
            padding_mask = lengths_to_padding_mask_with_maxlen(lengths, x.size(0))
            mask_pad = padding_mask.unsqueeze(2)
            if mask_pad is not None:
                x = x.transpose(0, 1)
                x.masked_fill_(mask_pad, 0.0)
                x = x.transpose(0, 1)

        return x, lengths


@register_model("pdss2t_transformer")
class PDSS2TTransformerModel(S2TTransformerModel):
    """Progressive down-sampling for acoustic encoding."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

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
                "rel_pos_legacy",
                "rel_pos",
                "rope",
                "abs",
                "transfer",
                "reduced_rel_pos",
            ],
            help="transformer encoder self-attention layer type"
        )
        # transfer
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

        # reduced attention
        parser.add_argument(
            "--attention-reduced-method",
            type=str,
            default="conv",
            help="reduction method for attention",
        )
        parser.add_argument(
            "--attention-reduced-q",
            action="store_true",
            help="use reduction for query or not"
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
            help="the ratios of the down-sampling in the self attention module",
        )
        parser.add_argument(
            "--pds-ffn-ratios",
            type=str,
            help="the ratio of the ffn in each stage",
        )
        parser.add_argument(
            "--pds-cnn-kernel-sizes",
            type=str,
            help="the kernel size of convolutional modules in Conformer",
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

        # intermedia ctc
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
        parser.add_argument(
            "--intermedia-temperature",
            default=1,
            type=float,
            help="temperature of the intermedia ctc probability",
        )
        # mixup
        parser.add_argument(
            "--inter-mixup",
            action="store_true",
            help="use mixup or not",
        )
        parser.add_argument(
            "--inter-mixup-layer",
            default=None,
            type=int,
            help="the layers for mixup",
        )
        parser.add_argument(
            "--inter-mixup-beta",
            default=0.5,
            type=float,
            help="the coefficient beta for mixup",
        )
        parser.add_argument(
            "--inter-mixup-prob",
            default=1,
            type=float,
            help="the probability for mixup",
        )
        parser.add_argument(
            "--inter-mixup-ratio",
            default=1,
            type=float,
            help="the ratio for mixup",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = PDSS2TTransformerEncoder(args, task, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )

        return encoder


class PDSS2TTransformerEncoder(FairseqEncoder):
    """Progressive Down-sampling for Acoustic Encoding"""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)

        self.padding_idx = 1
        self.attn_type = getattr(args, "encoder_attention_type", "selfattn")
        self.embed_dim = args.encoder_embed_dim

        self.dropout = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.pds_dropout = FairseqDropout(
            p=getattr(args, "pds_dropout", args.dropout), module_name=self.__class__.__name__
        )

        self.pds_stages = getattr(args, "pds_stages", 4)
        self.pds_layers = [int(n) for n in args.pds_layers.split("_")]
        self.layers = sum(self.pds_layers)
        self.pds_ratios = [int(n) for n in args.pds_ratios.split("_")]

        # down-sampling module
        self.pds_ds_method = args.pds_ds_method
        self.pds_embed_dims = [int(n) for n in args.pds_embed_dims.split("_")]
        self.pds_kernel_sizes = [int(n) for n in args.pds_kernel_sizes.split("_")]
        self.pds_embed_norm = args.pds_embed_norm
        self.pds_position_embed = [int(n) for n in args.pds_position_embed.split("_")]
        self.pds_attn_heads = [int(n) for n in args.pds_attn_heads.split("_")]
        self.pds_ffn_ratios = [int(n) for n in args.pds_ffn_ratios.split("_")]
        self.pds_cnn_kernel_sizes = \
            [int(n) for n in args.pds_cnn_kernel_sizes.split("_")] \
                if getattr(args, "pds_cnn_kernel_sizes", None) is not None else None
        self.pds_attn_ds_ratios = \
            [int(n) for n in args.pds_attn_ds_ratios.split("_")] if args.pds_attn_ds_ratios is not None else None

        self.pds_conv_strides = \
            [int(n) for n in args.pds_conv_strides.split("_")] if args.pds_conv_strides is not None else None
        self.pds_attn_strides = \
            [int(n) for n in args.pds_attn_strides.split("_")] if args.pds_attn_strides is not None else None

        # fusion
        self.pds_fusion = args.pds_fusion
        self.pds_fusion_method = args.pds_fusion_method

        self.pds_fusion_transform = "conv"
        if len(self.pds_fusion_method.split("_")) == 2:
            items = self.pds_fusion_method.split("_")
            self.pds_fusion_method = items[0]
            self.pds_fusion_transform = items[1]

        fusion_stages_num = 0
        if self.pds_fusion:
            if self.pds_fusion_method == "all":
                fusion_stages_num = self.pds_stages
            elif self.pds_fusion_method == "same":
                for dim in self.pds_embed_dims:
                    if dim == self.embed_dim:
                        fusion_stages_num += 1
            else:
                logger.error("Unsupported fusion!")
            if fusion_stages_num == 1:
                fusion_stages_num = 0
        self.fusion_stages_num = fusion_stages_num

        args.pds_ctc = getattr(args, "pds_ctc", None)
        self.pds_ctc = [int(n) for n in args.pds_ctc.split("_")] if args.pds_ctc is not None else None
        inter_ctc_module = None
        inter_adapter = None

        for i in range(self.pds_stages):
            num_layers = self.pds_layers[i]
            ds_ratio = self.pds_ratios[i]

            embed_dim = self.pds_embed_dims[i]
            kernel_size = self.pds_kernel_sizes[i]
            use_pos_embed = self.pds_position_embed[i]
            use_ctc = self.pds_ctc[i] if self.pds_ctc is not None else False

            num_head = self.pds_attn_heads[i]
            ffn_ratio = self.pds_ffn_ratios[i]
            cnn_kernel_size = self.pds_cnn_kernel_sizes[i] if self.pds_cnn_kernel_sizes is not None else None
            attn_ds_ratio = self.pds_attn_ds_ratios[i] \
                if self.pds_conv_strides is not None and self.attn_type == "reduced" else 1
            conv_stride = self.pds_conv_strides[i] if self.pds_conv_strides is not None else 1
            attn_stride = self.pds_attn_strides[i] if self.pds_attn_strides is not None else 1
            if conv_stride != 1 or attn_stride != 1:
                expand_embed_dim = embed_dim if i == self.pds_stages - 1 else self.pds_embed_dims[i + 1]
            else:
                expand_embed_dim = None

            logger.info("The stage {}: layer {}, down-sample ratio {}, embed dim {}, "
                        "kernel size {}, position embed {}, ffn ratio {}, num head {}, "
                        "attn down-sample ratio {}, conv stride {}, attn stride {}, "
                        "fusion {}, fusion method {}, fusion transformer {}.".
                        format(i, num_layers, ds_ratio, embed_dim,
                               kernel_size, use_pos_embed, ffn_ratio, num_head,
                               attn_ds_ratio, conv_stride, attn_stride,
                               self.pds_fusion, self.pds_fusion_method, self.pds_fusion_transform))

            if i == 0:
                self.embed_scale = math.sqrt(embed_dim)
                if args.no_scale_embedding:
                    self.embed_scale = 1.0

            # down-sampling
            if ds_ratio == -1:
                downsampling = subsampling(args, embed_dim)
            else:
                downsampling = Downsampling(
                    self.pds_ds_method,
                    self.pds_embed_norm,
                    args.input_feat_per_channel * args.input_channels if i == 0 else self.pds_embed_dims[i - 1],
                    embed_dim,
                    kernel_sizes=kernel_size,
                    stride=ds_ratio,
                    padding=(kernel_size - 1) // 2,
                )

            # position encoding
            if use_pos_embed:
                if self.attn_type in ["rel_pos", "reduced_rel_pos"]:
                    pos_embed = RelPositionalEncoding(
                        args.max_source_positions, embed_dim
                    )
                elif self.attn_type in ["rel_selfattn", "rel_pos_legacy"]:
                    pos_embed = LegacyRelPositionalEncoding(
                        embed_dim, args.dropout, args.max_source_positions
                    )
                elif self.attn_type == "rope":
                    pos_embed = None
                else:  # Use absolute positional embedding
                    pos_embed = PositionalEmbedding(
                        args.max_source_positions, embed_dim, self.padding_idx
                    )
            else:
                pos_embed = None

            stage = nn.ModuleList([
                PDSTransformerEncoderLayer(
                    args,
                    embed_dim,
                    ffn_ratio,
                    num_head,
                    attn_ds_ratio,
                    conv_stride=conv_stride if layer_idx == num_layers - 1 else 1,
                    attn_stride=attn_stride if layer_idx == num_layers - 1 else 1,
                    expand_embed_dim=expand_embed_dim if layer_idx == num_layers - 1 else None,
                    cnn_kernel_size=cnn_kernel_size,
                )
                for layer_idx in range(num_layers)])

            # representation fusion
            fusion_pre_layer_norm = None
            fusion_post_layer_norm = None
            fusion_downsampling = None
            if fusion_stages_num != 0:
                if self.pds_fusion_method == "all" or (
                        self.pds_fusion_method == "same" and self.embed_dim == embed_dim
                ):
                    if i != self.pds_stages - 1:
                        ratio = reduce(lambda a, b: a * b, self.pds_ratios[i + 1:])
                    else:
                        ratio = 1

                    fusion_pre_layer_norm = LayerNorm(embed_dim)
                    fusion_post_layer_norm = LayerNorm(self.embed_dim)
                    # default conv
                    if self.pds_fusion_transform == "conv":
                        fusion_downsampling = nn.Sequential(
                            Permute120(),
                            nn.Conv1d(embed_dim, self.embed_dim,
                                      kernel_size=ratio,
                                      stride=ratio),
                            nn.BatchNorm1d(self.embed_dim),
                            nn.ReLU(),
                            Permute201(),
                        )
                    elif self.pds_fusion_transform == "pool":
                        fusion_downsampling = nn.Sequential(
                            nn.Conv1d(embed_dim, self.embed_dim, kernel_size=1),
                            nn.BatchNorm1d(self.embed_dim),
                            nn.ReLU(),
                            Permute201(),
                        )
                    elif self.pds_fusion_transform == "conv2":
                        fusion_downsampling = DownSampleConvolutionModule(
                            self.embed_dim,
                            kernel_size=ratio,
                            stride=ratio,
                        )
                    else:
                        logger.error("Unsupported fusion transform!")

            # intermedia modules for each stage
            if use_ctc:
                if inter_ctc_module is None:
                    ctc = CTC(embed_dim,
                              dictionary_size=len(task.source_dictionary),
                              dropout=args.dropout,
                              need_layernorm=True)
                    if task.source_dictionary == task.target_dictionary and embed_tokens is not None:
                        ctc.ctc_projection.weight = embed_tokens.weight

                    inter_ctc_module = ctc
                else:
                    ctc = inter_ctc_module
                if i != self.pds_stages - 1:
                    if inter_adapter is None:
                        strategy = None
                        if args.intermedia_adapter == "shrink":
                            strategy = getattr(args, "ctc_compress_strategy", "avg")
                        adapter = Adapter(embed_dim, args.intermedia_adapter,
                                          len(task.source_dictionary), strategy=strategy)
                        inter_adapter = adapter
                    else:
                        adapter = inter_adapter
                else:
                    adapter = Adapter(embed_dim, "none",
                                      len(task.source_dictionary))
            else:
                ctc = None
                adapter = None

            setattr(self, f"downsampling{i + 1}", downsampling)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"stage{i + 1}", stage)
            setattr(self, f"fusion_downsampling{i + 1}", fusion_downsampling)
            setattr(self, f"fusion_pre_layer_norm{i + 1}", fusion_pre_layer_norm)
            setattr(self, f"fusion_post_layer_norm{i + 1}", fusion_post_layer_norm)

            setattr(self, f"ctc{i + 1}", ctc)
            setattr(self, f"adapter{i + 1}", adapter)

        if self.fusion_stages_num != 0:
            self.fusion_weight = nn.Parameter(torch.Tensor(fusion_stages_num).fill_(1.0))
            self.fusion_weight.data = self.fusion_weight.data / self.fusion_weight.data.sum(0, keepdim=True)

        # self.use_ctc = "sate" in args.arch or \
        #                (getattr(args, "criterion", "") == "ctc") or \
        #                (("ctc" in getattr(args, "criterion", "")) and
        #                 (getattr(args, "ctc_weight", False) > 0))
        self.use_ctc = "sate" in args.arch or (getattr(args, "ctc_weight", 0) > 0)
        if self.use_ctc:
            # self.ctc_layer = (args.ctc_layer + self.layers) % self.layers
            # self.ctc_layer = self.layers if self.ctc_layer == 0 else self.ctc_layer
            # self.inter_ctc = True if self.ctc_layer != self.layers or self.fusion_stages_num != 0 else False

            self.ctc_layer = args.ctc_layer
            self.inter_ctc = True if self.ctc_layer != 0 else False
            if self.inter_ctc:
                logger.info("Intermedia CTC loss in layer %d" % self.ctc_layer)

            # embed_dim = self.pds_embed_dims[-1]
            embed_dim = self.embed_dim
            if self.inter_ctc:
                ctc_layer = self.ctc_layer
                for i in range(self.pds_stages):
                    ctc_layer -= self.pds_layers[i]
                    if ctc_layer <= 0:
                        embed_dim = self.pds_embed_dims[i]
                        break
            if inter_ctc_module is None or embed_dim != inter_ctc_module.embed_dim:
                self.ctc = CTC(embed_dim,
                               dictionary_size=len(task.source_dictionary),
                               dropout=args.dropout,
                               need_layernorm=True if self.inter_ctc else False)

                if task.source_dictionary == task.target_dictionary and \
                        embed_tokens is not None and embed_dim == embed_tokens.embedding_dim:
                    self.ctc.ctc_projection.weight = embed_tokens.weight
            else:
                self.ctc = inter_ctc_module

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

        self.intermedia_temperature = getattr(args, "intermedia_temperature", 1)

        # mixup
        self.mixup = getattr(args, "inter_mixup", False)
        if self.mixup:
            self.mixup_layer = int(args.inter_mixup_layer)
            self.mixup_prob = float(getattr(args, "inter_mixup_prob", 1.0))
            self.mixup_ratio = float(getattr(args, "inter_mixup_ratio", 1.0))
            beta = float(args.inter_mixup_beta)

            from torch.distributions import Beta
            self.beta = Beta(torch.Tensor([beta]), torch.Tensor([beta]))
            logger.info("Use mixup in layer %d with beta %.2f, prob %.2f, ratio %.2f." % (
                self.mixup_layer, beta, self.mixup_prob, self.mixup_ratio)
                        )

        # gather cosine similarity
        self.gather_cos_sim = getattr(args, "gather_cos_sim", False)
        self.dis = 2
        self.cos_sim = dict()

    def add_to_dict(self, x, dis, idx):
        sim = 0
        seq_len = x.size(0)
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        for i in range(dis, seq_len - dis):
            a = x[i, :, :]
            for j in range(-dis, dis + 1):
                if j == 0:
                    continue
                b = x[i + j, :, :]
                sim_j = cos(a, b).mean()
                sim += sim_j
        sim = sim / 2 / dis / (seq_len - 2 * dis)

        if idx not in self.cos_sim:
            self.cos_sim[idx] = []
        self.cos_sim[idx].append(float(sim))

    def apply_mixup(self, x, encoder_padding_mask):
        batch = x.size(1)
        indices = np.random.permutation(batch)
        if self.mixup_ratio == 1:
            if len(indices) % 2 != 0:
                indices = np.append(indices, (indices[-1]))
            idx1 = indices[0::2]
            idx2 = indices[1::2]

        else:
            mix_size = int(max(2, batch * self.mixup_ratio // 2 * 2))
            mix_indices = indices[: mix_size]
            idx1 = np.append(mix_indices[0::2], (indices[mix_size:]))
            idx2 = np.append(mix_indices[1::2], (indices[mix_size:]))

        idx1 = torch.from_numpy(idx1).to(x.device)
        idx2 = torch.from_numpy(idx2).to(x.device)

        x1 = x[:, idx1]
        x2 = x[:, idx2]

        coef = self.beta.sample().to(x.device).type_as(x)
        x = (coef * x1 + (1 - coef) * x2)

        pad1 = encoder_padding_mask[idx1]
        pad2 = encoder_padding_mask[idx2]
        encoder_padding_mask = pad1 + pad2
        input_lengths = (~encoder_padding_mask).sum(-1)

        mixup = {
            "coef": coef,
            "index1": idx1,
            "index2": idx2,
        }
        return x, encoder_padding_mask, input_lengths, mixup

    def forward(self, src_tokens, src_lengths):

        batch = src_tokens.size(0)

        x = src_tokens.transpose(0, 1)
        input_lengths = src_lengths

        # padding to the multiply of 2
        max_len = x.size(0)
        length = reduce(lambda a, b: max(1, a) * max(1, b), self.pds_ratios)
        padding_to_len = (length - max_len % length)
        if length > 1 and padding_to_len > 0:
            padding_for_pds = x.new_zeros((padding_to_len, batch, x.size(2)))
            x = torch.cat([x, padding_for_pds], dim=0)

        encoder_padding_mask = lengths_to_padding_mask_with_maxlen(input_lengths, x.size(0))

        # gather cosine similarity
        cos_sim_idx = -1
        dis = self.dis
        if self.gather_cos_sim:
            self.add_to_dict(x, dis, cos_sim_idx)

        layer_idx = 0
        ctc_logit = None
        mixup = None
        prev_state = []
        prev_padding = []
        intermedia_ctc_logits = []
        for i in range(self.pds_stages):
            downsampling = getattr(self, f"downsampling{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            stage = getattr(self, f"stage{i + 1}")
            ctc = getattr(self, f"ctc{i + 1}")
            adapter = getattr(self, f"adapter{i + 1}")

            if self.training and self.mixup and layer_idx == self.mixup_layer:
                x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(x, encoder_padding_mask)

            x, input_lengths = downsampling(x, input_lengths)
            encoder_padding_mask = lengths_to_padding_mask_with_maxlen(input_lengths, x.size(0))

            # gather cosine similarity
            cos_sim_idx += 10
            cos_sim_idx = cos_sim_idx // 10 * 10 - 1
            if self.gather_cos_sim:
                cos_sim_idx += 1
                self.add_to_dict(x, dis, cos_sim_idx)

            # add the position encoding and dropout
            if pos_embed:
                if self.attn_type in ["rel_pos", "rel_pos_legacy", "rel_selfattn", "reduced_rel_pos"]:
                    positions = pos_embed(x)

                elif self.attn_type == "rope":
                    positions = None

                else:
                    positions = pos_embed(encoder_padding_mask).transpose(0, 1)
                    x += positions
                    positions = None
            else:
                positions = None

            if i == 0:
                x = self.dropout(x)
            else:
                x = self.pds_dropout(x)

            for layer in stage:
                x = layer(x, encoder_padding_mask, pos_emb=positions)
                layer_idx += 1

                if layer.conv_stride > 1:

                    # Stride Mask (B, 1, T // S, T // S)
                    if encoder_padding_mask is not None:
                        encoder_padding_mask = encoder_padding_mask[:, ::layer.conv_stride]

                    # Update Seq Lengths
                    if input_lengths is not None:
                        input_lengths = torch.div(input_lengths - 1, layer.conv_stride, rounding_mode='floor') + 1

                # gather cosine similarity
                if self.gather_cos_sim:
                    cos_sim_idx += 1
                    self.add_to_dict(x, dis, cos_sim_idx)

                if self.training and self.mixup and layer_idx == self.mixup_layer:
                    if torch.rand(1) < self.mixup_prob:
                        x, encoder_padding_mask, input_lengths, mixup = self.apply_mixup(x, encoder_padding_mask)

                if self.use_ctc and self.inter_ctc and self.ctc_layer == layer_idx:
                    ctc_logit = self.ctc(x.clone())

            prev_state.append(x)
            prev_padding.append(encoder_padding_mask)

            # interleave CTC
            if ctc is not None:
                logit = ctc(x.clone())
                intermedia_ctc_logits.append([logit, encoder_padding_mask])

                prob = utils.softmax(logit / self.intermedia_temperature, dim=-1)
                x, encoder_padding_mask = adapter([x, prob], encoder_padding_mask)

        if self.fusion_stages_num != 0:
            fusion_state = []
            i = -1
            seq_len = x.size(0)
            for state in prev_state:
                i += 1
                fusion_downsampling = getattr(self, f"fusion_downsampling{i + 1}")
                fusion_pre_layer_norm = getattr(self, f"fusion_pre_layer_norm{i + 1}")
                fusion_post_layer_norm = getattr(self, f"fusion_post_layer_norm{i + 1}")

                if fusion_pre_layer_norm is not None or fusion_pre_layer_norm is not None:
                    state = fusion_pre_layer_norm(state)

                    if self.pds_fusion_transform == "conv":
                        state = fusion_downsampling(state)
                    elif self.pds_fusion_transform == "conv2":
                        state = fusion_downsampling(state, prev_padding[i])
                    elif self.pds_fusion_transform == "pool":
                        state = state.permute(1, 2, 0)  # bsz, dim, seq_len
                        if i != self.pds_stages - 1:
                            state = nn.functional.adaptive_max_pool1d(state, seq_len)
                        state = fusion_downsampling(state)

                    state = fusion_post_layer_norm(state)
                    fusion_state.append(state)
            x = (torch.stack(fusion_state, dim=0) * self.fusion_weight.view(-1, 1, 1, 1)).sum(0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.use_ctc and ctc_logit is None:
            ctc_logit = self.ctc(x)

        return {
            "encoder_out": [x],  # T x B x C
            "ctc_logit": [] if ctc_logit is None else [ctc_logit],  # T x B x C
            "intermedia_ctc_logits": intermedia_ctc_logits,  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "mixup": mixup,
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )
        new_ctc_logit = (
            [] if len(encoder_out["ctc_logit"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["ctc_logit"] if x is not None]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "ctc_logit": new_ctc_logit,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


@register_model_architecture(model_name="pdss2t_transformer", arch_name="pdss2t_transformer")
def base_architecture(args):
    # Convolutional subsampler
    args.subsampling_type = getattr(args, "subsampling_type", "conv1d")
    args.subsampling_layers = getattr(args, "subsampling_layers", 2)
    args.subsampling_filter = getattr(args, "subsampling_filter", 1024)
    args.subsampling_kernel = getattr(args, "subsampling_kernel", 5)
    args.subsampling_stride = getattr(args, "subsampling_stride", 2)
    args.subsampling_norm = getattr(args, "subsampling_norm", "none")
    args.subsampling_activation = getattr(args, "subsampling_activation", "glu")

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_type = getattr(args, "encoder_attention_type", "selfattn")
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_type = getattr(args, "decoder_attention_type", "selfattn")
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.max_encoder_relative_length = getattr(args, 'max_encoder_relative_length', -1)
    args.max_decoder_relative_length = getattr(args, 'max_decoder_relative_length', -1)
    args.k_only = getattr(args, 'k_only', True)

    # Conformer
    args.macaron_style = getattr(args, "macaron_style", False)
    args.use_cnn_module = getattr(args, "use_cnn_module", False)
    args.cnn_module_kernel = getattr(args, "cnn_module_kernel", 31)

    # PDS
    args.pds_stages = getattr(args, "pds_stages", None)
    args.pds_layers = getattr(args, "pds_layers", None)
    args.pds_ratios = getattr(args, "pds_ratios", None)

    args.pds_ds_method = getattr(args, "pds_ds_method", "conv")
    args.pds_embed_dims = getattr(args, "pds_embed_dims", None)
    args.pds_embed_norm = getattr(args, "pds_embed_norm", False)
    args.pds_position_embed = getattr(args, "pds_position_embed", None)

    args.pds_attn_heads = getattr(args, "pds_attn_heads", None)
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", None)
    args.pds_cnn_kernel_sizes = getattr(args, "pds_cnn_kernel_sizes", None)

    args.pds_attn_ds_ratios = getattr(args, "pds_attn_ds_ratios", None)
    args.pds_conv_strides = getattr(args, "pds_conv_strides", None)
    args.pds_attn_strides = getattr(args, "pds_attn_strides", None)

    args.ctc_layer = getattr(args, "ctc_layer", 0)
    args.pds_dropout = getattr(args, "pds_dropout", args.dropout)

    args.pds_fusion = getattr(args, "pds_fusion", False)
    args.pds_fusion_method = getattr(args, "pds_fusion_method", "all_conv")

    # intermedia CTC
    args.pds_ctc = getattr(args, "pds_ctc", None)
    args.intermedia_adapter = getattr(args, "intermedia_adapter", "none")
    args.intermedia_drop_prob = getattr(args, "intermedia_drop_prob", 0)

    # mixup
    args.inter_mixup = getattr(args, "inter_mixup", False)
    args.inter_mixup_layer = getattr(args, "inter_mixup_layer", None)
    args.inter_mixup_beta = getattr(args, "inter_mixup_beta", 0.5)
    args.inter_mixup_prob = getattr(args, "inter_mixup_prob", 0.5)


def set_pds_base_8(args):
    args.pds_stages = getattr(args, "pds_stages", 4)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_1_2")
    args.pds_layers = getattr(args, "pds_layers", "3_3_3_3")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1")


def set_pds_base_16(args):
    args.pds_stages = getattr(args, "pds_stages", 4)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_2_2")
    args.pds_layers = getattr(args, "pds_layers", "2_2_6_2")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1")


def set_pds_base_32(args):
    args.pds_stages = getattr(args, "pds_stages", 5)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_2_2_2")
    args.pds_layers = getattr(args, "pds_layers", "2_2_3_3_2")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1_1")


def set_pds_deep_8(args):
    args.pds_stages = getattr(args, "pds_stages", 4)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_1_2")
    args.pds_layers = getattr(args, "pds_layers", "7_7_7_9")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1")


def set_pds_deep_16(args):
    args.pds_stages = getattr(args, "pds_stages", 4)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_2_2")
    args.pds_layers = getattr(args, "pds_layers", "5_5_12_8")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1")


def set_pds_deep_32(args):
    args.pds_stages = getattr(args, "pds_stages", 5)
    args.pds_ratios = getattr(args, "pds_ratios", "2_2_2_2_2")
    args.pds_layers = getattr(args, "pds_layers", "5_5_7_7_6")
    args.pds_kernel_sizes = getattr(args, "pds_kernel_sizes", "5_5_5_5_5")
    args.pds_position_embed = getattr(args, "pds_position_embed", "1_1_1_1_1")


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_s")
def pdss2t_transformer_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_base_16(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_s_8")
def pdss2t_transformer_s_8(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_base_8(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_s_16")
def pdss2t_transformer_s_16(args):
    pdss2t_transformer_s(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_s_32")
def pdss2t_transformer_s_32(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_base_32(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_sd")
def pdss2t_transformer_sd(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_deep_16(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_sd_8")
def pdss2t_transformer_sd_8(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_deep_8(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_sd_16")
def pdss2t_transformer_sd_16(args):
    pdss2t_transformer_sd(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_sd_32")
def pdss2t_transformer_sd_32(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    # PDS
    set_pds_deep_32(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "256_256_256_256")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "4_4_4_4")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "8_8_8_8")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_m")
def pdss2t_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)

    # PDS
    set_pds_base_16(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "512_512_512_512")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "8_8_8_8")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "4_4_4_4")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_m_8")
def pdss2t_transformer_m_8(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)

    # PDS
    set_pds_base_8(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "512_512_512_512")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "8_8_8_8")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "4_4_4_4")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_m_16")
def pdss2t_transformer_m_16(args):
    pdss2t_transformer_m(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_m_32")
def pdss2t_transformer_m_32(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)

    # PDS
    set_pds_base_32(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "512_512_512_512")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "8_8_8_8")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "4_4_4_4")

    base_architecture(args)


@register_model_architecture("pdss2t_transformer", "pdss2t_transformer_l")
def pdss2t_transformer_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)

    # PDS
    set_pds_base_16(args)
    args.pds_embed_dims = getattr(args, "pds_embed_dims", "1024_1024_1024_1024")
    args.pds_attn_heads = getattr(args, "pds_attn_heads", "16_16_16_16")
    args.pds_ffn_ratios = getattr(args, "pds_ffn_ratios", "4_4_4_4")

    base_architecture(args)
