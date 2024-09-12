from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text import (
    S2TTransformerModel,
    S2TTransformerEncoder,
    PDSS2TTransformerModel,
    PDSS2TTransformerEncoder,
    S2TSATEEncoder,
)
from fairseq.modules.speech_to_text import Adapter, CTC
from fairseq.models.transformer_s2 import (
    Embedding,
    TransformerS2Encoder,
    TransformerS2Decoder,
)
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    LegacyRelPositionalEncoding,
    RelPositionalEncoding,
    S2TTransformerEncoderLayer,
    DynamicLinearCombination,
    TransformerS2DecoderLayer,
    TransformerS2EncoderLayer,
)

logger = logging.getLogger(__name__)


@register_model("s2t_dual")
class S2TDualModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        PDSS2TTransformerModel.add_args(parser)
        # SATE setting
        parser.add_argument(
            "--text-encoder-layers",
            default=6,
            type=int,
            help="layers of the text encoder",
        )
        parser.add_argument(
            "--text-attention-type",
            default="selfattn",
            type=str,
            help="attention type of the textual encoder",
        )
        parser.add_argument(
            "--adapter",
            default="league",
            type=str,
            help="adapter type",
        )
        parser.add_argument(
            "--ctc-compress-strategy",
            default="avg",
            type=str,
            help="compress strategy, such as avg, weighted, and softmax",
        )
        parser.add_argument(
            "--share-ctc-and-adapter",
            default=False,
            action="store_true",
            help="share the projection weights of the ctc and adapter",
        )
        parser.add_argument(
            "--temperature",
            default=1.0,
            type=float,
            help="temperature of the CTC softmax",
        )
        parser.add_argument(
            "--acoustic-encoder",
            default="transformer",
            type=str,
            help="the architecture of the acoustic encoder",
        )
        parser.add_argument(
            "--target-ctc-layers",
            default=None,
            type=str,
            help="ctc layers for target sentence",
        )
        parser.add_argument(
            "--load-pretrained-acoustic-encoder-from",
            type=str,
            metavar="STR",
            help="model to take acoustic encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-text-encoder-from",
            type=str,
            metavar="STR",
            help="model to take text encoder weights from (for initialization)",
        )

        # multi-encoder
        parser.add_argument(
            "--asr-encoder",
            default="transformer",
            type=str,
            help="the architecture of the ASR encoder",
        )
        parser.add_argument(
            "--mt-encoder",
            default="transformer",
            type=str,
            help="the architecture of the MT encoder",
        )
        # parser.add_argument(
        #     "--mt-encoder-dim",
        #     default="transformer",
        #     type=str,
        #     help="the dimension of the MT encoder",
        # )
        parser.add_argument(
            "--mt-encoder-layers",
            default=6,
            type=int,
            help="the layers of the MT encoder",
        )
        parser.add_argument(
            "--encoder-asr-ratio",
            default=1,
            type=float,
            help="the ratio of the asr representation",
        )
        parser.add_argument(
            "--encoder-mt-ratio",
            default=1,
            type=float,
            help="the ratio of the mt representation",
        )
        parser.add_argument(
            "--encoder-drop-net",
            action="store_true",
            help="drop an input",
        )
        parser.add_argument(
            "--encoder-drop-net-prob",
            default=0.5,
            type=float,
            help="the probability of dropping",
        )
        parser.add_argument(
            "--encoder-drop-net-mix",
            action="store_true",
            help="mix the two input with any probability",
        )

        parser.add_argument(
            "--load-pretrained-asr-encoder-from",
            type=str,
            metavar="STR",
            help="model to take asr encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-mt-encoder-from",
            type=str,
            metavar="STR",
            help="model to take mt encoder weights from (for initialization)",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, embed_tokens=None):
        encoder = S2TDualEncoder(args, task, embed_tokens)

        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        if getattr(args, "load_pretrained_asr_encoder_from", None):
            encoder.asr_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.asr_encoder, checkpoint=args.load_pretrained_asr_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained asr encoder from: "
                f"{args.load_pretrained_asr_encoder_from}"
            )
        if getattr(args, "load_pretrained_mt_encoder_from", None):
            encoder.mt_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.mt_encoder, checkpoint=args.load_pretrained_mt_encoder_from, strict=False
            )
            logger.info(
                f"loaded pretrained mt encoder from: "
                f"{args.load_pretrained_mt_encoder_from}"
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = TransformerS2Decoder(args, task.target_dictionary, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None):
            logger.info(
                f"loaded pretrained decoder from: "
                f"{args.load_pretrained_decoder_from}"
            )
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from, strict=False
            )

        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim
            )

        setattr(args, "encoder_s1_ratio", args.encoder_asr_ratio)
        setattr(args, "encoder_s2_ratio", args.encoder_mt_ratio)

        encoder = cls.build_encoder(args, task, encoder_embed_tokens)
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logging.info("freeze the encoder module: {}".format(args.encoder_freeze_module))

        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        if getattr(args, "decoder_freeze_module", None):
            utils.freeze_parameters(decoder, args.decoder_freeze_module)
            logging.info("freeze the decoder module: {}".format(args.decoder_freeze_module))
        return cls(encoder, decoder)

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, speech_src_tokens, speech_src_lengths,
                text_src_tokens, text_src_lengths,
                prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(speech_src_tokens, speech_src_lengths,
                                   text_src_tokens, text_src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class S2TDualEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, embed_tokens=None):
        super().__init__(None)

        asr_encoder_type = args.asr_encoder
        if asr_encoder_type == "transformer":
            self.asr_encoder = S2TTransformerEncoder(args, task)
        elif asr_encoder_type == "pds":
            self.asr_encoder = PDSS2TTransformerEncoder(args, task)
        elif asr_encoder_type == "sate":
            self.asr_encoder = S2TSATEEncoder(args, task)
        else:
            logger.error("Unsupported ASR architecture: %s." % asr_encoder_type)

        attn_type = args.encoder_attention_type
        setattr(args, "encoder_layers", args.mt_encoder_layers)
        setattr(args, "encoder_attention_type", "selfattn")
        self.mt_encoder = TransformerS2Encoder(args, task.source_dictionary, embed_tokens)
        setattr(args, "encoder_attention_type", attn_type)

    def forward(self, speech_src_tokens, speech_src_lengths, text_src_tokens, text_src_lengths, **kwargs):
        asr_encoder_out = self.asr_encoder(speech_src_tokens, speech_src_lengths)
        encoder_representation = asr_encoder_out["encoder_out"][0]
        encoder_padding_mask = asr_encoder_out["encoder_padding_mask"][0]

        encoder_out = self.mt_encoder(text_src_tokens, text_src_lengths,
                                      encoder_representation, encoder_padding_mask)
        
        encoder_out["ctc_logit"] = asr_encoder_out["ctc_logit"]
        encoder_out["ctc_padding_mask"] = asr_encoder_out["encoder_padding_mask"]
        return encoder_out

    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        speech_src_tokens = net_input["src_tokens"]
        speech_src_lengths = net_input["src_lengths"]
        text_src_tokens = net_input["text_src_tokens"]
        text_src_lengths = net_input["text_src_lengths"]

        encoder_out = self.forward(speech_src_tokens, speech_src_lengths, text_src_tokens, text_src_lengths)
        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        """
         Reorder encoder output according to *new_order*.

         Args:
             encoder_out: output from the ``forward()`` method
             new_order (LongTensor): desired order

         Returns:
             *encoder_out* rearranged according to *new_order*
         """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_out_s2"]) == 0:
            new_encoder_out_s2 = []
        else:
            new_encoder_out_s2 = [encoder_out["encoder_out_s2"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask_s2"]) == 0:
            new_encoder_padding_mask_s2 = []
        else:
            new_encoder_padding_mask_s2 = [
                encoder_out["encoder_padding_mask_s2"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_out_s2": new_encoder_out_s2,   # T x B x C
            "encoder_padding_mask_s2": new_encoder_padding_mask_s2,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


@register_model_architecture(model_name="s2t_dual", arch_name="s2t_dual")
def base_architecture(args):
    # Convolutional subsampler
    args.subsampling_type = getattr(args, "subsampling_type", "conv1d")
    args.subsampling_layers = getattr(args, "subsampling_layers", 2)
    args.subsampling_filter = getattr(args, "subsampling_filter", 1024)
    args.subsampling_kernel = getattr(args, "subsampling_kernel", 5)
    args.subsampling_stride = getattr(args, "subsampling_stride", 2)
    args.subsampling_norm = getattr(args, "subsampling_norm", "none")
    args.subsampling_activation = getattr(args, "subsampling_activation", "glu")

    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
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
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
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

    # SATE
    args.acoustic_encoder = getattr(args, "acoustic_encoder", "transformer")
    args.adapter = getattr(args, "adapter", "league")
    args.ctc_compress_strategy = getattr(args, "ctc_compress_strategy", "avg")
    args.temperature = getattr(args, "temperature", 1.0)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.text_attention_type = getattr(args, "text_attention_type", "selfattn")
    args.share_ctc_and_adapter = getattr(args, "share_ctc_and_adapter", False)

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
    args.pds_ctc = getattr(args, "pds_ctc", "0_0_0_0")
    args.intermedia_adapter = getattr(args, "intermedia_adapter", "none")
    args.intermedia_drop_prob = getattr(args, "intermedia_drop_prob", 0)

    # dual
    args.encoder_asr_ratio = getattr(args, "encoder_asr_ratio", 1.0)
    args.encoder_mt_ratio = getattr(args, "encoder_mt_ratio", 1.0)
    args.encoder_drop_net = getattr(args, "encoder_drop_net", False)
    args.encoder_drop_net_prob = getattr(args, "encoder_drop_net_prob", 1.0)
    args.encoder_drop_net_mix = getattr(args, "encoder_drop_net_mix", False)


@register_model_architecture("s2t_dual", "s2t_dual_s")
def s2t_dual_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_dual", "s2t_dual_s_relative")
def s2t_dual_s_relative(args):
    args.max_encoder_relative_length = 100
    args.k_only = True
    s2t_dual_s(args)


@register_model_architecture("s2t_dual", "s2t_dual_xs")
def s2t_dual_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_dual_s(args)


@register_model_architecture("s2t_dual", "s2t_dual_sp")
def s2t_dual_sp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_dual_s(args)


@register_model_architecture("s2t_dual", "s2t_dual_m")
def s2t_dual_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_dual", "s2t_dual_mp")
def s2t_dual_mp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_dual_m(args)


@register_model_architecture("s2t_dual", "s2t_dual_l")
def s2t_dual_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)


@register_model_architecture("s2t_dual", "s2t_dual_lp")
def s2t_dual_lp(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    s2t_dual_l(args)
