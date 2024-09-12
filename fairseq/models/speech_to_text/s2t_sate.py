import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text import (
    S2TTransformerModel,
    S2TTransformerEncoder,
    PDSS2TTransformerModel,
    PDSS2TTransformerEncoder,
)
from fairseq.modules.speech_to_text import Adapter, CTC
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    DynamicLinearCombination
)

logger = logging.getLogger(__name__)


@register_model("s2t_sate")
class S2TSATEModel(S2TTransformerModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

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
        # target CTC
        parser.add_argument(
            "--target-ctc-layer",
            default=None,
            type=str,
            help="ctc layer for target sentence",
        )
        parser.add_argument(
            "--target-intermedia-ctc-layers",
            default=None,
            type=str,
            help="intermedia ctc layers for target sentence",
        )
        # freeze
        parser.add_argument(
            "--freeze-acoustic-encoder",
            action="store_true",
            help="freeze the parameters of the acoustic encoder",
        )
        parser.add_argument(
            "--freeze-textual-encoder",
            action="store_true",
            help="freeze the parameters of the acoustic encoder",
        )
        parser.add_argument(
            "--freeze-decoder",
            action="store_true",
            help="freeze the parameters of the decoder",
        )
        pass

    @classmethod
    def build_encoder(cls, args, task=None, decoder_embed_tokens=None):
        encoder = S2TSATEEncoder(args, task, decoder_embed_tokens)

        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(
                f"loaded pretrained acoustic encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from, strict=False
            )

        if getattr(args, "load_pretrained_acoustic_encoder_from", None):
            logger.info(
                f"loaded pretrained acoustic encoder from: "
                f"{args.load_pretrained_acoustic_encoder_from}"
            )
            encoder.acoustic_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.acoustic_encoder, checkpoint=args.load_pretrained_acoustic_encoder_from, strict=False
            )

        if getattr(args, "load_pretrained_text_encoder_from", None):
            logger.info(
                f"loaded pretrained text encoder from: "
                f"{args.load_pretrained_text_encoder_from}"
            )
            encoder.text_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.text_encoder, checkpoint=args.load_pretrained_text_encoder_from, strict=False
            )

        if args.share_ctc_and_adapter and hasattr(encoder.adapter, "embed_adapter"):
            encoder.acoustic_encoder.ctc.ctc_projection.weight = encoder.adapter.embed_adapter.weight

        return encoder

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class TextEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens=None):

        super().__init__(None)

        self.register_buffer("version", torch.Tensor([3]))  # for consistent
        embed_dim = args.encoder_embed_dim
        layer_num = args.text_encoder_layers
        self.layer_num = layer_num
        self.embed_tokens = embed_tokens

        self.embed_scale = math.sqrt(embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = dictionary.pad_index

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(layer_num)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        # CTC
        self.use_ctc = getattr(args, "target_ctc_weight", 0) > 0
        if self.use_ctc:
            self.ctc_layer = getattr(args, "target_ctc_layer", layer_num)
            self.inter_ctc = True if self.ctc_layer != args.encoder_layers else False
            if self.inter_ctc:
                logger.info("Target CTC loss in layer %d" % self.ctc_layer)
            self.ctc = CTC(embed_dim,
                           dictionary_size=embed_tokens.num_embeddings,
                           dropout=args.dropout,
                           need_layernorm=True if self.inter_ctc else False)

            self.ctc.ctc_projection.weight = embed_tokens.weight

        self.intermedia_ctc_layers = []
        self.target_intermedia_ctc_layers = getattr(args, "target_intermedia_ctc_layers", None)
        if self.target_intermedia_ctc_layers is not None:
            target_intermedia_ctc_layers = self.target_intermedia_ctc_layers.split(",")
            for layer_idx in target_intermedia_ctc_layers:
                layer_idx = int(layer_idx)
                assert layer_idx <= layer_num, (layer_idx, layer_num)

                if layer_idx <= 0:
                    layer_idx += layer_num
                self.intermedia_ctc_layers.append(layer_idx)

                logger.info("Intermedia target CTC loss in layer %d" % layer_idx)

                self.ctc = CTC(embed_dim,
                               dictionary_size=len(dictionary),
                               dropout=args.dropout)

                if embed_tokens is not None:
                    self.ctc.ctc_projection.weight = embed_tokens.weight

            strategy = None
            if args.intermedia_adapter == "shrink":
                strategy = getattr(args, "ctc_compress_strategy", None)
            elif args.intermedia_adapter == "league":
                strategy = getattr(args, "intermedia_distribution_cutoff", None)
            self.adapter = Adapter(embed_dim, args.intermedia_adapter,
                                   len(dictionary),
                                   # embed_tokens=embed_tokens,
                                   strategy=strategy)
            self.intermedia_drop_prob = getattr(args, "intermedia_drop_prob", 0)
            self.intermedia_temperature = getattr(args, "intermedia_temperature", 1)

    def forward(self, x, encoder_padding_mask=None, history=None):

        x = self.embed_scale * x
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x = positions + x
        x = self.dropout_module(x)

        target_ctc_logit = None
        target_intermedia_ctc_logits = []
        layer_idx = 0
        for layer in self.layers:
            if history is not None:
                x = history.pop()
            x = layer(x, encoder_padding_mask, pos_emb=positions)
            layer_idx += 1

            if self.use_ctc and self.inter_ctc and self.ctc_layer == layer_idx:
                target_ctc_logit = self.ctc(x.clone())

            if layer_idx != self.layer_num and layer_idx in self.intermedia_ctc_layers:
                if self.intermedia_drop_prob > 0:
                    p = torch.rand(1).uniform_()
                    if p < self.intermedia_drop_prob:
                        break

                norm_x = self.layer_norm(x)
                logit = self.ctc(norm_x)
                target_intermedia_ctc_logits.append(logit)

                prob = utils.softmax(logit / self.intermedia_temperature, dim=-1)
                x, encoder_padding_mask = self.adapter([x, prob], encoder_padding_mask)

            if history is not None:
                history.push(x)

        if history is not None:
            x = history.pop()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.use_ctc and target_ctc_logit is None:
            target_ctc_logit = self.ctc(x)

        return x, target_ctc_logit, target_intermedia_ctc_logits


class S2TSATEEncoder(FairseqEncoder):
    """Speech-to-text Conformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task=None, decoder_embed_tokens=None):
        super().__init__(None)

        # acoustic encoder
        acoustic_encoder_type = args.acoustic_encoder
        if acoustic_encoder_type == "transformer":
            self.acoustic_encoder = S2TTransformerEncoder(args, task)
        elif acoustic_encoder_type == "pds":
            self.acoustic_encoder = PDSS2TTransformerEncoder(args, task)
        else:
            logging.error("Unsupported model arch {}!".format(acoustic_encoder_type))

        # adapter
        self.temperature = args.temperature

        strategy = None
        if args.adapter == "shrink":
            strategy = getattr(args, "ctc_compress_strategy", "avg")
        elif args.adapter == "league":
            strategy = getattr(args, "intermedia_distribution_cutoff", None)

        self.adapter = Adapter(args.encoder_embed_dim,
                               args.adapter,
                               len(task.source_dictionary),
                               decoder_embed_tokens if task.source_dictionary == task.target_dictionary else None,
                               strategy=strategy)

        if args.share_ctc_and_adapter and hasattr(self.adapter, "embed_adapter"):
            self.acoustic_encoder.ctc.ctc_projection.weight = self.adapter.embed_adapter.weight

        acoustic_encoder_attention_type = args.encoder_attention_type
        args.encoder_attention_type = args.text_attention_type

        # text encoder
        self.text_encoder = TextEncoder(args, task.source_dictionary, decoder_embed_tokens)

        args.encoder_attention_type = acoustic_encoder_attention_type

        self.freeze_acoustic_encoder = getattr(args, "freeze_acoustic_encoder", False)
        self.freeze_textual_encoder = getattr(args, "freeze_textual_encoder", False)

        if getattr(args, "use_enc_dlcl", False):
            layer_num = args.encoder_layers + args.text_encoder_layers + 2
            self.history = DynamicLinearCombination(args, is_encoder=True, layer_num=layer_num)
        else:
            self.history = None

    def forward(self, src_tokens, src_lengths):
        if self.history is not None:
            self.history.clean()

        if self.freeze_acoustic_encoder:
            with torch.no_grad():
                acoustic_encoder_out = self.acoustic_encoder(src_tokens, src_lengths)
        else:
            acoustic_encoder_out = self.acoustic_encoder(src_tokens, src_lengths)

        encoder_out = acoustic_encoder_out["encoder_out"][0]
        encoder_padding_mask = acoustic_encoder_out["encoder_padding_mask"][0]
        ctc_padding_mask = encoder_padding_mask

        if "ctc_logit" in acoustic_encoder_out and len(acoustic_encoder_out["ctc_logit"]) > 0:
            ctc_logit = acoustic_encoder_out["ctc_logit"][0]
            ctc_prob = F.softmax(ctc_logit / self.temperature, dim=-1, dtype=torch.float32)
        else:
            ctc_logit = None
            ctc_prob = None
        x = (encoder_out, ctc_prob)

        x, encoder_padding_mask = self.adapter(x, encoder_padding_mask)

        if self.history is not None:
            acoustic_history = self.acoustic_encoder.history
            layer_num = acoustic_history.layer_num
            idx = torch.arange(layer_num).unsqueeze(0).T.repeat(1, layer_num).to(x.device).unsqueeze(2)
            self.history.weight.scatter(0, idx, acoustic_history.weight)
            self.history.layers.extend(acoustic_history.layers)
            self.history.count = acoustic_history.count

            self.history.push(x)

        if self.freeze_textual_encoder:
            with torch.no_grad():
                x, target_ctc_logit, target_intermedia_ctc_logits = self.text_encoder(x, encoder_padding_mask, self.history)
        else:
            x, target_ctc_logit, target_intermedia_ctc_logits = self.text_encoder(x, encoder_padding_mask, self.history)

        return {
            "encoder_out": [x],  # T x B x C
            "ctc_logit": [ctc_logit],    # T x B x C
            "intermedia_ctc_logits": acoustic_encoder_out.get("intermedia_ctc_logits", []),  # B x T x C
            "target_ctc_logit": target_ctc_logit,  # B x T x C
            "target_intermedia_ctc_logits": target_intermedia_ctc_logits,  # B x T x C
            "ctc_padding_mask": [ctc_padding_mask], # B x T
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        new_ctc_logit = (
            [] if len(encoder_out["ctc_logit"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["ctc_logit"]]
        )

        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
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
            "ctc_logit": new_ctc_logit,
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


@register_model_architecture(model_name="s2t_sate", arch_name="s2t_sate")
def base_architecture(args):
    # Convolutional subsampler
    args.subsampling_type = getattr(args, "subsampling_type", "conv1d")
    args.subsampling_layers = getattr(args, "subsampling_layers", 2)
    args.subsampling_filter = getattr(args, "subsampling_filter", 1024)
    args.subsampling_kernel = getattr(args, "subsampling_kernel", 5)
    args.subsampling_stride = getattr(args, "subsampling_stride", 2)
    args.subsampling_norm = getattr(args, "subsampling_norm", "none")
    args.subsampling_activation = getattr(args, "subsampling_activation", "glu")

    # transformer
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


@register_model_architecture("s2t_sate", "s2t_sate_s")
def s2t_sate_s(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("s2t_sate", "s2t_sate_s_relative")
def s2t_sate_s_relative(args):
    args.encoder_attention_type = "relative"
    args.decoder_attention_type = "relative"
    args.max_encoder_relative_length = 100
    args.max_decoder_relative_length = 20
    args.k_only = True
    s2t_sate_s(args)


@register_model_architecture("s2t_sate", "s2t_sate_xs")
def s2t_sate_xs(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 3)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256 * 4)
    args.dropout = getattr(args, "dropout", 0.3)
    s2t_sate_s(args)


@register_model_architecture("s2t_sate", "s2t_sate_m")
def s2t_sate_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)


@register_model_architecture("s2t_sate", "s2t_sate_l")
def s2t_sate_l(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.2)
    base_architecture(args)
