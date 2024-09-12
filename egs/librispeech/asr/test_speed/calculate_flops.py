"""Computes the flops needed for training/running transformer networks."""

import collections


# random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
# DROPOUT_FLOPS = 4
DROPOUT_FLOPS = 0 #infer =0
# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5


class TransformerHparams(object):
  """Computes the train/inference FLOPs for transformers."""

  def __init__(self, h, l, s=512, v=30522, e=None, i=None, heads=None,
      head_size=None, output_frac=0.15625, sparse_embed_lookup=False,
      decoder=False):
    self.h = h  # hidden size
    self.l = l  # number of layers
    self.s = s  # sequence length
    self.v = v  # vocab size
    self.e = h if e is None else e  # embedding size
    self.i = h * 4 if i is None else i  # intermediate size
    self.kqv = h if head_size is None else head_size * heads  # attn proj sizes ,256
    self.heads = max(h // 64, 1) if heads is None else heads  # attention heads ,4
    self.output_frac = output_frac  # percent of tokens using an output softmax
    self.sparse_embed_lookup = sparse_embed_lookup  # sparse embedding lookups
    self.decoder = decoder  # decoder has extra attn to encoder states


  def get_block_flops(self):
    """Get the forward-pass FLOPs for a single transformer block."""
    attn_mul = 2 if self.decoder else 1
    block_flops = dict(
        kqv=3 * 2 * self.h * self.kqv * attn_mul, #qkv的映射
        kqv_bias=3 * self.kqv * attn_mul, # qkv映射后的 bias
        attention_scores=2 * self.kqv * self.s * attn_mul,
        attn_softmax=SOFTMAX_FLOPS * self.s * self.heads * attn_mul,
        #attention_dropout=DROPOUT_FLOPS * self.s * self.heads * attn_mul,
        attention_scale=self.s * self.heads * attn_mul,
        attention_weighted_avg_values=2 * self.h * self.s * attn_mul,
        attn_output=2 * self.h * self.h * attn_mul,
        attn_output_bias=self.h * attn_mul,
        attn_output_dropout=DROPOUT_FLOPS * self.h * attn_mul,
        attn_output_residual=self.h * attn_mul,
        attn_output_layer_norm=LAYER_NORM_FLOPS * attn_mul,
        intermediate=2 * self.h * self.i,
        intermediate_act=ACTIVATION_FLOPS * self.i,
        intermediate_bias=self.i,
        output=2 * self.h * self.i,
        output_bias=self.h,
        output_dropout=DROPOUT_FLOPS * self.h,
        output_residual=self.h,
        output_layer_norm=LAYER_NORM_FLOPS * self.h,
    )
    return sum(block_flops.values()) * self.s

  def get_block_flops_AAT(self):
      """Get the forward-pass FLOPs for a single transformer block."""
      attn_mul = 2 if self.decoder else 1
      block_flops = dict(
          kqv=3 * 2 * self.h * self.kqv * 1,
          kqv_bias=3 * self.kqv * 1,
          attention_scores=2 * self.kqv * self.s * 1,
          attn_softmax=SOFTMAX_FLOPS * self.s * self.heads * 1,
          attention_dropout=DROPOUT_FLOPS * self.s * self.heads * 1,
          attention_scale=self.s * self.heads * attn_mul,
          attention_weighted_avg_values=2 * self.h * self.s * attn_mul,
          attn_output=2 * self.h * self.h * attn_mul,
          attn_output_bias=self.h * attn_mul,
          attn_output_dropout=DROPOUT_FLOPS * self.h * attn_mul,
          attn_output_residual=self.h * attn_mul,
          attn_output_layer_norm=LAYER_NORM_FLOPS * attn_mul,
          intermediate=2 * self.h * self.i,
          intermediate_act=ACTIVATION_FLOPS * self.i,
          intermediate_bias=self.i,
          output=2 * self.h * self.i,
          output_bias=self.h,
          output_dropout=DROPOUT_FLOPS * self.h,
          output_residual=self.h,
          output_layer_norm=LAYER_NORM_FLOPS * self.h,
      )
      return sum(block_flops.values()) * self.s

  def get_embedding_flops(self, output=False):
    """Get the forward-pass FLOPs the transformer inputs or output softmax."""
    embedding_flops = {}
    if output or (not self.sparse_embed_lookup):
      embedding_flops["main_multiply"] = 2 * self.e * self.v
    # input embedding post-processing
    # if not output:
    #   embedding_flops.update(dict(
    #       tok_type_and_position=2 * self.e * (self.s + 2),
    #       add_tok_type_and_position=2 * self.e,
    #       emb_layer_norm=LAYER_NORM_FLOPS * self.e,
    #       emb_dropout=DROPOUT_FLOPS * self.e
    #   ))
    # projection layer if e != h
    if self.e != self.h or output:
      # embedding_flops.update(dict(
      #     hidden_kernel=2 * self.h * self.e,
      #     hidden_bias=self.e if output else self.h
      # ))
      # extra hidden layer and output softmax
      if output:
        embedding_flops.update(dict(
            # hidden_activation=ACTIVATION_FLOPS * self.e,
            # hidden_layernorm=LAYER_NORM_FLOPS * self.e,
            output_softmax=SOFTMAX_FLOPS * self.v,
            output_target_word=2 * self.v
        ))
        return self.output_frac * sum(embedding_flops.values()) * self.s
    return sum(embedding_flops.values()) * self.s

  def get_binary_classification_flops(self):
    classification_flops = dict(
        hidden=2 * self.h * self.h,
        hidden_bias=self.h,
        hidden_act=ACTIVATION_FLOPS * self.h,
        logits=2 * self.h
    )
    return sum(classification_flops.values()) * self.s

  def get_train_flops(self, batch_size, train_steps, discriminator=False):
    """Get the FLOPs for pre-training the transformer."""
    # 2* for forward/backward pass
    return 2 * batch_size * train_steps * (
        (self.l * self.get_block_flops()) +
        self.get_embedding_flops(output=False) +
        (self.get_binary_classification_flops() if discriminator else
         self.get_embedding_flops(output=True))
    )

  def get_infer_flops(self):
    """Get the FLOPs for running inference with the transformer on a
    classification task."""
    return ((self.l * self.get_block_flops()))

  def get_infer_flops_aat(self):
    """Get the FLOPs for running inference with the transformer on a
    classification task."""
    return ((self.l * self.get_block_flops_AAT()))



MODEL_FLOPS = collections.OrderedDict([
    # These runtimes were computed with tensorflow FLOPs counting instead of the
    # script, as the neural architectures are quite different.
    # 768648884 words in LM1b benchmark, 10 epochs with batch size 20,
    # seq length 128, 568093262680 FLOPs per example.
    ("elmo", 2 * 10 * 768648884 * 568093262680 / (20.0 * 128)),
    # 15064773691518 is FLOPs for forward pass on 32 examples.
    # Therefore 2 * steps * batch_size * 15064773691518 / 32 is XLNet compute
    ("xlnet", 2 * 500000 * 8192 * 15064773691518 / 32.0),

    # Runtimes computed with the script
    ("transformer_base", TransformerHparams(256, 6,s=1, v=10000, i=2048,heads=4,head_size=256//4,decoder=True,output_frac=1.0).get_infer_flops()),
    ("transformer_small", TransformerHparams(144, 6,s=1,v=10000, i=1152,heads=4,head_size=144//4,decoder=True,output_frac=1.0).get_infer_flops()),
    ("transformer_small_AAT", TransformerHparams(144, 6,s=1,v=10000, i=1152,heads=4,head_size=144//4,decoder=True,output_frac=1.0).get_infer_flops_aat()),
    ("transformer_small_embedding", TransformerHparams(144, 6,s=1,v=10000, i=1152,heads=4,head_size=144//4,decoder=True,output_frac=1.0).get_embedding_flops(output=True))

])


def main():
  for k, v in MODEL_FLOPS.items():
    print(k, v)


if __name__ == "__main__":
  main()