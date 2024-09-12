def self_attn(d, n, aatn=False):
    q_len = n
    k_len = n

    if aatn:
        qkv_proj = q_len * d * d  # 只需要算一个映射
        attn_weight = 0  # 无需计算注意力权重
        softmax = 0  # 无需计算softmax
        attn = k_len * d  # attn=k.mean() #应该是累加，再除
    else:
        qkv_proj = q_len * d * d + 2 * k_len * d * d  # q,k,v的映射。
        attn_weight = q_len * k_len * d  # a=q x k' = (q_len,d) x (d,k_len)
        softmax = q_len * k_len  # softmax(a)
        attn = q_len * k_len * d  # a x v' = (q_len,k_len) x (k_len,d)

    attn_proj = q_len * d * d
    return qkv_proj + attn_weight + softmax + attn + attn_proj

def cross_attn(d, q_len, k_len):
    qkv_proj = q_len * d * d + 2 * k_len * d * d  # q,k,v的映射
    attn_weight = q_len * k_len * d  # a=q x k' = (q_len,d) x (d,k_len)
    softmax = q_len * k_len  # softmax(a)
    attn = q_len * k_len * d  # a x v' = (q_len,k_len) x (k_len,d)

    attn_proj = q_len * d * d
    return qkv_proj + attn_weight + softmax + attn + attn_proj


def encoder(encoder_layer=12, d=256, d_ffn=256 * 8, n=100):
    attn = self_attn(d, n, False)
    ffn = 2 * n * d * d_ffn
    return encoder_layer * (attn + ffn)


def decoder(decoder_layer=6, d=256, d_ffn=256 * 8, audio_len=100, text_len=10, aatn=False):
    selfAttn = self_attn(d, text_len, aatn)
    crossAttn = cross_attn(d, text_len, audio_len)
    ffn = 2 * text_len * d * d_ffn
    return decoder_layer * (selfAttn + crossAttn + ffn)


def output_layer(d, vocab_size, split_num=1):
    if split_num == 1:
        return 1 * d * vocab_size
    return 1 * d * vocab_size / split_num


def flops(encoder_layer=12, decoder_layer=6, d=256, d_ffn=256 * 8, vocab_size=10000, split_num=1, audio_len=100,
          seq_len=10, aatn=False):
    flops_encoder = encoder(encoder_layer, d, d_ffn, audio_len)
    flops_decoder=decoder(decoder_layer, d, d_ffn, audio_len, seq_len, aatn)
    flops_output=output_layer(d, vocab_size, split_num)*seq_len

    print("enc={:.0f}\tdec={:.0f}\toutput={:.0f}".format(flops_encoder / 1000, flops_decoder / 1000, flops_output / 1000))
    print("all={:.0f}".format((flops_encoder + flops_decoder + flops_output) / 1000))
    return (flops_encoder + flops_decoder + flops_output) / 1000


if __name__ == '__main__':
    audio_len = 178
    text_len = 25
    print("encoder\tdecoder\toutput")

    print("-------base--------")
    flops(6, 6, 256, 256 * 8, 10000, 1, audio_len, text_len, False )

    print("-------base_e16d3--------")
    flops(16, 3, 256, 256 * 8, 10000, 1, audio_len, text_len, False)

    print("-------small--------")
    flops(12, 6, 144, 144 * 8, 10000, 1, audio_len, text_len, False)

    print("-------small_aatn--------")
    flops(12, 6, 144, 144 * 8, 10000, 1, audio_len, text_len, True)

    print("-------small_e16d3--------")
    flops(16, 3, 144, 144 * 8, 10000, 1, audio_len, text_len, False)

    print("-------small_e16d3_aatn--------")
    flops(16, 3, 144, 144 * 8, 10000, 1, audio_len, text_len, True)

