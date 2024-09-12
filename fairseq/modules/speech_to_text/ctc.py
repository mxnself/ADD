import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
)

logger = logging.getLogger(__name__)


class CTC(nn.Module):
    
    def __init__(self, embed_dim, dictionary_size, dropout, need_layernorm=False):
        super(CTC, self).__init__()

        self.embed_dim = embed_dim
        self.ctc_projection = nn.Linear(embed_dim, dictionary_size, bias=False)

        nn.init.normal_(
            self.ctc_projection.weight, mean=0, std=embed_dim ** -0.5
        )

        self.ctc_dropout_module = FairseqDropout(
            p=dropout, module_name=self.__class__.__name__
        )
        self.need_layernorm = need_layernorm
        if self.need_layernorm:
            self.LayerNorm = LayerNorm(embed_dim)

    def forward(self, x):
        if self.need_layernorm:
            x = self.LayerNorm(x)

        x = self.ctc_projection(self.ctc_dropout_module(x))
        return x

    def softmax(self, x, temperature=1.0):
        return F.softmax(self.ctc_projection(x) / temperature, dim=-1, dtype=torch.float32)

    def log_softmax(self, x, temperature=1.0):
        return F.log_softmax(self.ctc_projection(x) / temperature, dim=-1, dtype=torch.float32)

    def argmax(self, x):
        return torch.argmax(self.ctc_projection(x), dim=-1)

