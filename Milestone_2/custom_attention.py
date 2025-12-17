import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class TemperatureMultiheadAttention(MultiheadAttention):
    # inherit from normal MultiheadAttention component
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialise new learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(self.num_heads))

    # define custom forward 
    def forward(self, x, *args, **kwargs):

        # default behaviour
        B, N, C = x.shape
        head_dim = C // self.num_heads

        attn_mask = kwargs.get("attn_mask", None)

        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores /= math.sqrt(head_dim)

        # reshape temperature
        temp = self.temperature.view(1, -1, 1, 1)

        # apply temperature to attn_scors / softmax function
        attn_scores = attn_scores / temp

        if attn_mask is not None:
            attn_scores += attn_mask

        attn = F.softmax(attn_scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)

        return x, attn
