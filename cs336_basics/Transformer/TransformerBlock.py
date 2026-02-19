import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn
import math
from cs336_basics.Transformer.PreNormTransformerBlock import RMSNorm, RoPE, SwiGLU, MultiHeadSelfAttention, MultiHeadSelfAttentionWithRoPE


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Extract attention weights
        self.weights_attn = {
            "q_proj": weights["attn.q_proj.weight"],
            "k_proj": weights["attn.k_proj.weight"],
            "v_proj": weights["attn.v_proj.weight"],
            "o_proj": weights["attn.output_proj.weight"],
        }

        # RMSNorm
        self.ln1 = RMSNorm(self.d_model, weights["ln1.weight"], eps=1e-5)
        self.ln2 = RMSNorm(self.d_model, weights["ln2.weight"], eps=1e-5)

        # FFN (SwiGLU)
        self.ffn = SwiGLU(
            self.d_model, self.d_ff,
            weights["ffn.w1.weight"],
            weights["ffn.w2.weight"],
            weights["ffn.w3.weight"],
        )

        # Multi-head self-attention with RoPE — 在 __init__ 中创建一次
        self.attn = MultiHeadSelfAttentionWithRoPE(
            self.d_model, self.num_heads, self.max_seq_len, self.theta
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm block 1: x = x + Attention(LN1(x))
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        x_normed = self.ln1(x)
        attn_out = self.attn(
            x_normed,
            self.weights_attn["q_proj"],
            self.weights_attn["k_proj"],
            self.weights_attn["v_proj"],
            self.weights_attn["o_proj"],
            token_positions=token_positions,
        )
        x = x + attn_out

        # Pre-norm block 2: x = x + FFN(LN2(x))
        x_normed2 = self.ln2(x)
        ffn_out = self.ffn(x_normed2)
        x = x + ffn_out

        return x