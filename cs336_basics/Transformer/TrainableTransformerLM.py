"""
TrainableTransformerLM — 可从零训练的 Transformer 语言模型。

已有的 TransformerLM / TransformerBlock 需要传入预构建 weights dict，
无法从零训练。这个版本内部自己创建并初始化所有参数。
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

# 复用已有的 RoPE 和 softmax
from cs336_basics.Transformer.PreNormTransformerBlock import RoPE, softmax


# ── 初始化辅助 ────────────────────────────────────────────────────────────

def _trunc_normal_init(tensor: Tensor, fan_in: int, fan_out: int):
    """与 EmbeddingModule 一致的截断正态初始化。"""
    std = 2.0 / (fan_in + fan_out) ** 0.5
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)


# ── RMSNorm ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) ** 0.5
        return ((x / rms) * self.W).to(in_dtype)


# ── SwiGLU FFN ───────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model))
        _trunc_normal_init(self.W1, d_model, d_ff)
        _trunc_normal_init(self.W2, d_ff, d_model)
        _trunc_normal_init(self.W3, d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x1 = x @ self.W1.T
        x3 = x @ self.W3.T
        return (x1 * torch.sigmoid(x1)) * x3 @ self.W2.T


# ── Self Attention with RoPE ─────────────────────────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0

        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model))
        for w in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            _trunc_normal_init(w, d_model, d_model)

        self.rope = RoPE(theta, self.head_dim, max_seq_len)

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        q = (x @ self.q_proj.T).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = (x @ self.k_proj.T).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = (x @ self.v_proj.T).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        tp = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        tp = tp.unsqueeze(1).expand(-1, self.num_heads, -1)

        q = self.rope(q, tp)
        k = self.rope(k, tp)

        mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        d_k = self.head_dim
        scores = q.float() @ k.float().transpose(-2, -1) / math.sqrt(d_k)
        scores.masked_fill_(~mask, float('-inf'))
        attn = softmax(scores, dim=-1)
        out = (attn @ v.float()).transpose(1, 2).contiguous().view(B, S, self.d_model)
        return out @ self.o_proj.T


# ── Transformer Block ───────────────────────────────────────────────────

class TrainableTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ── Transformer Language Model ──────────────────────────────────────────

class TrainableTransformerLM(nn.Module):
    """可从零训练的 Transformer 语言模型。"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        _trunc_normal_init(self.token_embeddings.weight, vocab_size, d_model)

        self.layers = nn.ModuleList([
            TrainableTransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        _trunc_normal_init(self.lm_head.weight, d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, sequence_length) 整数 token ids
        Returns:
            logits: (batch_size, sequence_length, vocab_size)
        """
        h = self.token_embeddings(x)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_final(h)
        logits = self.lm_head(h)
        return logits
