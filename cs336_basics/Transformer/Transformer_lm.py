import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import nn
import math
from cs336_basics.Transformer.PreNormTransformerBlock import RMSNorm, RoPE, SwiGLU, MultiHeadSelfAttention, MultiHeadSelfAttentionWithRoPE
from cs336_basics.Transformer.TransformerBlock import TransformerBlock
from cs336_basics.Transformer.EmbeddingModule import EmbeddingModule

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights

        # Token embedding
        self.embedding = EmbeddingModule(self.vocab_size, self.d_model)
        self.embedding.W.data = weights["token_embeddings.weight"]

        # 每一层创建一个 TransformerBlock
        self.layers = nn.ModuleList()
        for layer in range(self.num_layers):
            layer_weights = {
                "attn.q_proj.weight": weights[f"layers.{layer}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{layer}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{layer}.attn.v_proj.weight"],
                "attn.output_proj.weight": weights[f"layers.{layer}.attn.output_proj.weight"],
                "ln1.weight": weights[f"layers.{layer}.ln1.weight"],
                "ln2.weight": weights[f"layers.{layer}.ln2.weight"],
                "ffn.w1.weight": weights[f"layers.{layer}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{layer}.ffn.w2.weight"],
                "ffn.w3.weight": weights[f"layers.{layer}.ffn.w3.weight"],
            }
            self.layers.append(TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                max_seq_len=self.context_length,
                theta=self.rope_theta,
                weights=layer_weights,
            ))

        # Final RMSNorm
        self.ln_final = RMSNorm(self.d_model, weights["ln_final.weight"], eps=1e-5)

        # LM head weight
        self.lm_head_weight = weights["lm_head.weight"]  # (vocab_size, d_model)

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]):
        # Token embedding
        embedding = self.embedding(in_indices)  # (batch, seq_len, d_model)

        # 依次通过每一层 TransformerBlock
        for layer in self.layers:
            embedding = layer(embedding)

        # Final RMSNorm
        norm_out = self.ln_final(embedding)  # (batch, seq_len, d_model)

        # LM head: project to vocab
        logits = norm_out @ self.lm_head_weight.T  # (batch, seq_len, vocab_size)
        return logits
