import torch
from torch import Tensor
from torch import nn
from cs336_basics.Transformer.PreNormTransformerBlock import RMSNorm
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

        # Token embedding
        self.token_embeddings = EmbeddingModule(num_embddings=vocab_size, embdding_dim=d_model)
        self.token_embeddings.W.data = weights["token_embeddings.weight"]

        # Transformer blocks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_weights = {
                k.replace(f"layers.{i}.", ""): v
                for k, v in weights.items()
                if k.startswith(f"layers.{i}.")
            }
            self.layers.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    weights=layer_weights,
                )
            )

        # Final RMSNorm
        self.ln_final = RMSNorm(d_model, eps=1e-5)
        self.ln_final.W.data = weights["ln_final.weight"]

        # LM head (output projection)
        self.lm_head_weight = weights["lm_head.weight"]  # (vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length) — token indices

        # Token embedding
        h = self.token_embeddings(x)  # (batch, seq_len, d_model)

        # Transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Final LayerNorm
        h = self.ln_final(h)  # (batch, seq_len, d_model)

        # LM head: project to vocab
        logits = h @ self.lm_head_weight.T  # (batch, seq_len, vocab_size)

        return logits
