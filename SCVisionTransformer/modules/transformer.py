import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from selfcompress import SelfCompressingMLP


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, init_num_bits, init_exponent):
        super().__init__()
        self.num_heads     = num_heads
        self.embedding_dim = embedding_dim

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.multi_head_attention = MultiHeadAttention(num_heads, embedding_dim, init_num_bits, init_exponent)

        self.mlp = SelfCompressingMLP(
            embedding_dim,
            [hidden_dim, embedding_dim],
            init_num_bits, init_exponent,
            activation_layer=nn.GELU
        )

    def forward(self, x):
        out1 = x + self.multi_head_attention(self.layer_norm_1(x))
        out2 = out1 + self.mlp(self.layer_norm_2(out1))
        return out2

    @torch.no_grad()
    def prune(self, optimizer):
        pruned = self.multi_head_attention.prune_heads(optimizer)
        keep_cut_mlp = self.mlp.keep_and_cut_indices()
        pruned += sum(len(cut) for _, cut in keep_cut_mlp[:-1])
        self.mlp.prune_MLP(optimizer, keep_cut_mlp)
        return pruned

    def get_num_bits(self):
        return self.multi_head_attention.get_num_bits() + self.mlp.get_tot_num_bits()
