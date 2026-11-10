import torch
import torch.nn as nn
from selfcompress import SelfCompressingLinear, quantize



class AttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size, init_num_bits, init_exponent):
        super().__init__()

        self.head_size = head_size
        self.scale = head_size ** -0.5      # Fixed at init: it must not change when Q/K dimensions are pruned

        self.query_proj = SelfCompressingLinear(embedding_size, head_size, True, init_num_bits, init_exponent)
        self.key_proj   = SelfCompressingLinear(embedding_size, head_size, True, init_num_bits, init_exponent)
        self.value_proj = SelfCompressingLinear(embedding_size, head_size, True, init_num_bits, init_exponent)


    def forward(self, x):
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        attn = q @ k.transpose(-2, -1)      # We tranpose only the last two dimensions of the matrice to not touch the batch dim
        attn = attn * self.scale
        attn = torch.softmax(attn, dim=-1)  # We apply the softmax on the last dimension e.i. the column
        attn = attn @ v

        return attn

    def get_num_bits(self):
        return self.query_proj.get_num_bits() + self.key_proj.get_num_bits() + self.value_proj.get_num_bits()



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_size, init_num_bits, init_exponent):
        super().__init__()

        self.head_size = embedding_size // num_heads
        self.out_proj  = SelfCompressingLinear(embedding_size, embedding_size, False, init_num_bits, init_exponent)
        self.heads     = nn.ModuleList([AttentionHead(embedding_size, self.head_size, init_num_bits, init_exponent) for _ in range(num_heads)])

    def forward(self, x):
        out = [head(x) for head in self.heads]
        out = torch.cat(out, dim=-1)
        out = self.out_proj(out)
        return out

    @torch.no_grad()
    def prune_heads(self, optimizer):
        pruned_count = 0
        out_keep_indices = []
        offset = 0          # Position of the head's V dimensions in the input of out_proj

        for head in self.heads:
            # Q/K: when the key dimension d has 0 bits, k_d is its bias for every token, so the score of row i
            # gets q_d(x_i) * b_d on every column: the softmax ignores it and dimension d can go from Q and K
            keep_k, cut_k = head.key_proj.get_keep_and_cut_indices()
            if len(cut_k) > 0:
                pruned_count += 2 * len(cut_k)
                head.query_proj.prune_output(keep_k, optimizer)
                head.key_proj.prune_output(keep_k, optimizer)

            # V: when dimension d has 0 bits, v_d is its bias for every token and the attention weights sum to 1,
            # so the head outputs this bias: we fold it into the bias of out_proj
            v_size        = head.value_proj.output_size
            keep_v, cut_v = head.value_proj.get_keep_and_cut_indices()
            if len(cut_v) > 0:
                pruned_count += len(cut_v)
                removed_bias = head.value_proj.bias.data[cut_v]
                out_weight   = quantize(self.out_proj.weight.data, self.out_proj.bits.data, self.out_proj.exponent.data)
                self.out_proj.bias.data += out_weight[:, offset + cut_v] @ removed_bias
                head.value_proj.prune_output(keep_v, optimizer)
            out_keep_indices.append(offset + keep_v)
            offset += v_size

        out_keep_indices = torch.cat(out_keep_indices)
        if len(out_keep_indices) < self.out_proj.input_size:
            self.out_proj.prune_input(out_keep_indices, optimizer)

        return pruned_count

    def get_num_bits(self):
        tot_num_bits = self.out_proj.get_num_bits()
        for head in self.heads:
            tot_num_bits = tot_num_bits + head.get_num_bits()
        return tot_num_bits


