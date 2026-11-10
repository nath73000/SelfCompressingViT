import torch

from selfcompress import model_size_bits


@torch.no_grad()
def compression_metrics(model) -> dict:
    total_bits = model.compute_compression_loss().item() * model.initial_weights
    pct_active, pct_size_fp32 = model.stats()
    active_weights = pct_active / 100 * model.initial_weights
    size = model_size_bits(model)
    return {
        # --- What the loss sees: the weights only, with continuous bit-widths ---
        "bits_per_weight":        total_bits / model.initial_weights,     # = norm_num_bits, "Bits" in the table
        "bits_per_active_weight": total_bits / active_weights if active_weights > 0 else 0.0,
        "pct_active_weights":     pct_active,
        "pct_size_vs_fp32":       pct_size_fp32,
        "size_kb":                total_bits / 8 / 1024,
        "stored_weights":         model.count_weights(),                  # weights still in the tensors
        # --- What it takes to store the model: the weights on whole bits + everything stored next to them ---
        "weights_size_kb":        size["weight_bits"] / 8 / 1024,
        "overhead_size_kb":       size["overhead_bits"] / 8 / 1024,       # biases, LayerNorm, bit-widths, exponents
        "model_size_kb":          size["total_bits"] / 8 / 1024,
        "pct_model_size_vs_fp32": 100 * size["total_bits"] / model.initial_fp32_bits,
    }


def architecture_metrics(model) -> dict:
    """Size of every prunable dimension: the number of columns never changes, only their values"""
    row = {}
    for b, block in enumerate(model.blocks):
        for h, head in enumerate(block.multi_head_attention.heads):
            row[f"block{b}_head{h}_qk"] = head.key_proj.output_size
            row[f"block{b}_head{h}_v"]  = head.value_proj.output_size
        row[f"block{b}_mlp_hidden"] = block.mlp.count_num_neurones_by_layer()[0]
    return row