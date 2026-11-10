import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from selfcompress import DynamicAdamW
from SCVisionTransformer.config import load_config
from SCVisionTransformer.data import build_dataloaders
from SCVisionTransformer.metrics import architecture_metrics, compression_metrics
from SCVisionTransformer.modules.visiontransformer import Visiontransformer
from SCVisionTransformer.tracking import CSVLogger, create_run_dir


# Console table: (title, key in the row, format). Adding a column = adding a line
CONSOLE_COLUMNS = [
    ("Epoch",     "epoch",                  "{:d}"),
    ("Train acc", "train_acc",              "{:.2f}%"),
    ("Test acc",  "test_acc",               "{:.2f}%"),
    ("Bits/w",    "bits_per_weight",        "{:.3f}"),
    ("Active w",  "pct_active_weights",     "{:.1f}%"),
    ("vs FP32",   "pct_model_size_vs_fp32", "{:.2f}%"),     # the whole model: weights, biases, LayerNorm, bit-widths, exponents
    ("Pruned",    "pruned",                 "{:d}"),
    ("Time",      "epoch_time",             "{:.1f}s"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train self-compressing ViT")
    parser.add_argument("--config", type=Path, required=True, help="e.g. configs/mnist.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. --set training.gamma=0.03 seed=2")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, gamma: float, device) -> dict:
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct  = torch.zeros((), device=device)
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits = model(x)
        task_loss = F.cross_entropy(logits, y)
        norm_num_bits = model.compute_compression_loss()

        loss = task_loss + gamma * norm_num_bits
        loss.backward()
        optimizer.step()

        loss_sum += task_loss.detach() * y.size(0)      # weighted by the batch size: the last batch is smaller
        correct  += (logits.argmax(dim=-1) == y).sum()
        total    += y.size(0)

        # The graph of this step keeps the old shapes of the parameters: free it before pruning
        del logits, task_loss, norm_num_bits, loss

    return {"train_loss": loss_sum.item() / total, "train_acc": 100.0 * correct.item() / total}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct  += (logits.argmax(dim=-1) == y).sum().item()
        total    += y.size(0)
    return {"test_loss": loss_sum / total, "test_acc": 100.0 * correct / total}


def print_header():
    print(" | ".join(f"{title:^10}" for title, _, _ in CONSOLE_COLUMNS))
    print("-" * (13 * len(CONSOLE_COLUMNS) - 3))


def print_row(row: dict):
    print(" | ".join(f"{fmt.format(row[key]):^10}" for _, key, fmt in CONSOLE_COLUMNS))


def print_architecture(model, config):
    head_size = config.model.embedding_dim // config.model.num_heads
    print("\n--- ViT after compression ---")
    for b_idx, blk in enumerate(model.blocks):
        mha = blk.multi_head_attention
        print(f"Transformer block {b_idx} :")
        print(f"  • Q/K dimension by head (initial = {head_size} per head) : {[h.key_proj.output_size for h in mha.heads]}")
        print(f"  • V dimension by head   (initial = {head_size} per head) : {[h.value_proj.output_size for h in mha.heads]}")
        print(f"  • Input of out_proj     (initial = {config.model.embedding_dim})          : {mha.out_proj.input_size}")
        print(f"  • Hidden dim of MLP     (initial = {config.model.hidden_dim})         : {blk.mlp.count_num_neurones_by_layer()[0]}")
    print(f"Remaining weights : {model.count_weights():,} / {model.initial_weights:,}")


def main():
    args = parse_args()
    config = load_config(args.config, args.set)

    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    train_loader, test_loader = build_dataloaders(config.data)

    model = Visiontransformer(
        image_shape=config.data.image_shape,
        num_classes=config.data.num_classes,
        **asdict(config.model)
    ).to(device)

    optimizer = DynamicAdamW(
        model=model,
        **asdict(config.optim)
    )

    run_dir = create_run_dir(config)
    logger  = CSVLogger(run_dir / "metrics.csv")
    print(f"Run : {run_dir} | device : {device} | initial weights : {model.initial_weights:,}")
    print_header()

    for epoch in range(1, config.training.epochs + 1):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, config.training.gamma, device)
        pruned = model.prune(optimizer) if epoch >= config.training.prune_after else 0
        epoch_time = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            **train_metrics,
            **evaluate(model, test_loader, device),
            **compression_metrics(model),
            "pruned": pruned,
            "lr_weight": optimizer.lr_weight,               # learning rates used during this epoch
            "lr_quantization": optimizer.lr_quantization,
            "epoch_time": epoch_time,
            **architecture_metrics(model),
        }
        logger.log(row)
        print_row(row)

        # Annealing at the end of training (Section Experiments): after the log, so the row keeps this epoch's lr
        if epoch in config.training.lr_decay_epochs:
            optimizer.scale_learning_rate(config.training.lr_decay_factor)

    print_architecture(model, config)


if __name__ == "__main__":
    main()