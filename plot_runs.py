"""Plot one or several runs saved by train.py.

    python plot_runs.py                                       # most recent run    -> <run>/training.png
    python plot_runs.py runs/mnist/<run>                      # one run            -> <run>/training.png
    python plot_runs.py runs/mnist/<run1> runs/mnist/<run2>   # compare runs       -> runs/comparison.png
    python plot_runs.py --mean runs/mnist/*_s1 runs/mnist/*_s2 runs/mnist/*_s3
                                                              # average the seeds  -> runs/mean.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Light chart theme: surface, inks, chrome, then one color per curve (fixed order, never cycled)
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

PANELS = [  # (column of the dataframe, title)
    ("test_acc",         "Test accuracy (%)"),
    ("loss",             "Loss: test (solid), train (dashed)"),
    ("size",             None),         # column and title chosen by size_panel()
    ("qk_kept",          "Q/K dimensions kept (%)"),
    ("v_kept",           "V dimensions kept (%)"),
    ("mlp_kept",         "MLP hidden neurons kept (%)"),
]

SIZE_PANELS = [  # first column found in every run: older runs only logged the size of the weights
    ("pct_model_size_vs_fp32", "Model size (% of float32)"),
    ("pct_size_vs_fp32",       "Weights only (% of float32)"),
]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def load_run(run_dir: Path):
    config = json.loads((run_dir / "config.json").read_text())
    df = pd.read_csv(run_dir / "metrics.csv")
    model = config["model"]
    initial_dims = model["num_layers"] * model["embedding_dim"]       # Q/K (or V) of all heads of all blocks
    df["qk_kept"]  = 100 * df.filter(regex=r"_qk$").sum(axis=1) / initial_dims
    df["v_kept"]   = 100 * df.filter(regex=r"_v$").sum(axis=1) / initial_dims
    df["mlp_kept"] = 100 * df.filter(regex=r"_mlp_hidden$").sum(axis=1) / (model["num_layers"] * model["hidden_dim"])
    return config, df


def size_panel(runs: list) -> tuple:
    """(column, title) of the size panel: the whole model when every run logged it, else the weights only"""
    for column, title in SIZE_PANELS:
        if all(column in df for _, df in runs):
            return column, title
    raise SystemExit("No size column found in the metrics of these runs")


def without_seed(config: dict) -> dict:
    return {k: v for k, v in config.items() if k != "seed"}


def make_curves(runs: list, mean: bool) -> list:
    """One curve per run, or with mean=True one curve per config: the runs that only differ by their seed
    are averaged epoch by epoch, with their min and max kept for the shaded band"""
    groups = {}
    for i, (config, df) in enumerate(runs):
        key = json.dumps(without_seed(config), sort_keys=True) if mean else i
        groups.setdefault(key, []).append((config, df))

    curves = []
    for members in groups.values():
        dfs = [df for _, df in members]
        common = set.intersection(*(set(df.epoch) for df in dfs))     # epochs reached by every run of the group
        by_epoch = pd.concat([df[df.epoch.isin(common)] for df in dfs]).groupby("epoch")
        curves.append({
            "config": members[0][0],
            "seeds":  [c["seed"] for c, _ in members],
            "runs":   dfs,
            "mean":   by_epoch.mean().reset_index(),
            "min":    by_epoch.min().reset_index() if len(dfs) > 1 else None,
            "max":    by_epoch.max().reset_index() if len(dfs) > 1 else None,
        })
    return curves


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def curve_labels(curves: list, run_dirs: list, mean: bool) -> list:
    """Name each curve by what differs from the others, e.g. 'gamma=0.02, patch_size=4'"""
    configs = [without_seed(c["config"]) if mean else c["config"] for c in curves]
    flat = [flatten(c) for c in configs]
    differing = [k for k in flat[0] if len({json.dumps(f.get(k)) for f in flat}) > 1]
    if differing:
        return [", ".join(f"{k.split('.')[-1]}={f.get(k)}" for k in differing) for f in flat]
    if mean:
        return [f"mean of {len(c['seeds'])} seeds" for c in curves]
    return [d.name for d in run_dirs]


def mean_std(values: list) -> str:
    """'98.18' for one run, '98.18 ± 0.05' (standard deviation between seeds) for several"""
    values = pd.Series(values, dtype=float)
    return f"{values.mean():.2f}" if len(values) == 1 else f"{values.mean():.2f} ± {values.std():.2f}"


def style_axis(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def end_dot(ax, x, y, color):
    ax.plot([x], [y], "o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)


def mark_events(axes, configs):
    """Vertical hairlines where pruning starts and where the learning rate drops (labelled on the first panel)"""
    events = {}
    for c in configs:
        events[c["training"]["prune_after"]] = "pruning"
        for e in c["training"]["lr_decay_epochs"]:
            events[e] = f"lr ×{c['training']['lr_decay_factor']}"
    for ax in axes.flat:
        for x in events:
            ax.axvline(x, color=BASELINE, linewidth=1, zorder=0)
    for x, text in events.items():
        axes.flat[0].text(x + 0.5, 0.03, text, transform=axes.flat[0].get_xaxis_transform(), fontsize=8, color=MUTED)


def plot(curves: list, labels: list, size: tuple):
    panels = [size if column == "size" else (column, title) for column, title in PANELS]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    single = len(curves) == 1

    for curve, label, color in zip(curves, labels, SERIES):
        df, last = curve["mean"], curve["mean"].iloc[-1]
        for ax, (column, _) in zip(axes.flat, panels):
            if column == "loss":
                ax.plot(df.epoch, df.test_loss, color=color, linewidth=2, label=label)
                ax.plot(df.epoch, df.train_loss, color=color, linewidth=2, linestyle=(0, (4, 2)))
                column = "test_loss"                                  # the band below follows the test loss
            else:
                ax.plot(df.epoch, df[column], color=color, linewidth=2)
                end_dot(ax, last.epoch, last[column], color)
                if single:      # one curve: the final value next to its end dot
                    ax.annotate(f"{last[column]:.2f}" if column == "test_acc" else f"{last[column]:.1f}",
                                (last.epoch, last[column]), xytext=(8, 0), textcoords="offset points",
                                va="center", fontsize=9.5, color=INK)
            if curve["min"] is not None:                              # several seeds: min-max band
                ax.fill_between(df.epoch, curve["min"][column], curve["max"][column], color=color, alpha=0.12, linewidth=0)

    for ax, (_, title) in zip(axes.flat, panels):
        style_axis(ax, title)
    for ax in axes[1]:
        ax.set_xlabel("Epoch", fontsize=9, color=MUTED)

    # Accuracy zoomed past the first epochs (otherwise 80% -> 98% flattens the end of the curve)
    frames = [f for c in curves for f in (c["mean"], c["min"], c["max"]) if f is not None]
    accs = pd.concat([f.test_acc[f.epoch >= 5] for f in frames])
    axes.flat[0].set_ylim(accs.min() - 0.3, accs.max() + 0.3)
    axes.flat[1].set_yscale("log")
    for ax in axes.flat[2:]:
        ax.set_ylim(bottom=0)
    max_epoch = max(c["mean"].epoch.max() for c in curves)
    axes.flat[0].set_xlim(0, max_epoch * 1.12)          # room for the end labels
    mark_events(axes, [c["config"] for c in curves])

    datasets = sorted({c["config"]["data"]["name"] for c in curves})
    fig.text(0.01, 0.985, f"Self-compressing ViT · {', '.join(datasets)}", fontsize=13, fontweight="semibold", color=INK, va="top")
    if single:
        curve, config = curves[0], curves[0]["config"]
        seeds = curve["seeds"]
        seed_text = f"seed {seeds[0]}" if len(seeds) == 1 else f"mean of seeds {', '.join(map(str, seeds))}, shaded: min to max"
        last10 = mean_std([df.test_acc.tail(10).mean() for df in curve["runs"]])
        subtitle = (f"γ = {config['training']['gamma']} · patch {config['model']['patch_size']} · "
                    f"{config['training']['epochs']} epochs · {seed_text} · "
                    f"test accuracy of the last 10 epochs: {last10}%")
    else:
        shaded = " · shaded: min to max over the seeds" if any(c["min"] is not None for c in curves) else ""
        subtitle = f"{len(curves)} curves · each one is named by what differs from the others{shaded}"
        fig.legend(*axes.flat[1].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.005, 0.915),
                   ncol=min(len(curves), 4), frameon=False, fontsize=9.5, labelcolor=INK_2, handlelength=1.5)
    fig.text(0.01, 0.945, subtitle, fontsize=10, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90 if single else 0.86), h_pad=2.5, w_pad=2.5)
    return fig


def print_summary(curves: list, labels: list, size: tuple):
    """The same numbers as text (mean ± standard deviation between seeds when several are averaged)"""
    size_column, _ = size
    size_title = "model size % fp32" if size_column == "pct_model_size_vs_fp32" else "weights only % fp32"
    rows = [{
        "curve":          label,
        "seeds":          ", ".join(map(str, c["seeds"])),
        "final test acc": mean_std([df.test_acc.iloc[-1] for df in c["runs"]]),
        "last-10 mean":   mean_std([df.test_acc.tail(10).mean() for df in c["runs"]]),
        size_title:       mean_std([df[size_column].iloc[-1] for df in c["runs"]]),
        "Q/K kept %":     mean_std([df.qk_kept.iloc[-1] for df in c["runs"]]),
        "V kept %":       mean_std([df.v_kept.iloc[-1] for df in c["runs"]]),
        "MLP kept %":     mean_std([df.mlp_kept.iloc[-1] for df in c["runs"]]),
    } for c, label in zip(curves, labels)]
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Plot runs saved by train.py")
    parser.add_argument("runs", nargs="*", type=Path, help="run folders (default: the most recent run)")
    parser.add_argument("--mean", action="store_true", help="average the runs that only differ by their seed")
    parser.add_argument("--out", type=Path, help="output image (default: <run>/training.png, runs/comparison.png or runs/mean.png)")
    args = parser.parse_args()

    run_dirs = args.runs
    if not run_dirs:
        finished = [p.parent for p in Path("runs").glob("*/*/metrics.csv")]
        if not finished:
            raise SystemExit("No run found in runs/")
        run_dirs = [max(finished, key=lambda p: p.name)]          # folder names start with the date

    runs   = [load_run(d) for d in run_dirs]
    curves = make_curves(runs, mean=args.mean)
    if len(curves) > len(SERIES):
        raise SystemExit(f"At most {len(SERIES)} curves per figure: compare fewer runs")
    labels = curve_labels(curves, run_dirs, mean=args.mean)
    size   = size_panel(runs)
    fig    = plot(curves, labels, size)

    if args.out:
        out = args.out
    elif args.mean:
        out = Path("runs") / "mean.png"
    else:
        out = run_dirs[0] / "training.png" if len(run_dirs) == 1 else Path("runs") / "comparison.png"
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print_summary(curves, labels, size)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
