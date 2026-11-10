from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DataConfig:
    name: str
    image_shape: tuple[int, int, int]   # (C, H, W)
    num_classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    batch_size: int
    eval_batch_size: int


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    hidden_dim: int
    init_num_bits: int
    init_exponent: int


@dataclass(frozen=True)
class OptimConfig:
    lr_weight: float
    lr_quantization: float
    eps_weight: float
    eps_quantization: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    gamma: float                      # weight of the compression loss
    prune_after: int                  # first epoch with pruning
    lr_decay_epochs: tuple[int, ...]
    lr_decay_factor: float


@dataclass(frozen=True)
class Config:
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    training: TrainingConfig

    def __post_init__(self):
        c, h, w = self.data.image_shape
        p, d, n = self.model.patch_size, self.model.embedding_dim, self.model.num_heads
        _check(h % p == 0 and w % p == 0, f"model.patch_size={p} must divide data.image_shape {h}x{w}")
        _check(d % n == 0, f"model.embedding_dim={d} must be divisible by model.num_heads={n}")
        _check(d % 4 == 0, f"model.embedding_dim={d} must be divisible by 4 (2D sin/cos positional encoding)")
        _check(len(self.data.mean) == c and len(self.data.std) == c, f"data.mean and data.std need {c} values (one per channel)")


def load_config(path: str | Path, overrides: list[str] = ()) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        _apply_override(raw, item)
    return _build(Config, raw, prefix="")


def _check(condition: bool, message: str):
    if not condition:
        raise ConfigError(message)


def _apply_override(raw: dict, item: str):
    """'training.gamma=0.03' -> raw['training']['gamma'] = 0.03 (the value is parsed as YAML: '[40, 50]' -> list)"""
    key, sep, value = item.partition("=")
    _check(sep == "=" and key != "", f"invalid override {item!r}: expected key=value, e.g. training.gamma=0.03")
    *parents, leaf = key.split(".")
    node = raw
    for name in parents:
        node = node.setdefault(name, {})
        _check(isinstance(node, dict), f"invalid override {item!r}: {name} is not a section")
    node[leaf] = yaml.safe_load(value)


def _build(cls, raw, prefix: str):
    _check(isinstance(raw, dict), f"{prefix.rstrip('.') or 'the config'} must be a section (key: value)")
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(raw.keys() - set(names))
    missing = [n for n in names if n not in raw]
    _check(not unknown, "unknown key(s): " + ", ".join(prefix + k for k in unknown))
    _check(not missing, "missing key(s): " + ", ".join(prefix + k for k in missing))
    return cls(**{n: _coerce(hints[n], raw[n], prefix + n) for n in names})


def _coerce(hint, value, key: str):
    if is_dataclass(hint):
        return _build(hint, value, prefix=key + ".")

    if get_origin(hint) is tuple:
        _check(isinstance(value, (list, tuple)), f"{key}: expected a list, got {value!r}")
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:      # tuple[int, ...]: any length
            item_types = [args[0]] * len(value)
        else:                                           # tuple[int, int, int]: fixed length
            _check(len(value) == len(args), f"{key}: expected {len(args)} values, got {len(value)}")
            item_types = args
        return tuple(_coerce(t, v, f"{key}[{i}]") for i, (t, v) in enumerate(zip(item_types, value)))

    if hint is float and isinstance(value, (int, str)) and not isinstance(value, bool):
        try:
            return float(value)                         # PyYAML reads "2e-3" (no dot) as a string
        except ValueError:
            pass

    is_bool_as_int = hint is int and isinstance(value, bool)
    _check(isinstance(value, hint) and not is_bool_as_int, f"{key}: expected {hint.__name__}, got {value!r}")
    return value