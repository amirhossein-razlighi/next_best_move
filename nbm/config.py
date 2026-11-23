"""YAML configs with command-line overrides.

A config file (configs/<task>/<image|video>.yaml) holds every hyper-parameter of one paper model. Sections:
  task      name of the task package (nbm/tasks/<task>/)
  family    "image" (Qwen-Image-2.1) or "video" (Wan2.2-TI2V-5B)
  data      dir (under $NBM_DATA), train_limit; tasks may add their own keys here
  model     base (under $NBM_MODELS), resolution, lora {rank, alpha, dropout, targets};
            image: split_devices; video: flow_shift
  train     the optimisation / logging / checkpoint settings read by nbm.train (all keys required, see TRAIN_KEYS)
  eval      steps, batch_size, seed; tasks may add their own keys (e.g. a loop mode)
  options   free-form task options (e.g. a target style), read only by the task
Overrides are "dotted.key=value" strings; the value is parsed as YAML (so 20 is an int, 1.0e-4 a float, null None,
[a, b] a list). Write floats with a dot ("1.0e-4"): YAML reads "1e-4" as a string.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

FAMILIES = ("image", "video")
TOP_KEYS = {"task", "family", "data", "model", "train", "eval"}
DATA_KEYS = {"dir", "train_limit"}
LORA_KEYS = {"rank", "alpha", "dropout", "targets"}
MODEL_KEYS = {"base", "resolution", "lora"}
MODEL_FAMILY_KEYS = {"image": {"split_devices"}, "video": {"flow_shift"}}
TRAIN_KEYS = {"output_dir", "processes", "batch_size", "grad_accum", "learning_rate", "weight_decay", "lr_scheduler",
              "warmup_steps", "max_steps", "max_grad_norm", "gradient_checkpointing", "seed", "num_workers", "log_every",
              "save_every", "eval_every", "eval_n", "eval_steps", "eval_batch_size", "dist_timeout_min", "resume"}
TRAIN_FAMILY_KEYS = {"image": {"highlight_loss_weight"}, "video": set()}
EVAL_KEYS = {"steps", "batch_size", "seed"}


class Config(dict):
    """A dict with attribute access for nested sections: cfg.train.batch_size == cfg["train"]["batch_size"]."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"config has no key {key!r} (keys: {sorted(self)})") from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Config({k: copy.deepcopy(v, memo) for k, v in self.items()})

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """cfg.get_path("eval.votes", 1): a nested value, or default if any level is missing."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return _plain(self)


def _wrap(x: Any) -> Any:
    if isinstance(x, dict):
        return Config({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x


def apply_override(cfg: Config, item: str) -> None:
    """Apply one "a.b.c=value" override in place (intermediate sections are created if missing)."""
    if "=" not in item:
        raise ValueError(f"override {item!r} is not of the form key.path=value")
    key, raw = item.split("=", 1)
    value = _wrap(yaml.safe_load(raw)) if raw != "" else ""
    node = cfg
    parts = key.strip().split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = Config()
        node = node[part]
    node[parts[-1]] = value


def validate(cfg: Config) -> None:
    """Check that every section has the keys the core reads (and nothing it would silently ignore in model / train)."""
    missing = TOP_KEYS - set(cfg)
    if missing:
        raise ValueError(f"config is missing top-level keys {sorted(missing)}")
    if cfg.family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {cfg.family!r}")
    fam = cfg.family

    def check(section: str, required: set, strict: bool) -> None:
        have = set(cfg[section])
        if required - have:
            raise ValueError(f"config section {section!r} is missing {sorted(required - have)}")
        if strict and have - required:
            raise ValueError(f"config section {section!r} has unknown keys {sorted(have - required)}")

    check("data", DATA_KEYS, strict=False)
    check("model", MODEL_KEYS | MODEL_FAMILY_KEYS[fam], strict=True)
    check("train", TRAIN_KEYS | TRAIN_FAMILY_KEYS[fam], strict=True)
    check("eval", EVAL_KEYS, strict=False)
    lora = set(cfg.model.lora)
    if lora != LORA_KEYS:
        raise ValueError(f"model.lora must have exactly the keys {sorted(LORA_KEYS)}, got {sorted(lora)}")
    if "options" in cfg and not isinstance(cfg.options, dict):
        raise ValueError("options must be a mapping")


def from_dict(d: dict, overrides: Iterable[str] = ()) -> Config:
    """A validated Config from a plain dict, with the overrides applied in order (an empty `options` section is added if absent)."""
    cfg = _wrap(copy.deepcopy(d))
    if not isinstance(cfg, Config):
        raise ValueError("a config must be a mapping")
    for item in overrides:
        apply_override(cfg, item)
    cfg.setdefault("options", Config())
    validate(cfg)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Read a YAML config, apply the overrides in order, validate."""
    with open(path) as fh:
        d = yaml.safe_load(fh)
    if not isinstance(d, dict):
        raise ValueError(f"{path}: not a mapping")
    return from_dict(d, overrides)


def dump_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
