"""The Task interface: everything task-specific that the core trainer and evaluator need.

A task (one package nbm/tasks/<name>/, its __init__.py sets TASK = <the Task subclass>) owns its data, its pictures and its
evaluation; the core owns the models, the optimisation and the files. The core calls:

  task = TASK(cfg)                          the full config (nbm.config.Config) of one model; cfg.family is "image" or "video"
  task.load_split(split) -> DataFrame       rows of a split, in evaluation order ("train", "val", "test", or task-specific names)
  task.batch_groups(df) -> int array        image training only: rows with the same label share a prompt layout (default: all 0)
  task.image_example(row, rng) -> ImageExample      image training: condition, target, prompt, highlight boxes
  task.video_example(row, rng) -> VideoExample      video training: the 17 clip blocks, prompt, optional block weights
  task.prompts() -> list[str]               video: every prompt the task uses (embedded once with the text encoder)
  task.evaluate(ctx, df) -> EvalResult      the task's evaluation loop on df (one-shot, closed loop, open-loop clip, receding
                                            horizon, ...), sampling through ctx (an EvalContext: model, steps, votes, seed,
                                            sharding over processes); the trainer also calls it on the
                                            validation prefix as a training monitor

Rows are plain dicts (column -> value) in training (row = one training item) and whatever the task likes in evaluate.
`rng` is None when the core needs the row's deterministic example; otherwise an AccessRNG for per-access randomness (e.g. a
random state along a path). See nbm/tasks/README.md for a worked example.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image

from nbm import paths
from nbm.config import Config
from nbm.models.noise import BATCH_SEED_STRIDE
from nbm.progress import Progress

Box = tuple[int, int, int, int]  # pixel box (x0, y0, x1, y1), x1 / y1 exclusive
SAMPLE_SEED_STRIDE = 10_000_019


def row_seed(seed: int, row: int, sample: int = 0) -> int:
    """The common per-row noise seed: seed * 1,000,003 + row + 10,000,019 * sample (sample k of a majority vote). Tasks whose
    paper evaluation used another formula (e.g. one more term per closed-loop round) compute their own."""
    return seed * BATCH_SEED_STRIDE + row + SAMPLE_SEED_STRIDE * sample


@dataclass
class ImageExample:
    """One image training example. cond / target: RGB pictures of the same size (multiples of 32 px). prompt: the instruction.
    boxes: pixel boxes of the highlight region whose latents get the loss weight train.highlight_loss_weight (e.g. the cells
    of the marked action); may be empty when that weight is 1."""
    cond: Image.Image
    target: Image.Image
    prompt: str
    boxes: Sequence[Box] = ()


@dataclass
class VideoExample:
    """One video training example. blocks: (17, H, W, 3) uint8, block 0 = the condition picture, blocks 1..16 the clip
    (nbm.models.video: each is held for 4 frames). block_weights: optional (16,) loss weight of blocks 1..16 (None = the plain
    mean, as in every paper run)."""
    blocks: np.ndarray
    prompt: str
    block_weights: np.ndarray | None = None


@dataclass
class EvalResult:
    """records: one JSON-serialisable dict per evaluated item (row, game, map, ...), written to predictions.jsonl.
    metrics: JSON-serialisable summary, written to metrics.json (numeric entries are also logged by the training monitor)."""
    records: list[dict]
    metrics: dict


class AccessRNG:
    """Randomness of one training access (a row can give a different example every time it is drawn).

    Backed by torch's default RNG unless a generator is given. Inside DataLoader workers torch seeds that RNG per worker, which
    is how the training stream stays reproducible: uniform() is torch.rand(()) and randint(n) is torch.randint(0, n, ())."""

    def __init__(self, generator: torch.Generator | None = None):
        self.generator = generator

    def uniform(self) -> float:
        """A float in [0, 1)."""
        return torch.rand((), generator=self.generator).item()

    def randint(self, high: int) -> int:
        """An int in [0, high)."""
        return int(torch.randint(0, high, (), generator=self.generator).item())


@dataclass
class EvalContext:
    """What a task's evaluate() gets: the model with its LoRA, the sampling settings and the process layout. Task-specific
    evaluation settings (e.g. a loop mode) are read from ctx.cfg.eval / ctx.cfg.options.

    Sharding: a task splits its work into global batches (lists of row indices, in a fixed order), process r handles the batches
    k with k % world == r (`my_batches`), then `gather` collects everybody's results on every process. With per-row noise the
    results do not depend on the number of processes. Only the main process (rank 0) should write files or log."""
    cfg: Config
    model: Any  # ImageModel or VideoModel
    split: str
    steps: int
    votes: int
    seed: int
    batch_size: int
    rank: int = 0
    world: int = 1
    out_dir: Path | None = None  # where the main process may write pictures / strips (None: write nothing)
    in_training: bool = False  # True for the validation monitor run during training (a task may use a quicker check)

    @property
    def family(self) -> str:
        return self.cfg.family

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def flow_shift(self) -> float:
        """Video sampler shift (model.flow_shift)."""
        return float(self.cfg.model.flow_shift)

    def global_batches(self, items: Sequence | int, bs: int | None = None) -> list[list]:
        """Consecutive batches of `items` (or of range(items)) of size bs (default ctx.batch_size)."""
        items = list(range(items)) if isinstance(items, int) else list(items)
        bs = bs or self.batch_size
        return [items[j:j + bs] for j in range(0, len(items), bs)]

    def my_batches(self, items: Sequence | int, bs: int | None = None) -> list[tuple[int, list]]:
        """[(k, batch)] of this process: global batch k goes to process k % world."""
        return [(k, b) for k, b in enumerate(self.global_batches(items, bs)) if k % self.world == self.rank]

    def gather(self, items: list) -> list:
        """Concatenation of every process's list, on every process (order: rank 0's items, then rank 1's, ...)."""
        if self.world == 1 or not (dist.is_available() and dist.is_initialized()):
            return list(items)
        buf: list = [None] * dist.get_world_size()
        dist.all_gather_object(buf, items)
        return [x for b in buf for x in b]

    def barrier(self) -> None:
        if self.world > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def log(self, msg: str) -> None:
        if self.is_main:
            print(msg, flush=True)

    def progress(self, total: int | None, desc: str, unit: str = "batch") -> Progress:
        return Progress(total=total, desc=desc, unit=unit, enabled=self.is_main)

    def generate(self, inputs: Sequence[Image.Image], prompts: Sequence[str], seeds: Sequence[int]):
        """Sample one output per input with per-row noise and ctx.steps sampling steps. Image model: a list of PIL pictures;
        video model: a uint8 tensor (B, 65, H, W, 3) (nbm.models.video.video_to_blocks reads the 17 blocks of a clip).
        For other noise (e.g. nbm.models.noise.batch_noise) call ctx.model.generate directly."""
        if self.family == "image":
            return self.model.generate_seeded(inputs, prompts, self.steps, seeds)
        return self.model.generate_seeded(inputs, prompts, self.steps, self.flow_shift, seeds)


class Task:
    """Base class of a task; subclass it in nbm/tasks/<name>/ and set TASK = <subclass> in the package's __init__.py."""

    families: tuple[str, ...] = ("image", "video")  # the models this task supports

    def __init__(self, cfg: Config):
        if cfg.family not in self.families:
            raise ValueError(f"task {cfg.task!r} has no {cfg.family} model (supported: {self.families})")
        self.cfg = cfg

    # ---------- convenience ----------
    @property
    def family(self) -> str:
        return self.cfg.family

    @property
    def data_dir(self) -> Path:
        """data.dir resolved under $NBM_DATA."""
        return paths.data_path(self.cfg.data.dir)

    @property
    def resolution(self) -> int:
        return int(self.cfg.model.resolution)

    @property
    def options(self) -> Config:
        return self.cfg.options

    # ---------- data ----------
    def load_split(self, split: str) -> pd.DataFrame:
        """All rows of a split in the evaluation order (the core takes prefixes: data.train_limit for training, --limit and
        train.eval_n for evaluation)."""
        raise NotImplementedError

    def batch_groups(self, df: pd.DataFrame) -> np.ndarray:
        """Image training: an int label per row; a training batch holds rows of one label only (their prompts must share a
        token layout, e.g. label 0 = white to move, 1 = black). The label of each batch is drawn with probability
        proportional to its row count. Default: one group."""
        return np.zeros(len(df), dtype=np.int64)

    # ---------- training examples ----------
    def image_example(self, row: dict, rng: AccessRNG | None) -> ImageExample:
        raise NotImplementedError(f"task {type(self).__name__} has no image model")

    def video_example(self, row: dict, rng: AccessRNG | None) -> VideoExample:
        raise NotImplementedError(f"task {type(self).__name__} has no video model")

    def prompts(self) -> list[str]:
        """Video: every prompt the task's clips use (they are embedded once, before the DiT is loaded)."""
        raise NotImplementedError(f"task {type(self).__name__} does not list its video prompts")

    # ---------- evaluation ----------
    def evaluate(self, ctx: EvalContext, df: pd.DataFrame) -> EvalResult:
        """Run the task's evaluation on df with ctx's model / steps / votes / seed; called on every process, must return the
        full (gathered) result on every process."""
        raise NotImplementedError


def jsonable(x: Any) -> Any:
    """numpy / torch scalars and arrays -> plain Python, recursively (for predictions.jsonl / metrics.json)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def write_result(result: EvalResult, out_dir: str | os.PathLike, meta: dict) -> None:
    """metrics.json (meta + metrics), predictions.jsonl, then _DONE."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as fh:
        json.dump(jsonable({**meta, **result.metrics}), fh, indent=2)
    with open(out / "predictions.jsonl", "w") as fh:
        for r in result.records:
            fh.write(json.dumps(jsonable(r)) + "\n")
    (out / "_DONE").touch()


class RowStore:
    """Training rows: store[i] -> {column: value} with plain Python values (as df.to_dict("records")), kept column-wise so
    that DataLoader workers do not copy millions of dicts."""

    def __init__(self, df: pd.DataFrame):
        self.cols = {c: df[c].tolist() for c in df.columns}
        self.n = len(df)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {c: v[i] for c, v in self.cols.items()}
