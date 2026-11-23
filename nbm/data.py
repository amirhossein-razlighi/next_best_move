"""Training data: per-access examples from the task, collation, and the infinite batch streams of the paper's runs.

Batch streams (both seeded by train.seed, both resumable by skipping the batches of the steps already trained):
  GroupBatchSampler    image model: every batch holds rows of one prompt-layout group (Task.batch_groups); per batch the group
                       is drawn with probability proportional to its size, then the next rows of that group's permutation
                       (a fresh permutation when it runs out).
  ShardedBatchSampler  video model: consecutive batches of one permutation stream over all rows (prompt embeddings have a fixed
                       length, so any rows can share a batch).
Both enumerate global batches; process r of W takes the global batches g with g % W == r, so every process draws disjoint
batches and the global batch of one optimizer step is batch_size x grad_accum x W.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, Sampler

from nbm.tasks.base import AccessRNG, RowStore, Task


class GroupBatchSampler(Sampler):
    def __init__(self, groups: np.ndarray, batch_size: int, seed: int, rank: int = 0, world: int = 1, skip: int = 0):
        """groups: int label per row (0 .. K-1). skip: global batches to pass over (resume)."""
        groups = np.asarray(groups)
        n_groups = int(groups.max()) + 1 if len(groups) else 1
        self.pools = [np.flatnonzero(groups == g) for g in range(n_groups)]
        self.cum = np.cumsum([len(p) for p in self.pools]) / len(groups)
        self.bs, self.seed, self.rank, self.world, self.skip = batch_size, seed, rank, world, skip

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        perms = [rng.permutation(p) for p in self.pools]
        ptr = [0] * len(perms)
        g = 0
        while True:
            k = int(np.searchsorted(self.cum, rng.random(), side="right"))
            if ptr[k] + self.bs > len(perms[k]):
                perms[k], ptr[k] = rng.permutation(self.pools[k]), 0
            batch = perms[k][ptr[k]:ptr[k] + self.bs]
            ptr[k] += self.bs
            if g >= self.skip and g % self.world == self.rank:
                yield batch.tolist()
            g += 1

    def __len__(self) -> int:
        return 10**12


class ShardedBatchSampler(Sampler):
    def __init__(self, n: int, batch_size: int, seed: int, rank: int = 0, world: int = 1, skip: int = 0):
        self.n, self.bs, self.seed, self.rank, self.world, self.skip = n, batch_size, seed, rank, world, skip

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        perm, ptr, g = rng.permutation(self.n), 0, 0
        while True:
            if ptr + self.bs > self.n:
                perm, ptr = rng.permutation(self.n), 0
            batch = perm[ptr:ptr + self.bs]
            ptr += self.bs
            if g >= self.skip and g % self.world == self.rank:
                yield batch.tolist()
            g += 1

    def __len__(self) -> int:
        return 10**12


def to_vae_tensor(img: Image.Image) -> torch.Tensor:
    """RGB / RGBA PIL -> (4, H, W) in [-1, 1]: the Qwen-Image-2.1 VAE input (it takes RGBA)."""
    return TF.normalize(TF.to_tensor(img.convert("RGBA")), [0.5], [0.5])


class ImageTrainSet(Dataset):
    """Item i = task.image_example(row i, AccessRNG()) as tensors. torch's RNG is seeded per DataLoader worker, so the
    workers draw different per-access randomness."""

    def __init__(self, task: Task, df):
        self.task, self.rows = task, RowStore(df)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        ex = self.task.image_example(self.rows[i], AccessRNG())
        if ex.cond.size != ex.target.size:
            raise ValueError(f"row {i}: condition {ex.cond.size} and target {ex.target.size} differ in size")
        return {"cond_pil": ex.cond, "cond": to_vae_tensor(ex.cond), "target": to_vae_tensor(ex.target),
                "boxes": [tuple(int(v) for v in b) for b in ex.boxes], "prompt": ex.prompt}


def collate_image(items: list[dict]) -> dict:
    return {"cond_pil": [e["cond_pil"] for e in items], "cond": torch.stack([e["cond"] for e in items]),
            "target": torch.stack([e["target"] for e in items]), "boxes": [e["boxes"] for e in items],
            "prompts": [e["prompt"] for e in items]}


class VideoTrainSet(Dataset):
    """Item i = task.video_example(row i, AccessRNG()) as tensors."""

    def __init__(self, task: Task, df):
        self.task, self.rows = task, RowStore(df)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        ex = self.task.video_example(self.rows[i], AccessRNG())
        blocks = np.asarray(ex.blocks)
        if blocks.dtype != np.uint8 or blocks.ndim != 4 or blocks.shape[-1] != 3:
            raise ValueError(f"row {i}: blocks must be uint8 (17, H, W, 3), got {blocks.dtype} {blocks.shape}")
        w = None if ex.block_weights is None else torch.as_tensor(np.asarray(ex.block_weights, np.float32))
        return {"blocks": torch.from_numpy(np.ascontiguousarray(blocks)), "prompt": ex.prompt, "block_weights": w}


def collate_video(items: list[dict]) -> dict:
    ws = [e["block_weights"] for e in items]
    if any(w is None for w in ws) and not all(w is None for w in ws):
        raise ValueError("block_weights given for some examples of a batch but not for others")
    return {"blocks": torch.stack([e["blocks"] for e in items]), "prompts": [e["prompt"] for e in items],
            "block_weights": None if ws[0] is None else torch.stack(ws)}
