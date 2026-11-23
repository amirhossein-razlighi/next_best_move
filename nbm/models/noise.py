"""Initial sampling noise.

Two conventions, both from the paper's evaluations:
  row_noise    one CPU generator per row, seeded with that row's seed: a row's noise does not depend on the other rows of its
               batch, on the batch size or on the number of processes. Used by every task except chess image.
  batch_noise  one generator per batch on the DiT device, seeded with seed * 1_000_003 + batch index: the noise of a row
               depends on its batch. Used by the chess image model's evaluations (Table 1), whose batches are fixed groups
               of 8 boards with the same side to move.
The latent shape comes from the model (ImageModel.latent_shape / VideoModel.latent_shape).
"""
from __future__ import annotations

from typing import Sequence

import torch

BATCH_SEED_STRIDE = 1_000_003


def row_noise(seeds: Sequence[int], shape: Sequence[int]) -> torch.Tensor:
    """(len(seeds), *shape) float32 CPU tensor; row i = randn(shape) from a CPU generator seeded with seeds[i]."""
    return torch.stack([torch.randn(tuple(shape), generator=torch.Generator().manual_seed(int(s))) for s in seeds])


def batch_noise(seed: int, batch_index: int, n: int, shape: Sequence[int], device: torch.device | str) -> torch.Tensor:
    """(n, *shape) float32 tensor drawn on `device` from a generator seeded with seed * 1_000_003 + batch_index."""
    g = torch.Generator(device=device).manual_seed(seed * BATCH_SEED_STRIDE + batch_index)
    return torch.randn((n, *shape), generator=g, device=device, dtype=torch.float32)
