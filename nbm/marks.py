"""The marks M(x, a) drawn into a picture to express an action, and the shared decoder D that fits their strengths.

Every task draws its action as coloured marks on top of the condition picture x: an alpha blend
    out = (1 - alpha * m) * x + alpha * m * colour
with coverage m = 1 inside a box (a tint, `tint`) or a soft coverage map (a dot, `blend`). The decoder inverts the blend: per
region, the change out - x is a linear combination of (colour_k - x) over the colours that can appear there, so a least-squares
fit (`fit_strengths`) gives one coefficient per colour; divided by alpha (`to_strength`) it is 1 for a mark drawn at full
target strength and 0 for no mark. The readout (nbm/readout.py) then takes the strongest mark.

Colours and alpha are the paper's: BLUE marks where an action starts, RED where it ends, GREEN extra information (legal
destinations, a path); ALPHA = 0.6.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

BLUE = (20, 80, 255)
RED = (240, 20, 20)
GREEN = (10, 200, 40)
ALPHA = 0.6
MAX_STRENGTH = 1.5  # strengths are clipped to [0, MAX_STRENGTH]

Box = tuple[int, int, int, int]  # pixel box (x0, y0, x1, y1), x1 / y1 exclusive


def as_array(img: Image.Image | np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """RGB float32 array (H, W, 3); a PIL image of another size is first resized (bilinear) to `size` = (width, height)."""
    if isinstance(img, np.ndarray):
        return img.astype(np.float32)
    img = img.convert("RGB")
    if size is not None and img.size != tuple(size):
        img = img.resize(tuple(size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.round().clip(0, 255).astype(np.uint8))


def tint(img: Image.Image, boxes_colors: Sequence[tuple[Box, Sequence[float]]], alpha: float = ALPHA) -> Image.Image:
    """Blend each box with its colour, out = (1 - alpha) * img + alpha * colour, rounded once at the end."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32).copy()
    for (x0, y0, x1, y1), color in boxes_colors:
        arr[y0:y1, x0:x1] = (1 - alpha) * arr[y0:y1, x0:x1] + alpha * np.asarray(color, np.float32)
    return _to_image(arr)


def blend(img: Image.Image, coverage: np.ndarray, color: Sequence[float], alpha: float = ALPHA) -> Image.Image:
    """Blend a soft mark: coverage (H, W) in [0, 1] (e.g. an anti-aliased disc), out = (1 - alpha m) img + alpha m colour."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32).copy()
    m = (alpha * coverage.astype(np.float32))[..., None]
    arr = (1 - m) * arr + m * np.asarray(color, np.float32)
    return _to_image(arr)


def fit_strengths(out: np.ndarray, cond: np.ndarray, boxes: Sequence[Box], colors: Sequence[Sequence[float]],
                  margin: int = 0, coverages: Sequence[np.ndarray | None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares mark coefficients per box.

    out, cond: float32 (H, W, 3) arrays (as_array) of the generated and the condition picture. For box k only the inner pixels
    (`margin` px dropped at each edge: grid lines, VAE blur) are used; the change out - cond is regressed on one column per colour,
    (colour - cond), or coverage * (colour - cond) if that colour has a coverage map (H, W). Returns (coef (K, C) float64,
    residual (K,) = mean absolute residual of the fit, 0..255 scale). Coefficients are in blend units (alpha for a full mark):
    pass them to `to_strength`."""
    coef = np.zeros((len(boxes), len(colors)))
    res = np.zeros(len(boxes))
    cols_f32 = [np.asarray(c, np.float32) for c in colors]
    covs = list(coverages) if coverages is not None else [None] * len(colors)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        ys, xs = slice(y0 + margin, y1 - margin), slice(x0 + margin, x1 - margin)
        cc = cond[ys, xs].reshape(-1, 3)
        d = (out[ys, xs].reshape(-1, 3) - cc).reshape(-1)
        cols = []
        for color, cov in zip(cols_f32, covs):
            if cov is None:
                cols.append((color - cc).reshape(-1))
            else:
                cols.append((cov[ys, xs].reshape(-1, 1) * (color - cc)).reshape(-1))
        A = np.stack(cols, axis=1)
        c, *_ = np.linalg.lstsq(A, d, rcond=None)
        coef[k] = c
        res[k] = np.abs(d - A @ c).mean()
    return coef, res


def to_strength(coef: np.ndarray, alpha: float = ALPHA, max_strength: float = MAX_STRENGTH) -> np.ndarray:
    """Blend coefficients -> mark strengths (1 = full target strength), clipped to [0, max_strength]."""
    return np.clip(coef / alpha, 0, max_strength)
