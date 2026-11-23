"""N-queens pictures and the readout of a generated picture.

Picture: 384 x 384 px, a fixed canvas of 12 x 12 cells of 32 px (2 x 2 latent tokens of 16 px per cell). The N x N board sits
in the top-left corner (rows 0..N-1 from the top, columns 0..N-1 from the left) as a chess-style checkerboard ((r + c) even =
light); every other canvas cell is off-board, a uniform dark grey. A queen is a crown glyph (black body, light outline). The
training target is the same picture with the placed queen drawn as the same glyph with a red body (the action mark).

Readout (no learned part, no strength threshold): for every one of the 144 canvas cells, the least-squares strength alpha of
the red glyph, computed on the glyph's fully covered interior pixels (their mean colour) against the input picture:
    alpha = <m - ref, P - ref> / |P - ref|^2,
m = mean colour of the generated picture there, ref = that of the input, P = that of a red glyph on the cell's own base
(light / dark / off-board square, or an existing black queen). The decision is the cell with the largest alpha, at any
strength; it may be off-board or on a queen (the readout does not know the board or the rules, the scorer does).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from nbm import readout
from nbm.tasks.nqueens import rules as R

SIZE = 384
CELL = 32
GRID = 12  # canvas cells per side
MAX_N = GRID

LIGHT = np.array([240, 217, 181], np.float32)
DARK = np.array([181, 136, 99], np.float32)
OFF = np.array([62, 62, 68], np.float32)
QUEEN_BODY = np.array([24, 24, 28], np.float32)
QUEEN_LINE = np.array([236, 236, 232], np.float32)
RED = np.array([224, 32, 36], np.float32)


def _glyph_masks() -> tuple[np.ndarray, np.ndarray]:
    """(body, outline) coverage masks (32, 32) of the queen glyph, drawn at 4x and box-downsampled."""
    big = 4 * CELL
    im = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(im)
    poly = [(30, 98), (18, 44), (36, 72), (40, 30), (53, 64), (64, 22), (75, 64), (88, 30), (92, 72), (110, 44), (98, 98)]
    d.polygon(poly, fill=255)
    for x, y in ((18, 44), (40, 30), (64, 22), (88, 30), (110, 44)):
        d.ellipse((x - 8, y - 8, x + 8, y + 8), fill=255)
    d.rounded_rectangle((24, 96, 104, 114), radius=5, fill=255)
    grown = im.filter(ImageFilter.MaxFilter(13))  # about a 1.5 px outline at 32 px
    body = np.asarray(im.resize((CELL, CELL), Image.BOX), np.float32) / 255.0
    outer = np.asarray(grown.resize((CELL, CELL), Image.BOX), np.float32) / 255.0
    return body, np.clip(outer - body, 0, 1)


BODY, OUTLINE = _glyph_masks()
READ = BODY >= 0.999  # readout region: the glyph's fully covered interior pixels


def _kind(r: int, c: int, n: int) -> str:
    """Base of a canvas cell: L(ight) / D(ark) board square or O(ff-board)."""
    if r >= n or c >= n:
        return "O"
    return "L" if (r + c) % 2 == 0 else "D"


@lru_cache(maxsize=None)
def tile(kind: str, glyph: str | None) -> np.ndarray:
    """float32 tile (32, 32, 3): base kind "L" / "D" / "O", glyph None / "black" / "red"."""
    base = {"L": LIGHT, "D": DARK, "O": OFF}[kind]
    t = np.broadcast_to(base, (CELL, CELL, 3)).astype(np.float32).copy()
    if glyph is None:
        return t
    body = QUEEN_BODY if glyph == "black" else RED
    t = t * (1 - OUTLINE[..., None]) + QUEEN_LINE * OUTLINE[..., None]
    t = t * (1 - BODY[..., None]) + body * BODY[..., None]
    return t


def render(s: str, mark: R.Cell | None = None) -> np.ndarray:
    """uint8 (384, 384, 3) picture of state s; mark = (r, c): the placed queen drawn red (any canvas cell; on a queen's cell
    the red glyph replaces the black one)."""
    n = len(s)
    cols = R.cols_of(s)
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    for r in range(GRID):
        for c in range(GRID):
            queen = r < n and cols[r] == c
            glyph = "red" if mark is not None and (r, c) == tuple(mark) else "black" if queen else None
            img[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = tile(_kind(r, c, n), glyph)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def render_pil(s: str, mark: R.Cell | None = None) -> Image.Image:
    return Image.fromarray(render(s, mark))


def cell_box(r: int, c: int) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of canvas cell (r, c)."""
    return (c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL)


# ------------------------------------------------------------------ readout
def to_array(img: Image.Image | np.ndarray) -> np.ndarray:
    """uint8 (384, 384, 3); a picture of another size is resized (bilinear)."""
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        if img.size != (SIZE, SIZE):
            img = img.resize((SIZE, SIZE), Image.BILINEAR)
        return np.asarray(img)
    a = np.asarray(img)
    if a.shape[:2] != (SIZE, SIZE):
        a = np.asarray(Image.fromarray(a).resize((SIZE, SIZE), Image.BILINEAR))
    return a


def _region_means(a: np.ndarray) -> np.ndarray:
    """(12, 12, 3) mean colour over the glyph interior of every canvas cell of a (384, 384, 3) array."""
    x = a.astype(np.float32).reshape(GRID, CELL, GRID, CELL, 3).transpose(0, 2, 1, 3, 4)  # (G, G, 32, 32, 3)
    return x[:, :, READ].mean(2)


@lru_cache(maxsize=64)
def _reference(s: str) -> tuple[np.ndarray, np.ndarray]:
    """(ref, proto), each (12, 12, 3): interior mean colour of the clean input and of a red glyph on each cell's own base."""
    n = len(s)
    cols = R.cols_of(s)
    ref = np.zeros((GRID, GRID, 3), np.float32)
    proto = np.zeros((GRID, GRID, 3), np.float32)
    for r in range(GRID):
        for c in range(GRID):
            queen = r < n and cols[r] == c
            ref[r, c] = tile(_kind(r, c, n), "black" if queen else None)[READ].mean(0)
            proto[r, c] = tile(_kind(r, c, n), "red")[READ].mean(0)
    return ref, proto


def mark_strength(img: Image.Image | np.ndarray, s: str) -> np.ndarray:
    """(12, 12) least-squares strength of the red glyph in every canvas cell (0 = like the input, 1 = a full red glyph)."""
    a = to_array(img)
    ref, proto = _reference(s)
    m = _region_means(a)
    d = proto - ref
    return ((m - ref) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-6)


def decode(img: Image.Image | np.ndarray, s: str) -> dict:
    """Threshold-free readout: the canvas cell (r, c) with the strongest red glyph (first in row-major order on ties), its
    strength alpha, the second-largest strength, and copy_err = mean |pixel difference| to the input outside that cell."""
    a = to_array(img)
    flat = mark_strength(a, s).reshape(-1)
    order = readout.ranked(flat)
    k = order[0]
    r, c = divmod(k, GRID)
    diff = np.abs(a.astype(np.float32) - render(s).astype(np.float32)).mean(-1)
    x0, y0, x1, y1 = cell_box(r, c)
    outside = np.ones_like(diff, bool)
    outside[y0:y1, x0:x1] = False
    return {"cell": (r, c), "alpha": float(flat[k]), "second": float(flat[order[1]]), "copy_err": float(diff[outside].mean())}
