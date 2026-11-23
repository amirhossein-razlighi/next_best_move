"""Pictures of a pathfinding state, the training targets built from them, and the fixed cell decoder that reads them back.

Canvas: S = 384 px, CELL = 32 px per cell (= one Wan2.2 latent token = 2 x 2 Qwen-Image tokens); an N x N map sits at the
token-aligned offset ((12 - N) // 2) * CELL on a black background. Each cell has a 1 px grey grid line on each side and a 30 x 30
inner area: floor light grey, wall dark grey, mud brown with darker stripes, sand yellow with darker dots (the costs are never
drawn). A state shows the agent (BLUE disc, radius 8), the visited cells (TRAIL: inner area GREEN) and the goal (RED ring, 4 px,
on floor); a DECIDE frame adds a RED disc (radius 8) on the next cell. The colours are the paper's (nbm.marks).

Training pictures:
  image model  condition = a state along the target route, target = the same + the red disc on the next cell (= DECIDE frame).
  video model  a clip of 17 blocks (segment_blocks): block 0 = the state after 8 k moves, then up to 8 DECIDE / IMAGINE pairs
               (the red disc on the next cell / the agent moved there, its old cell green), then the final state held.

Decoder (decode_cells): per cell, the mean colour of three fixed regions - CENTRE (radius 5 around the cell centre), RING band
(2-3 px inside the inner edge) and BODY (four 5 x 5 corner squares) - is matched to the nearest clean prototype of that region:
a red ring = the goal cell (base floor); else base = the nearest BODY prototype among floor / wall / mud / sand / trail; centre =
the nearest of {the base's own centre, BLUE, RED} = none / agent / mark. The threshold-free readout of a move is the cell with the
strongest red disc (red_strength: least-squares strength of the red disc against the cell's unmarked centre), at any strength.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

from nbm import marks
from nbm.tasks.pathfinding.maps import FLOOR, MAXN, MUD, SAND, TRAIL, WALL, Cell

S, CELL = 384, 32
FLOOR_C, WALL_C = (232, 232, 232), (56, 56, 56)
MUD_C, MUD_T = (150, 105, 60), (112, 76, 40)  # base / stripe colour
SAND_C, SAND_T = (230, 202, 130), (200, 165, 85)  # base / dot colour
GRID_C, BG_C = (160, 160, 160), (0, 0, 0)
BLUE, RED, GREEN = marks.BLUE, marks.RED, marks.GREEN
RING_W, DISC_R = 4, 8
N_BLOCKS, MOVES_PER_CLIP = 17, 8
BASES = (FLOOR, WALL, MUD, SAND, TRAIL)


# ------------------------------------------------------------------------------------------------ geometry
def offset(n: int) -> int:
    return ((MAXN - n) // 2) * CELL


def cell_box(r: int, c: int, n: int) -> marks.Box:
    """Pixel box (x0, y0, x1, y1) of cell (r, c) of an n x n map."""
    o = offset(n)
    return o + c * CELL, o + r * CELL, o + (c + 1) * CELL, o + (r + 1) * CELL


def _masks() -> dict[str, np.ndarray]:
    y, x = np.mgrid[0:CELL, 0:CELL]
    inner = (y >= 1) & (y <= CELL - 2) & (x >= 1) & (x <= CELL - 2)
    edge = np.minimum(np.minimum(y - 1, CELL - 2 - y), np.minimum(x - 1, CELL - 2 - x))  # distance to the inner edge
    cy = cx = (CELL - 1) / 2
    rr = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    corner = np.zeros_like(inner)
    for a in (slice(2, 7), slice(CELL - 7, CELL - 2)):
        for b in (slice(2, 7), slice(CELL - 7, CELL - 2)):
            corner[a, b] = True
    return {"inner": inner, "ring": inner & (edge < RING_W), "disc": rr <= DISC_R, "centre": rr <= 5,
            "band": inner & (edge >= 1) & (edge <= 2), "body": corner}


MASK = _masks()


def _base_patch(t: int) -> np.ndarray:
    p = np.empty((CELL, CELL, 3), np.uint8)
    p[:] = GRID_C
    y, x = np.mgrid[0:CELL, 0:CELL]
    inner = MASK["inner"]
    if t == FLOOR:
        p[inner] = FLOOR_C
    elif t == WALL:
        p[inner] = WALL_C
    elif t == MUD:
        p[inner] = MUD_C
        p[inner & ((y - 1) % 6 >= 4)] = MUD_T  # horizontal stripes
    elif t == SAND:
        p[inner] = SAND_C
        p[inner & ((y - 1) % 6 >= 3) & ((y - 1) % 6 <= 4) & ((x - 1) % 6 >= 3) & ((x - 1) % 6 <= 4)] = SAND_T  # dots
    elif t == TRAIL:
        p[inner] = GREEN
    return p


PATCH = {t: _base_patch(t) for t in BASES}


# ------------------------------------------------------------------------------------------------ rendering
def render(m: dict, pos: Cell | None, visited: Sequence[Cell] = (), mark: Cell | None = None) -> np.ndarray:
    """uint8 (S, S, 3) picture of a state: agent at pos (None = no agent), visited cells green, the goal ring, optional red disc."""
    n, g = m["n"], m["grid"]
    a = np.empty((S, S, 3), np.uint8)
    a[:] = BG_C
    vis = set(map(tuple, visited))
    goal = tuple(m["goal"])
    for r in range(n):
        for c in range(n):
            p = PATCH[TRAIL if (r, c) in vis else int(g[r, c])].copy()
            if (r, c) == goal:
                p = PATCH[FLOOR].copy()
                p[MASK["ring"]] = RED
            if pos is not None and (r, c) == tuple(pos):
                p[MASK["disc"]] = BLUE
            if mark is not None and (r, c) == tuple(mark):
                p[MASK["disc"]] = RED
            x0, y0, _, _ = cell_box(r, c, n)
            a[y0:y0 + CELL, x0:x0 + CELL] = p
    return a


def state_picture(m: dict, path: Sequence[Cell], k: int) -> np.ndarray:
    """The state after k moves along path: the agent at path[k], the cells before it green."""
    return render(m, path[k], path[:k])


def decide_picture(m: dict, path: Sequence[Cell], k: int) -> np.ndarray:
    """The state after k moves + the red disc on the next cell path[k + 1] (the image model's target, the video DECIDE block)."""
    return render(m, path[k], path[:k], mark=path[k + 1])


def segment_blocks(m: dict, path: Sequence[Cell], seg: int) -> np.ndarray:
    """uint8 (17, S, S, 3) clip of segment `seg` of the solve along path: block 0 = the state after 8 seg moves, then a DECIDE /
    IMAGINE pair per move (at most 8 moves, fewer if the goal comes first), then the final state held."""
    L = len(path) - 1
    k0 = MOVES_PER_CLIP * seg
    assert 0 <= k0 <= L and (seg == 0 or k0 < L), (seg, L)
    blocks = [state_picture(m, path, k0)]
    k = k0
    while len(blocks) < N_BLOCKS:
        if k < L and k < k0 + MOVES_PER_CLIP:
            blocks.append(decide_picture(m, path, k))
            k += 1
            blocks.append(state_picture(m, path, k))
        else:
            blocks.append(state_picture(m, path, min(k, L)))
    return np.stack(blocks)


def n_segments(L: int) -> int:
    """Clips needed for a route of L moves."""
    return max(1, -(-L // MOVES_PER_CLIP))


# ------------------------------------------------------------------------------------------------ decoder
def _prototypes() -> dict:
    out = {}
    for t in BASES:
        p = PATCH[t].astype(np.float32)
        for region in ("body", "band", "centre"):
            out[(region, t)] = p[MASK[region]].mean(0)
    for name, col in (("blue", BLUE), ("red", RED)):
        out[("centre", name)] = np.array(col, np.float32)
        out[("band", name)] = np.array(col, np.float32)
    return out


PROTO = _prototypes()


def as_frame(frame: Image.Image | np.ndarray) -> np.ndarray:
    """uint8 (S, S, 3); a picture of another size is resized (bilinear) first."""
    a = np.asarray(frame.convert("RGB") if isinstance(frame, Image.Image) else frame)
    if a.shape[:2] != (S, S):
        a = np.asarray(Image.fromarray(a).resize((S, S), Image.BILINEAR))
    return a


def _region_means(a: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """(n, n, 3) mean colours of the body / band / centre regions of every cell."""
    o = offset(n)
    cells = a[o:o + n * CELL, o:o + n * CELL].astype(np.float32).reshape(n, CELL, n, CELL, 3).transpose(0, 2, 1, 3, 4)
    return {k: cells[:, :, MASK[k]].mean(2) for k in ("body", "band", "centre")}


def decode_cells(frame: Image.Image | np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Decode every cell. Returns (n, n) arrays: base (FLOOR / WALL / MUD / SAND / TRAIL; the goal cell -> FLOOR), goal (red ring),
    centre (0 none, 1 blue agent, 2 red mark), blue / red (distance margins: > 0 = nearer the disc colour than the plain cell)."""
    R = _region_means(as_frame(frame), n)
    band_d = np.stack([np.linalg.norm(R["band"] - PROTO[("band", t)], axis=-1) for t in BASES], -1)
    goal = np.linalg.norm(R["band"] - PROTO[("band", "red")], axis=-1) < band_d.min(-1)
    body_d = np.stack([np.linalg.norm(R["body"] - PROTO[("body", t)], axis=-1) for t in BASES], -1)
    base = np.where(goal, FLOOR, np.array(BASES)[body_d.argmin(-1)])
    own = np.stack([PROTO[("centre", int(t))] for t in base.reshape(-1)]).reshape(n, n, 3)
    d_none = np.linalg.norm(R["centre"] - own, axis=-1)
    d_blue = np.linalg.norm(R["centre"] - PROTO[("centre", "blue")], axis=-1)
    d_red = np.linalg.norm(R["centre"] - PROTO[("centre", "red")], axis=-1)
    centre = np.stack([d_none, d_blue, d_red], -1).argmin(-1)
    return {"base": base.astype(np.int8), "goal": goal, "centre": centre.astype(np.int8), "blue": d_none - d_blue,
            "red": d_none - d_red}


def expected_cells(m: dict, pos: Cell | None, visited: Sequence[Cell] = (), mark: Cell | None = None):
    """(base, goal, centre) that decode_cells returns for a clean picture of the state."""
    n, g = m["n"], m["grid"]
    base = np.array(g, np.int8).copy()
    for p in visited:
        base[tuple(p)] = TRAIL
    goal = np.zeros((n, n), bool)
    goal[tuple(m["goal"])] = True
    base[tuple(m["goal"])] = FLOOR
    centre = np.zeros((n, n), np.int8)
    if pos is not None:
        centre[tuple(pos)] = 1
    if mark is not None:
        centre[tuple(mark)] = 2
    return base, goal, centre


def mismatches(dec: dict, m: dict, pos: Cell | None, visited: Sequence[Cell] = (), mark: Cell | None = None,
               skip: Sequence[Cell] = ()) -> int:
    """Number of cells whose decode differs from the clean picture of the state (cells in `skip` ignored)."""
    b, g, c = expected_cells(m, pos, visited, mark)
    bad = (dec["base"] != b) | (dec["goal"] != g) | (dec["centre"] != c)
    for p in skip:
        bad[tuple(p)] = False
    return int(bad.sum())


def _strongest_margin(score: np.ndarray, centre: np.ndarray, code: int) -> tuple[Cell, bool, int]:
    k = int(np.argmax(score))
    n = score.shape[0]
    return divmod(k, n), bool(centre.reshape(-1)[k] == code), int((centre == code).sum())


def agent_cell(dec: dict) -> tuple[Cell, bool, int]:
    """(the cell with the largest blue margin, whether it decodes as the agent, the number of agent cells)."""
    return _strongest_margin(dec["blue"], dec["centre"], 1)


def red_disc_cell(dec: dict) -> tuple[Cell, bool, int]:
    """(the cell with the largest red margin, whether it decodes as a red disc, the number of red-disc cells). Used for the
    validity bookkeeping (an edit check, IMAGINE consistency), never to choose the move."""
    return _strongest_margin(dec["red"], dec["centre"], 2)


def red_strength(frame: Image.Image | np.ndarray, dec: dict, pos: Cell | None) -> np.ndarray:
    """(n, n) least-squares strength of a red disc in each cell centre against the cell's unmarked centre (the prototype of its
    decoded base; the agent cell `pos` against the blue disc): <c - own, RED - own> / |RED - own|^2 (0 = unmarked, 1 = a full disc)."""
    n = dec["base"].shape[0]
    c = _region_means(as_frame(frame), n)["centre"]
    own = np.stack([PROTO[("centre", int(t))] for t in dec["base"].reshape(-1)]).reshape(n, n, 3)
    if pos is not None:
        own[tuple(pos)] = PROTO[("centre", "blue")]
    d = PROTO[("centre", "red")][None, None] - own
    return ((c - own) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-6)


def read_move(frame: Image.Image | np.ndarray, dec: dict, pos: Cell) -> tuple[Cell, float]:
    """The threshold-free readout of a move: the cell with the strongest red disc (red_strength, first on ties), at any strength.
    Returns (cell, strength)."""
    al = red_strength(frame, dec, pos)
    k = int(np.argmax(al))
    return divmod(k, al.shape[0]), float(al.reshape(-1)[k])
