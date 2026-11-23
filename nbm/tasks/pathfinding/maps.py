"""Grid maps of the pathfinding tasks: cell types, map generators, shortest / cheapest paths and route checks.

A map is an N x N grid (N <= 12) of cells: FLOOR (cost 1), WALL (impassable) and, on terrain maps only, MUD (cost 2) and SAND
(cost 4). A move is a step to one of the 4 neighbours; its cost is the cost of the cell entered. A map is a plain dict
{"n", "family", "grid" (int8 (n, n)), "start" (r, c), "goal" (r, c), "terrain" (bool)}; uniform maps have terrain False and
every step costs 1 (so the cheapest route is a shortest one).

Layout families: "blocks" (random 1x1 .. 3x1 wall blocks, density 15-32 %), "walls" (full-length walls with one or two one-cell
gaps + a few single walls), "rooms" (recursive division with doors) and the held-out "maze" (depth-first maze on the even lattice,
braided so that several routes exist). Terrain maps use the three training families with fewer walls and mud / sand patches on
25-45 % of the free cells; start and goal are always floor.

Every generator draws from one random.Random seeded with a string of (n, family, seed, terrain), so a map is a pure function of
those four values (the data builder, nbm/tasks/pathfinding/data.py, relies on this).
"""
from __future__ import annotations

import heapq
import random
from collections import deque
from typing import Iterator, Sequence

import numpy as np

MAXN = 12
FLOOR, WALL, MUD, SAND, TRAIL = 0, 1, 2, 3, 4  # TRAIL = a visited cell (drawn green; only in pictures, never in a grid)
COST = {FLOOR: 1, MUD: 2, SAND: 4}  # extra costs 1 / 3 per cell vs detours of 2 per side step: few exact ties
INF = 10 ** 9
FAMILIES = ("blocks", "walls", "rooms")
OOD_FAMILY = "maze"
DIRS = ((0, 1, "R"), (1, 0, "D"), (0, -1, "L"), (-1, 0, "U"))  # also the tie-break order of the training target
DIR_OF = {(dr, dc): k for k, (dr, dc, _) in enumerate(DIRS)}

Cell = tuple[int, int]


def neighbours(n: int, r: int, c: int) -> Iterator[Cell]:
    """The on-map 4-neighbours of (r, c), in the order R, D, L, U."""
    for dr, dc, _ in DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            yield rr, cc


# ------------------------------------------------------------------------------------------------ generators
def gen_blocks(n: int, rng: random.Random, density: float | None = None) -> np.ndarray:
    g = np.zeros((n, n), np.int8)
    target = (rng.uniform(0.15, 0.32) if density is None else density) * n * n
    while g.sum() < target:
        h, w = rng.choice([(1, 1), (1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)])
        r, c = rng.randrange(n), rng.randrange(n)
        g[r:r + h, c:c + w] = WALL
    return g


def gen_walls(n: int, rng: random.Random) -> np.ndarray:
    g = np.zeros((n, n), np.int8)
    k = rng.randint(1, max(1, (n - 1) // 3))
    used: dict[int, set] = {0: set(), 1: set()}
    for _ in range(k):
        o = rng.randrange(2)  # 0 = a horizontal wall (a row), 1 = a vertical one (a column)
        cand = [p for p in range(1, n - 1) if all(abs(p - q) >= 2 for q in used[o])]
        if not cand:
            continue
        p = rng.choice(cand)
        used[o].add(p)
        line = np.ones(n, bool)
        for _ in range(rng.randint(1, 2)):
            line[rng.randrange(n)] = False
        if o == 0:
            g[p, line] = WALL
        else:
            g[line, p] = WALL
    for _ in range(int(rng.uniform(0.0, 0.06) * n * n)):
        g[rng.randrange(n), rng.randrange(n)] = WALL
    return g


def gen_rooms(n: int, rng: random.Random) -> np.ndarray:
    g = np.zeros((n, n), np.int8)

    def split(r0: int, c0: int, r1: int, c1: int, depth: int) -> None:  # inclusive region
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if max(h, w) < 5 or depth > 3 or (depth >= 1 and rng.random() < 0.15):
            return
        horiz = h > w if h != w else rng.random() < 0.5
        if horiz:
            p = rng.randint(r0 + 2, r1 - 2)
            g[p, c0:c1 + 1] = WALL
            for _ in range(1 if rng.random() < 0.7 else 2):
                g[p, rng.randint(c0, c1)] = FLOOR
            split(r0, c0, p - 1, c1, depth + 1)
            split(p + 1, c0, r1, c1, depth + 1)
        else:
            p = rng.randint(c0 + 2, c1 - 2)
            g[r0:r1 + 1, p] = WALL
            for _ in range(1 if rng.random() < 0.7 else 2):
                g[rng.randint(r0, r1), p] = FLOOR
            split(r0, c0, r1, p - 1, depth + 1)
            split(r0, p + 1, r1, c1, depth + 1)

    split(0, 0, n - 1, n - 1, 0)
    return g


def gen_maze(n: int, rng: random.Random, braid: float = 0.15) -> np.ndarray:
    g = np.ones((n, n), np.int8)
    o_r = rng.randrange(2) if n % 2 == 0 else 0
    o_c = rng.randrange(2) if n % 2 == 0 else 0
    lattice = [(r, c) for r in range(o_r, n, 2) for c in range(o_c, n, 2)]
    lset = set(lattice)
    start = rng.choice(lattice)
    g[start] = FLOOR
    stack, seen = [start], {start}
    while stack:
        r, c = stack[-1]
        nb = [(r + 2 * dr, c + 2 * dc) for dr, dc, _ in DIRS
              if (r + 2 * dr, c + 2 * dc) in lset and (r + 2 * dr, c + 2 * dc) not in seen]
        if not nb:
            stack.pop()
            continue
        rr, cc = rng.choice(nb)
        g[(r + rr) // 2, (c + cc) // 2] = FLOOR
        g[rr, cc] = FLOOR
        seen.add((rr, cc))
        stack.append((rr, cc))
    for r in range(n):  # braid: open walls between two corridors (loops -> several routes)
        for c in range(n):
            if g[r, c] == WALL and rng.random() < braid:
                h = 0 < c < n - 1 and g[r, c - 1] == FLOOR and g[r, c + 1] == FLOOR
                v = 0 < r < n - 1 and g[r - 1, c] == FLOOR and g[r + 1, c] == FLOOR
                if h != v:
                    g[r, c] = FLOOR
    return g


def add_terrain(g: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mud / sand patches (random rectangles and blobs) over floor cells until 25-45 % of the free cells are terrain."""
    n = len(g)
    g = g.copy()
    free = int((g != WALL).sum())
    target = rng.uniform(0.25, 0.45) * free
    covered, tries = 0, 0
    while covered < target and tries < 200:
        tries += 1
        t = MUD if rng.random() < 0.5 else SAND
        if rng.random() < 0.5:  # rectangle
            h, w = rng.randint(1, max(1, n // 3)), rng.randint(1, max(2, n // 2))
            if rng.random() < 0.5:
                h, w = w, h
            r, c = rng.randrange(n), rng.randrange(n)
            cells = [(rr, cc) for rr in range(r, min(n, r + h)) for cc in range(c, min(n, c + w))]
        else:  # blob: random growth
            r, c = rng.randrange(n), rng.randrange(n)
            blob, size = {(r, c)}, rng.randint(3, max(4, n))
            while len(blob) < size:
                rr, cc = rng.choice(sorted(blob))
                blob.add(rng.choice(list(neighbours(n, rr, cc))))
            cells = sorted(blob)
        for rr, cc in cells:
            if g[rr, cc] == FLOOR:
                g[rr, cc] = t
                covered += 1
    return g


def make_map(n: int, family: str, seed: int, terrain: bool = False, dmin: int = 2, dmax: int | None = None,
             tries: int = 200) -> dict:
    """The map of (n, family, seed, terrain): start / goal free floor cells with dmin <= BFS distance <= dmax. The distance is
    drawn uniformly over the available distance levels (so long routes are not rare), then a (start, goal) pair of that level.
    Raises RuntimeError if no layout of `tries` has such a pair."""
    rng = random.Random(f"sp:{n}:{family}:{seed}:{int(terrain)}")
    for _ in range(tries):
        if family == "blocks":
            g = gen_blocks(n, rng, rng.uniform(0.08, 0.2) if terrain else None)
        elif family == "walls":
            g = gen_walls(n, rng)
        elif family == "rooms":
            g = gen_rooms(n, rng)
        elif family == "maze":
            g = gen_maze(n, rng)
        else:
            raise ValueError(family)
        if terrain:
            g = add_terrain(g, rng)
        free = [(r, c) for r in range(n) for c in range(n) if g[r, c] == FLOOR]
        if len(free) < 2:
            continue
        pairs = []  # (start, goal, distance) from 8 random starts, pooled
        for _ in range(8):
            s = rng.choice(free)
            d = bfs(g, s)
            pairs += [(s, p, int(d[p])) for p in free if dmin <= d[p] < INF and (dmax is None or d[p] <= dmax)]
        if pairs:
            level = rng.choice(sorted({x[2] for x in pairs}))
            s, goal, _ = rng.choice([x for x in pairs if x[2] == level])
            return {"n": n, "family": family, "grid": g, "start": s, "goal": goal, "terrain": terrain}
    raise RuntimeError(f"no map for n={n} {family} seed={seed} dmin={dmin} dmax={dmax}")


def probe_map(seed: int, n: int = 10) -> tuple[dict, str]:
    """A terrain probe map: an open room with a sand (or mud) band between start and goal, so that the cheapest route either
    crosses the band ("cross": every cheapest path enters terrain) or goes around it ("detour": no cheapest path does). The band
    is always in the way (avoiding it costs extra steps); ambiguous layouts are redrawn. Returns (map, label)."""
    rng = random.Random(f"sp_probe:{seed}")
    for _ in range(500):
        g = np.zeros((n, n), np.int8)
        for _ in range(rng.randint(0, 3)):  # light clutter
            g[rng.randrange(n), rng.randrange(n)] = WALL
        t = SAND if rng.random() < 0.7 else MUD
        vertical = rng.random() < 0.5
        w = rng.randint(1, 3)
        length = rng.randint(3, n)
        c0 = rng.randint(2, n - 2 - w)
        r0 = rng.randint(0, n - length)
        band = np.zeros((n, n), bool)
        band[r0:r0 + length, c0:c0 + w] = True
        if not vertical:
            band = band.T
        g[band] = t
        sr, gr = rng.randrange(n), rng.randrange(n)
        sc, gc = rng.randint(0, c0 - 1), rng.randint(c0 + w, n - 1)
        s, goal = ((sr, sc), (gr, gc)) if vertical else ((sc, sr), (gc, gr))
        if rng.random() < 0.5:
            s, goal = goal, s
        if g[s] != FLOOR or g[goal] != FLOOR:
            continue
        m = {"n": n, "family": "probe", "grid": g, "start": s, "goal": goal, "terrain": True}
        D = cost_to_go(g, goal, True)
        if D[s] >= INF:
            continue
        terr = (g == MUD) | (g == SAND)
        if bfs(g, s, passable=~terr)[goal] <= bfs(g, s)[goal]:  # avoiding terrain must cost extra steps
            continue
        avoid = cost_to_go(g, goal, True, blocked=terr)
        cs = cost_from(g, s)  # cheapest path forced through a terrain cell x: cost(s -> x) + cost(x -> goal)
        cross = min((int(cs[x]) + int(D[x]) for x in zip(*np.nonzero(terr)) if cs[x] < INF and D[x] < INF), default=INF)
        best = int(D[s])
        if int(avoid[s]) == best and cross > best:
            label = "detour"
        elif int(avoid[s]) > best and cross == best:
            label = "cross"
        else:
            continue
        m["label"] = label
        return m, label
    raise RuntimeError(f"probe seed {seed}: no unambiguous layout")


# ------------------------------------------------------------------------------------------------ distances and targets
def bfs(g: np.ndarray, src: Cell, passable: np.ndarray | None = None) -> np.ndarray:
    """Step distances from src over non-wall cells (INF = unreachable); passable: an extra bool mask of allowed cells."""
    n = len(g)
    d = np.full((n, n), INF, np.int64)
    d[src] = 0
    q = deque([src])
    while q:
        r, c = q.popleft()
        for rr, cc in neighbours(n, r, c):
            if g[rr, cc] != WALL and d[rr, cc] == INF and (passable is None or passable[rr, cc]):
                d[rr, cc] = d[r, c] + 1
                q.append((rr, cc))
    return d


def cell_cost(g: np.ndarray, terrain: bool = True) -> np.ndarray:
    """Cost of entering each cell (INF for walls); terrain=False: every free cell costs 1."""
    c = np.full(g.shape, INF, np.int64)
    for t, v in COST.items():
        c[g == t] = v if terrain else 1
    return c


def cost_to_go(g: np.ndarray, goal: Cell, terrain: bool = True, blocked: np.ndarray | None = None) -> np.ndarray:
    """D[u] = the minimal cost from u to the goal (Dijkstra backwards from the goal). blocked: extra impassable cells (bool)."""
    n = len(g)
    cc = cell_cost(g, terrain)
    if blocked is not None:
        cc = np.where(blocked, INF, cc)
    D = np.full((n, n), INF, np.int64)
    D[goal] = 0
    h = [(0, goal)]
    while h:
        dist, (r, c) = heapq.heappop(h)
        if dist > D[r, c]:
            continue
        for rr, c2 in neighbours(n, r, c):  # predecessor (rr, c2) -> (r, c) costs cc[r, c]
            if g[rr, c2] == WALL or cc[r, c] >= INF or (blocked is not None and blocked[rr, c2]):
                continue
            nd = dist + int(cc[r, c])
            if nd < D[rr, c2]:
                D[rr, c2] = nd
                heapq.heappush(h, (nd, (rr, c2)))
    return D


def cost_from(g: np.ndarray, s: Cell, terrain: bool = True) -> np.ndarray:
    """C[x] = the minimal cost from s to x (s itself 0)."""
    n = len(g)
    cc = cell_cost(g, terrain)
    C = np.full((n, n), INF, np.int64)
    C[s] = 0
    h = [(0, s)]
    while h:
        dist, (r, c) = heapq.heappop(h)
        if dist > C[r, c]:
            continue
        for rr, c2 in neighbours(n, r, c):
            if cc[rr, c2] >= INF:
                continue
            nd = dist + int(cc[rr, c2])
            if nd < C[rr, c2]:
                C[rr, c2] = nd
                heapq.heappush(h, (nd, (rr, c2)))
    return C


def target_path(g: np.ndarray, start: Cell, goal: Cell, terrain: bool = True) -> list[Cell] | None:
    """The training target: a cheapest path with the fewest turns, remaining ties broken by the direction order R, D, L, U at
    each step. Returns the cells start .. goal (None if the goal is unreachable)."""
    n = len(g)
    cc = cell_cost(g, terrain)
    # V[(cell, d_in)] = the minimal (cost, turns) from cell to the goal having arrived with direction d_in (4 = the start)
    V: dict = {}
    h: list = []
    for d_in in range(5):
        V[(goal, d_in)] = (0, 0)
        heapq.heappush(h, ((0, 0), goal, d_in))
    while h:
        val, cell, d_in = heapq.heappop(h)
        if V.get((cell, d_in)) != val or d_in == 4:
            continue
        dr, dc, _ = DIRS[d_in]  # predecessor p = cell - DIRS[d_in], moving in direction d_in, having arrived at p with e
        p = (cell[0] - dr, cell[1] - dc)
        if not (0 <= p[0] < n and 0 <= p[1] < n) or g[p] == WALL or cc[cell] >= INF:
            continue
        for e in range(5):
            nv = (val[0] + int(cc[cell]), val[1] + (1 if e != 4 and e != d_in else 0))
            if nv < V.get((p, e), (INF, INF)):
                V[(p, e)] = nv
                heapq.heappush(h, (nv, p, e))
    if (start, 4) not in V:
        return None
    path, cell, d_in = [start], start, 4
    while cell != goal:
        best, choice = V[(cell, d_in)], None
        for k, (dr, dc, _) in enumerate(DIRS):
            nx = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nx[0] < n and 0 <= nx[1] < n) or g[nx] == WALL or (nx, k) not in V:
                continue
            v2 = V[(nx, k)]
            if (int(cc[nx]) + v2[0], v2[1] + (1 if d_in != 4 and k != d_in else 0)) == best:
                choice = (nx, k)
                break
        assert choice is not None, "tie-break walk lost the optimum"
        cell, d_in = choice
        path.append(cell)
    return path


def path_cost(g: np.ndarray, path: Sequence[Cell], terrain: bool = True) -> int:
    cc = cell_cost(g, terrain)
    return int(sum(int(cc[p]) for p in path[1:]))


def turns(path: Sequence[Cell]) -> int:
    ds = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return int(sum(1 for x, y in zip(ds, ds[1:]) if x != y))


def moves_str(path: Sequence[Cell]) -> str:
    """The moves of a path as letters R / D / L / U."""
    return "".join(DIRS[DIR_OF[(b[0] - a[0], b[1] - a[1])]][2] for a, b in zip(path, path[1:]))


def path_from_moves(start: Sequence[int], moves: str) -> list[Cell]:
    p = [(int(start[0]), int(start[1]))]
    for m in moves:
        dr, dc, _ = DIRS["RDLU".index(m)]
        p.append((p[-1][0] + dr, p[-1][1] + dc))
    return p


def grid_str(g: np.ndarray) -> str:
    """Row-major digits (the data tables' `grid` column)."""
    return "".join(str(int(x)) for x in np.asarray(g).reshape(-1))


def grid_from_str(s: str, n: int) -> np.ndarray:
    return np.array([int(ch) for ch in s], np.int8).reshape(n, n)


def map_info(m: dict) -> dict:
    """Targets / difficulty of a map: BFS distance d, the target path (moves, len, turns) and its cost; terrain maps also the
    step-shortest path (uniform_moves, ignoring the terrain costs) with its true cost, and the mud / sand counts."""
    g, s, t = m["grid"], tuple(m["start"]), tuple(m["goal"])
    terr = bool(m.get("terrain", False))
    p = target_path(g, s, t, terrain=terr)
    info = {"d": int(bfs(g, s)[t]), "cost": path_cost(g, p, terr), "len": len(p) - 1, "turns": turns(p), "moves": moves_str(p)}
    if terr:
        pu = target_path(g, s, t, terrain=False)
        info.update(uniform_moves=moves_str(pu), uniform_cost=path_cost(g, pu, True), n_mud=int((g == MUD).sum()),
                    n_sand=int((g == SAND).sum()))
    return info


# ------------------------------------------------------------------------------------------------ route checks
def is_step(g: np.ndarray, a: Cell, b: Cell) -> bool:
    """b is a 4-neighbour of a, on the map, not a wall."""
    n = len(g)
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and 0 <= b[0] < n and 0 <= b[1] < n and g[b] != WALL


def score_path(m: dict, path: Sequence[Cell], D: np.ndarray) -> dict:
    """Validity / optimality of a route (cells start .. goal): connected 4-neighbour steps, no wall, no revisit; its cost and gap
    to the optimal cost D[start] (D = cost_to_go of the map)."""
    g, s, t = m["grid"], tuple(m["start"]), tuple(m["goal"])
    terr = bool(m.get("terrain", False))
    ok = len(path) >= 1 and tuple(path[0]) == s and tuple(path[-1]) == t and len(set(map(tuple, path))) == len(path)
    ok = ok and all(is_step(g, tuple(a), tuple(b)) for a, b in zip(path, path[1:]))
    out = {"path_valid": bool(ok)}
    if ok:
        c = path_cost(g, path, terr)
        out.update(cost=c, gap=c - int(D[s]), optimal=c == int(D[s]))
    else:
        out.update(cost=None, gap=None, optimal=False)
    return out
