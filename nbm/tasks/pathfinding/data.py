"""Build the pathfinding data tables (deterministic: a rebuild is identical, row by row).

    python -m nbm.tasks.pathfinding.data --costs uniform --out $NBM_DATA/shortest_path [--workers 16]
    python -m nbm.tasks.pathfinding.data --costs terrain --out $NBM_DATA/terrain

Uniform costs (floor / wall; families blocks / walls / rooms, + the held-out maze), BFS distance d:
  train          200,000  N 5..10, 2 <= d <= 20
  val                600  same distribution
  test_id          1,000  same distribution (Table 2 "held-out maps")
  test_ood_size      500  N 11..12, larger than any training map ("larger maps")
  test_ood_long      500  N 9..10, d >= 21, longer than any training route ("longer paths")
  test_ood_maze      500  N 5..10, the unseen maze family (appendix)
Terrain (floor 1 / mud 2 / sand 4 / wall; the three families with terrain patches; target route length <= 20):
  train 200,000, val 600, test 1,000 (N 5..10), probe 500 (N 10; 250 "cross" + 250 "detour", maps.probe_map).

Row i of a split -> (n, family) = the i-th element of the cycle over (n, family) pairs (every prefix is balanced), map seed =
base + i (a base per split); if the map fails the split's constraints the seed moves on to seed + 10^7 j. The evaluation splits
are built first, in the order of SPLITS, and are pairwise disjoint; no training map equals an evaluation map: a (grid, start, goal)
already used is skipped (next seed). Columns: id, split, n, family, terrain, seed, grid (row-major digits), sr, sc, gr, gc (start /
goal), d (BFS distance), cost, len, turns, moves (the target route as R / D / L / U letters), uniform_moves / uniform_cost (terrain:
the step-shortest route and its true cost), label (probe), seg_di (the clip of the target route used for video training),
seg_walk (not used by these models), row_seed.
Writes <out>/<split>.parquet and <out>/build.json (a hash per table).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from multiprocessing import Pool

import pandas as pd

from nbm.progress import Progress
from nbm.tasks.pathfinding import maps
from nbm.tasks.pathfinding.render import n_segments

DMAX = 20
NS_ID = tuple(range(5, 11))
WALK_MOVES = 16  # moves per clip of a walking-agent clip format ; only for the seg_walk column
# costs -> split -> (rows, map sizes, families, terrain, dmin, dmax, seed base); evaluation splits first (build order)
SPLITS = {
    "uniform": {
        "val": (600, NS_ID, maps.FAMILIES, False, 2, DMAX, 20_000_000),
        "test_id": (1000, NS_ID, maps.FAMILIES, False, 2, DMAX, 30_000_000),
        "test_ood_size": (500, (11, 12), maps.FAMILIES, False, 2, DMAX, 40_000_000),
        "test_ood_long": (500, (9, 10), maps.FAMILIES, False, DMAX + 1, None, 50_000_000),
        "test_ood_maze": (500, NS_ID, (maps.OOD_FAMILY,), False, 2, DMAX, 60_000_000),
        "train": (200_000, NS_ID, maps.FAMILIES, False, 2, DMAX, 10_000_000),
    },
    "terrain": {
        "val": (600, NS_ID, maps.FAMILIES, True, 2, DMAX, 120_000_000),
        "test": (1000, NS_ID, maps.FAMILIES, True, 2, DMAX, 130_000_000),
        "train": (200_000, NS_ID, maps.FAMILIES, True, 2, DMAX, 110_000_000),
    },
}
PROBE_ROWS, PROBE_BASE = 500, 140_000_000

_EXCLUDE: set = set()  # map keys a new row must avoid (set per worker by _init)


def map_key(m: dict) -> str:
    return f"{m['n']}|{maps.grid_str(m['grid'])}|{m['start']}|{m['goal']}"


def row_of_map(m: dict, info: dict, split: str, i: int, seed: int) -> dict:
    rid = f"sp_{'t' if m['terrain'] else 'u'}_{split}_{i:06d}"
    rs = random.Random(f"seg:{rid}")
    L = info["len"]
    return {"id": rid, "split": split, "n": m["n"], "family": m["family"], "terrain": bool(m["terrain"]), "seed": int(seed),
            "grid": maps.grid_str(m["grid"]), "sr": m["start"][0], "sc": m["start"][1], "gr": m["goal"][0], "gc": m["goal"][1],
            "d": info["d"], "cost": info["cost"], "len": L, "turns": info["turns"], "moves": info["moves"],
            "uniform_moves": info.get("uniform_moves", info["moves"]), "uniform_cost": info.get("uniform_cost", info["cost"]),
            "label": m.get("label", ""), "seg_di": rs.randrange(n_segments(L)), "seg_walk": rs.randrange(max(1, -(-L // WALK_MOVES))),
            "row_seed": rs.randrange(2 ** 31)}


def make_row(args: tuple) -> dict:
    """Row i of a split: the first seed base + i + 10^7 j whose map meets the split's constraints and is not excluded."""
    costs, split, i = args
    _, ns, fams, terr, dmin, dmax, base = SPLITS[costs][split]
    cells = [(n, f) for n in ns for f in fams]
    n, fam = cells[i % len(cells)]
    for j in range(100):
        seed = base + i + j * 10_000_000
        try:
            m = maps.make_map(n, fam, seed, terrain=terr, dmin=dmin, dmax=dmax)
        except RuntimeError:
            continue
        info = maps.map_info(m)
        if terr and info["len"] > DMAX:
            continue
        if map_key(m) in _EXCLUDE:
            continue
        return row_of_map(m, info, split, i, seed)
    raise RuntimeError(f"{costs} {split} row {i}: no map")


def _init(exclude: set) -> None:
    global _EXCLUDE
    _EXCLUDE = exclude


def probe_rows() -> list[dict]:
    rows, count, j = [], {"cross": 0, "detour": 0}, 0
    while len(rows) < PROBE_ROWS:
        m, label = maps.probe_map(PROBE_BASE + j)
        j += 1
        if count[label] >= PROBE_ROWS // 2:
            continue
        count[label] += 1
        rows.append(row_of_map(m, maps.map_info(m), "probe", len(rows), PROBE_BASE + j - 1))
    return rows


def table_hash(df: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def build_split(costs: str, split: str, keys: set, workers: int, limit: int | None = None) -> pd.DataFrame:
    """One split given the map keys of the splits built before it. Evaluation splits are built sequentially (each row also
    excludes the earlier rows of its split); the training split in parallel against the evaluation keys. limit: only the first
    rows (a quick check; a prefix of the full split)."""
    n_rows = SPLITS[costs][split][0] if limit is None else min(limit, SPLITS[costs][split][0])
    if split == "train":
        with Pool(workers, initializer=_init, initargs=(keys,)) as pool:
            prog = Progress(total=n_rows, desc=f"{costs} train rows", unit="row")
            rows = []
            for r in pool.imap(make_row, [(costs, split, i) for i in range(n_rows)], chunksize=256):
                rows.append(r)
                prog.update(1)
            prog.close()
    else:
        _init(set(keys))
        rows = []
        for i in range(n_rows):
            rows.append(make_row((costs, split, i)))
            _EXCLUDE.add(map_key(row_map(rows[-1])))
    return pd.DataFrame(rows)


def row_map(row: dict) -> dict:
    n = int(row["n"])
    return {"n": n, "grid": maps.grid_from_str(row["grid"], n), "start": (int(row["sr"]), int(row["sc"])),
            "goal": (int(row["gr"]), int(row["gc"]))}


def build(costs: str, out: str, workers: int = 16, splits: list[str] | None = None) -> dict:
    """Build every split of one cost model into <out> (or only `splits`; the evaluation splits they depend on are still built,
    in memory)."""
    os.makedirs(out, exist_ok=True)
    keys: set = set()
    tables = {}
    order = [s for s in SPLITS[costs] if s != "train"] + ["train"]
    want = set(order + (["probe"] if costs == "terrain" else [])) if splits is None else set(splits)
    for split in order:
        if split == "train" and "train" not in want:
            continue
        tables[split] = build_split(costs, split, keys, workers)
        if split != "train":
            keys |= {map_key(row_map(r)) for r in tables[split].to_dict("records")}
    if costs == "terrain" and "probe" in want:
        tables["probe"] = pd.DataFrame(probe_rows())
    hashes = {}
    for split, df in tables.items():
        if split not in want:
            continue
        df.to_parquet(os.path.join(out, f"{split}.parquet"), index=False)
        hashes[f"{costs}/{split}"] = table_hash(df)
        print(f"{costs}/{split}: {len(df)} rows | d mean {df.d.mean():.1f} max {df.d.max()} | len mean {df['len'].mean():.1f} | "
              f"N {sorted(df.n.unique().tolist())} | hash {hashes[f'{costs}/{split}'][:12]}", flush=True)
    with open(os.path.join(out, "build.json"), "w") as fh:
        json.dump(hashes, fh, indent=2)
    return hashes


def main() -> None:
    p = argparse.ArgumentParser(description="Build the pathfinding data tables (module docstring).")
    p.add_argument("--costs", required=True, choices=sorted(SPLITS))
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--splits", nargs="*", default=None, help="only these splits (default: all)")
    a = p.parse_args()
    build(a.costs, a.out, a.workers, a.splits)


if __name__ == "__main__":
    main()
