"""Build the N-queens dataset: solution classes -> splits -> tables of partial boards (deterministic, about 30 s on one CPU).

    python -m nbm.tasks.nqueens.data build --out $NBM_DATA/nqueens     # tables + build_stats.json + digest.json + checks.json
    python -m nbm.tasks.nqueens.data check --out $NBM_DATA/nqueens     # re-run the leakage / label checks

Splits are by solution CLASS (a solution together with its 8 images under the symmetries of the square). The classes of N are
ordered by sha256("nqueens_v1|N|<canonical solution>") and the first ones are held out: N = 8: 2 val + 2 test of 12 classes,
N = 9: 8 + 8 of 46, N = 10: 15 + 15 of 92. N = 5, 6, 7 (2, 1 and 6 classes) are all train. N = 11, 12 are never trained on.
A partial board of a solution x is x restricted to a subset of its rows (k queens, 0 <= k < N). Training boards come from
train-class solutions and are never contained in a val / test solution of their N (so no test board, nor any of its
rotations or reflections, is a training board). By construction, training therefore has no empty board and almost no
single-queen boards for N = 8..10.

Label: the topmost empty row and the column of the source solution (the solution the board was drawn from) in that row.
Scoring accepts ANY extendable cell, in any row.

Tables (<out>/<name>.parquet, each shuffled with a fixed seed):
  train      200,071 boards: N = 5..8 every eligible board, N = 9: 80,000, N = 10: 105,000; k balanced ("water-filled":
             an equal share per k up to that k's pool, uniform within a k)
  seen       1,000 train-class boards of N = 8 / 9 / 10 (300 / 350 / 350) reserved before train together with their 8
             symmetric images (none of them is trained on): the training distribution, unseen
  val        1,000 boards of the val classes (N = 8, 9, 10: 334 / 333 / 333, k balanced)
  test_a     2,000 boards of the test classes (667 / 667 / 666), not in val                     (paper: held-out families)
  test_b     2,000 boards of N = 11 / 12 (1,000 each, k uniform in 0..N-1, any solution)         (paper: never-trained sizes)
  test_hard  100 "trap" + 100 "few" boards per N = 8..12 (test classes for 8..10, any solution for 11 / 12), not in the
             other evaluation tables: trap = the topmost empty row has >= 3 safe cells of which exactly 1 is extendable and
             at most half of all safe cells are extendable; few = a unique completion with >= 3 queens to go (paper: hard boards)
Columns: id, table, N, k, state, label_r, label_c, src_sol, src_class ("N:<canonical>"), class_split (train / val / test /
ood), n_comp (completions = solutions containing the board), comp_train / comp_val / comp_test (completions in the classes of
each split), n_ext / n_safe / n_empty (extendable / non-attacked empty / empty on-board cells), ext (extendable cells as
canvas indices 12 r + c), row_ext / row_safe (the same in the topmost empty row), subtype (test_hard: trap / few).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import time

import numpy as np
import pandas as pd

from nbm.tasks.nqueens import rules as R

VERSION = "nqueens_v1"  # part of the class-order hash: changing it changes the splits
HELD = {8: (2, 2), 9: (8, 8), 10: (15, 15)}  # (val, test) classes per N
TRAIN_N = (5, 6, 7, 8, 9, 10)
QUOTA = {9: 80_000, 10: 105_000}  # N <= 8: every eligible board
SEEN = {8: 300, 9: 350, 10: 350}
VAL = {8: 334, 9: 333, 10: 333}
TEST_A = {8: 667, 9: 667, 10: 666}
TEST_B = {11: 1000, 12: 1000}
HARD_PER = 100
TABLES = ("train", "seen", "val", "test_a", "test_b", "test_hard")
# shuffle seed of each table (fixes its row order; these are the paper build's)
SHUFFLE_SEED = {"train": 1, "seen": 2, "val": 3, "test_a": 4, "test_b": 5, "test_hard": 6}


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ classes / splits
def class_split(n: int) -> dict[str, str]:
    """Canonical solution -> split ("train" / "val" / "test" / "ood"), by hash order."""
    cl = R.classes(n)
    if n > 10:
        return {c: "ood" for c in cl}
    names = sorted(cl, key=lambda c: hashlib.sha256(f"{VERSION}|{n}|{c}".encode()).hexdigest())
    nv, nt = HELD.get(n, (0, 0))
    return {c: "val" if i < nv else "test" if i < nv + nt else "train" for i, c in enumerate(names)}


def sol_split(n: int) -> tuple[list[str], np.ndarray]:
    """(class name of every solution, split of every solution)."""
    sols, _ = R.solutions(n)
    sp = class_split(n)
    can = [R.canon(x) for x in sols]
    return can, np.array([sp[c] for c in can])


def sub_state(x: str, rows: set[int]) -> str:
    return "".join(x[r] if r in rows else "." for r in range(len(x)))


def is_prefix_of(s: str, arr: np.ndarray) -> bool:
    """True iff s is contained in some row of the solution array arr (M, n)."""
    if len(arr) == 0:
        return False
    q = R.queens(s)
    if not q:
        return True
    rs = [r for r, _ in q]
    return bool((arr[:, rs] == np.array([c for _, c in q], np.int8)).all(1).any())


_SOL_INDEX: dict[int, dict[str, int]] = {}


def sol_index(n: int) -> dict[str, int]:
    if n not in _SOL_INDEX:
        _SOL_INDEX[n] = {x: i for i, x in enumerate(R.solutions(n)[0])}
    return _SOL_INDEX[n]


# ------------------------------------------------------------------ pools
def enumerate_pool(n: int, sol_idx, held_arr: np.ndarray | None) -> dict[int, dict[str, str]]:
    """k -> {board: source solution} over every row subset (k < n) of the given solutions, minus the boards contained in a
    held-out solution (if held_arr is given). The source is the first solution (in index order) that gives the board."""
    sols, _ = R.solutions(n)
    pools: dict[int, dict[str, str]] = {k: {} for k in range(n)}
    for i in sol_idx:
        x = sols[i]
        for k in range(n):
            for rows in itertools.combinations(range(n), k):
                s = sub_state(x, set(rows))
                if s not in pools[k]:
                    pools[k][s] = x
    if held_arr is not None:
        for k in range(n):
            pools[k] = {s: x for s, x in pools[k].items() if not is_prefix_of(s, held_arr)}
    return pools


def waterfill(sizes: dict[int, int], total: int) -> dict[int, int]:
    """Equal allocation per key, capped by sizes, the remainder spread over the uncapped keys."""
    alloc = {k: 0 for k in sizes}
    left, open_ = total, [k for k in sizes if sizes[k] > 0]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        nxt = []
        for k in open_:
            take = min(share, sizes[k] - alloc[k], left)
            alloc[k] += take
            left -= take
            if alloc[k] < sizes[k]:
                nxt.append(k)
            if left == 0:
                break
        open_ = nxt
    return alloc


def draw(pools, alloc, rng: random.Random, exclude=()) -> list[tuple[str, str]]:
    """(board, source) pairs drawn uniformly without replacement within each k."""
    out = []
    ex = set(exclude)
    for k in sorted(alloc):
        items = sorted((s, x) for s, x in pools[k].items() if s not in ex)
        out += rng.sample(items, min(alloc[k], len(items)))
    return out


# ------------------------------------------------------------------ rows
def row_stats(n: int, s: str, split_arr: np.ndarray) -> dict:
    _, arr = R.solutions(n)
    m = R.completion_mask(s)
    comp = arr[m]
    sp = split_arr[m]
    ext = set()
    for r in range(n):
        if s[r] == ".":
            ext.update((r, int(c)) for c in np.unique(comp[:, r]))
    safe = R.safe_cells(s)
    tr = R.top_row(s)
    return {"n_comp": int(m.sum()), "comp_train": int((sp == "train").sum()), "comp_val": int((sp == "val").sum()),
            "comp_test": int((sp == "test").sum()), "n_ext": len(ext), "n_safe": len(safe), "n_empty": n * n - R.k_of(s),
            "ext": sorted(12 * r + c for r, c in ext), "row_ext": sum(1 for r, _ in ext if r == tr),
            "row_safe": sum(1 for r, _ in safe if r == tr)}


def make_rows(table: str, items, split_cache, subtypes=None) -> pd.DataFrame:
    """items: (n, board, source solution) triples -> DataFrame (unshuffled)."""
    recs = []
    for j, (n, s, x) in enumerate(items):
        can, sp = split_cache[n]
        ci = sol_index(n)[x]
        r = R.top_row(s)
        recs.append({"table": table, "N": n, "k": R.k_of(s), "state": s, "label_r": r, "label_c": R.DIGITS.index(x[r]),
                     "src_sol": x, "src_class": f"{n}:{can[ci]}", "class_split": str(sp[ci]), **row_stats(n, s, sp),
                     "subtype": "" if subtypes is None else subtypes[j]})
    return pd.DataFrame(recs)


def finish(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Shuffle with the table's seed and add the row ids nq_<table>_<i>."""
    df = df.sample(frac=1.0, random_state=SHUFFLE_SEED[name]).reset_index(drop=True) if len(df) else df
    df.insert(0, "id", [f"nq_{name}_{i:06d}" for i in range(len(df))])
    df["table"] = name
    return df


# ------------------------------------------------------------------ build
def build_tables() -> tuple[dict[str, pd.DataFrame], dict]:
    t0 = time.time()
    split_cache = {n: sol_split(n) for n in range(5, 13)}
    stats: dict = {"pools": {}, "classes": {}}
    for n in range(5, 13):
        sp = class_split(n)
        stats["classes"][n] = {k: sum(v == k for v in sp.values()) for k in ("train", "val", "test", "ood")}
    tabs = {}
    train_items, seen_items, val_items, testa_items = [], [], [], []
    for n in TRAIN_N:
        sols, arr = R.solutions(n)
        _, sp = split_cache[n]
        tr_idx = [i for i in range(len(sols)) if sp[i] == "train"]
        held = arr[np.isin(sp, ["val", "test"])]
        pool = enumerate_pool(n, tr_idx, held if len(held) else None)
        sizes = {k: len(v) for k, v in pool.items()}
        stats["pools"][f"train_N{n}"] = sizes
        rng = random.Random(1000 + n)
        seen = draw(pool, waterfill(sizes, SEEN[n]), rng) if n in SEEN else []
        orbit = {R.sym_state(g, s) for s, _ in seen for g in R.SYMMETRIES}  # a seen board's 8 images stay out of train too
        rest = {k: sizes[k] - sum(1 for s in pool[k] if s in orbit) for k in sizes}
        alloc = rest if n not in QUOTA else waterfill(rest, QUOTA[n])
        tr = draw(pool, alloc, rng, exclude=orbit)
        seen_items += [(n, s, x) for s, x in seen]
        train_items += [(n, s, x) for s, x in tr]
        log(f"N {n}: train {len(tr)}, seen {len(seen)}")
        if n in HELD:
            for split, quota, dst in (("val", VAL[n], val_items), ("test", TEST_A[n], testa_items)):
                idx = [i for i in range(len(sols)) if sp[i] == split]
                hp = enumerate_pool(n, idx, None)
                if split == "test":  # not in val
                    vs = {s for (m, s, _) in val_items if m == n}
                    hp = {k: {s: x for s, x in v.items() if s not in vs} for k, v in hp.items()}
                hs = {k: len(v) for k, v in hp.items()}
                stats["pools"][f"{split}_N{n}"] = hs
                got = draw(hp, waterfill(hs, quota), random.Random(2000 + 10 * n + (split == "test")))
                dst += [(n, s, x) for s, x in got]
    tabs["train"] = finish(make_rows("train", train_items, split_cache), "train")
    log(f"train rows done ({time.time() - t0:.0f}s)")
    tabs["seen"] = finish(make_rows("seen", seen_items, split_cache), "seen")
    tabs["val"] = finish(make_rows("val", val_items, split_cache), "val")
    tabs["test_a"] = finish(make_rows("test_a", testa_items, split_cache), "test_a")
    taken = {(n, s) for n, s, _ in val_items + testa_items}
    # never-trained sizes
    tb = []
    for n, quota in TEST_B.items():
        sols, _ = R.solutions(n)
        rng = random.Random(3000 + n)
        got: dict[str, str] = {}
        while len(got) < quota:
            x = rng.choice(sols)
            k = rng.randint(0, n - 1)
            s = sub_state(x, set(rng.sample(range(n), k)))
            got.setdefault(s, x)
        tb += [(n, s, x) for s, x in got.items()]
    tabs["test_b"] = finish(make_rows("test_b", tb, split_cache), "test_b")
    taken |= {(n, s) for n, s, _ in tb}
    # hard boards
    hard, hard_types = [], []
    for n in range(8, 13):
        sols, _ = R.solutions(n)
        _, sp = split_cache[n]
        idx = [i for i in range(len(sols)) if sp[i] == ("test" if n <= 10 else "ood")]
        rng = random.Random(4000 + n)
        found: dict[str, dict[str, str]] = {"trap": {}, "few": {}}
        cands = []
        if n <= 10:  # exhaustive over the test-class solutions
            pool = enumerate_pool(n, idx, None)
            cands = sorted((s, x) for k in range(2, n - 1) for s, x in pool[k].items())
            rng.shuffle(cands)
        tries = 0
        while True:
            if n <= 10:
                if not cands:
                    break
                s, x = cands.pop()
            else:
                if tries > 400_000 or all(len(v) >= HARD_PER for v in found.values()):
                    break
                x = sols[rng.choice(idx)]
                s = sub_state(x, set(rng.sample(range(n), rng.randint(2, n - 2))))
            tries += 1
            if (n, s) in taken or any(s in v for v in found.values()):
                continue
            st = row_stats(n, s, sp)
            if st["row_safe"] >= 3 and st["row_ext"] == 1 and st["n_ext"] <= 0.5 * st["n_safe"]:
                t = "trap"
            elif st["n_comp"] == 1 and R.k_of(s) <= n - 3:
                t = "few"
            else:
                continue
            if len(found[t]) < HARD_PER:
                found[t][s] = x
            if all(len(v) >= HARD_PER for v in found.values()):
                break
        stats["pools"][f"hard_N{n}"] = {t: len(v) for t, v in found.items()}
        for t, v in found.items():
            for s, x in sorted(v.items()):
                hard.append((n, s, x))
                hard_types.append(t)
        log(f"hard N {n}: trap {len(found['trap'])} few {len(found['few'])} ({tries} candidates)")
    tabs["test_hard"] = finish(make_rows("test_hard", hard, split_cache, hard_types), "test_hard")
    stats["rows"] = {k: len(v) for k, v in tabs.items()}
    stats["by_N"] = {k: {int(a): int(b) for a, b in v.N.value_counts().sort_index().items()} for k, v in tabs.items()}
    stats["by_k"] = {k: {int(a): int(b) for a, b in v.k.value_counts().sort_index().items()} for k, v in tabs.items()}
    stats["random_safe_expected"] = {k: float((v.n_ext / v.n_safe.clip(lower=1)).mean()) for k, v in tabs.items()}
    stats["seconds"] = time.time() - t0
    return tabs, stats


def digest(df: pd.DataFrame) -> str:
    """sha256 of the (N, board, label) sequence of a table: a quick identity check of a build."""
    return hashlib.sha256("".join(f"{n}|{s}|{r}|{c};" for n, s, r, c in zip(df.N, df.state, df.label_r, df.label_c)).encode()).hexdigest()


# ------------------------------------------------------------------ checks
def check(tabs: dict[str, pd.DataFrame], n_recheck: int = 2000, seed: int = 0) -> dict:
    """Leakage and label checks (each entry has "pass"; "all_pass" summarises)."""
    rng = random.Random(seed)
    out: dict = {}
    tr = tabs["train"]
    held = {n: R.solutions(n)[1][np.isin(sol_split(n)[1], ["val", "test"])] for n in TRAIN_N}
    # 1. classes: disjoint between train / val / test_a, every table from its own classes
    cs = {t: set(tabs[t].src_class) for t in ("train", "seen", "val", "test_a")}
    ok = not (cs["train"] & cs["val"]) and not (cs["train"] & cs["test_a"]) and not (cs["val"] & cs["test_a"])
    ok = ok and set(tr.class_split) == {"train"} and set(tabs["seen"].class_split) == {"train"} \
        and set(tabs["val"].class_split) == {"val"} and set(tabs["test_a"].class_split) == {"test"}
    out["classes_disjoint"] = {"pass": bool(ok), "train_classes": len(cs["train"]), "val_classes": len(cs["val"]),
                               "test_classes": len(cs["test_a"])}
    # 2. no train / seen board is contained in a held-out solution (every row), re-checked on a sample by string comparison
    bad = sum(len(held[n]) > 0 and is_prefix_of(s, held[n]) for t in ("train", "seen") for n, s in zip(tabs[t].N, tabs[t].state))
    sample = rng.sample(list(zip(tr.N, tr.state)), min(n_recheck, len(tr)))
    bad2 = 0
    for n, s in sample:
        hs = {R.from_cols(list(map(int, h))) for h in held[n]}
        bad2 += any(all(R.DIGITS.index(h[r]) == c for r, c in R.queens(s)) for h in hs)
    out["train_not_heldout_prefix"] = {"pass": bad == 0 and bad2 == 0, "violations": int(bad), "violations_recheck": int(bad2),
                                       "recheck_sample": len(sample)}
    # 3. no evaluation board is a training board, also under the 8 symmetries
    trs = set(zip(tr.N, tr.state))
    hits = {t: int(sum(any((n, R.sym_state(g, s)) in trs for g in R.SYMMETRIES) for n, s in zip(tabs[t].N, tabs[t].state)))
            for t in TABLES if t != "train"}
    out["eval_boards_not_in_train"] = {"pass": all(v == 0 for v in hits.values()), "hits_incl_symmetries": hits}
    # 4. tables disjoint, no duplicates
    def boards(t):
        return set(zip(tabs[t].N, tabs[t].state))
    inter = {f"{a}&{b}": len(boards(a) & boards(b)) for a, b in (("val", "test_a"), ("seen", "train"), ("test_hard", "test_a"),
                                                                 ("test_hard", "test_b"), ("test_hard", "val"))}
    dup = {t: int(len(tabs[t]) - len(boards(t))) for t in TABLES}
    out["tables_disjoint"] = {"pass": all(v == 0 for v in inter.values()) and all(v == 0 for v in dup.values()),
                              "intersections": inter, "duplicates": dup}
    # 5. labels: in the topmost empty row, from the source solution, extendable; row statistics re-derived on samples
    lab_bad, stat_bad = 0, 0
    for t, df in tabs.items():
        for n, s, r, c, x in zip(df.N, df.state, df.label_r, df.label_c, df.src_sol):
            if r != R.top_row(s) or R.DIGITS.index(x[r]) != c or not R.is_partial(s) or R.k_of(s) >= n:
                lab_bad += 1
        for i in rng.sample(range(len(df)), min(300, len(df))):
            row = df.iloc[i]
            s, n = row.state, int(row.N)
            ext = R.extendable_cells(s)
            slow = n >= 11 and R.k_of(s) < 5  # the pure-Python counter is slow on near-empty 11 / 12 boards
            if not R.is_extendable(s, int(row.label_r), int(row.label_c)) or sorted(12 * a + b for a, b in ext) != list(row.ext) \
                    or (not slow and int(row.n_comp) != R.count_completions_bitmask(s)) or len(R.safe_cells(s)) != int(row.n_safe):
                stat_bad += 1
    out["labels_extendable"] = {"pass": lab_bad == 0 and stat_bad == 0, "label_violations": lab_bad, "stat_mismatches": stat_bad}
    out["all_pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def load(out: str, name: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(out, f"{name}.parquet"))
    df["state"] = df["state"].astype(str)
    return df


def build(out: str, force: bool = False) -> dict:
    if os.path.exists(os.path.join(out, "build_stats.json")) and not force:
        raise SystemExit(f"{out} already holds a build (use --force)")
    os.makedirs(out, exist_ok=True)
    tabs, stats = build_tables()
    for name, df in tabs.items():
        df.to_parquet(os.path.join(out, f"{name}.parquet"), index=False)
    with open(os.path.join(out, "digest.json"), "w") as fh:
        json.dump({k: digest(v) for k, v in tabs.items()}, fh, indent=1)
    with open(os.path.join(out, "build_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    log("checks")
    ch = check({k: load(out, k) for k in TABLES})
    with open(os.path.join(out, "checks.json"), "w") as fh:
        json.dump(ch, fh, indent=1)
    log(json.dumps(stats["rows"]), "checks all_pass", ch["all_pass"])
    return ch


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build / check the N-queens dataset (nbm/tasks/nqueens/data.py docstring).")
    p.add_argument("cmd", choices=["build", "check"])
    p.add_argument("--out", required=True, help="dataset directory, e.g. $NBM_DATA/nqueens")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "build":
        ch = build(a.out, a.force)
    else:
        ch = check({k: load(a.out, k) for k in TABLES})
        print(json.dumps(ch, indent=1))
    if not ch["all_pass"]:
        raise SystemExit("checks FAILED")


if __name__ == "__main__":
    main()
