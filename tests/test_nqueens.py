"""Fast CPU tests of the N-queens task: solver and symmetry classes, completions, render -> mark -> decode round trips,
scoring, training examples, the class splits, and the evaluation loop with fake models (no GPU, no model weights)."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageFilter

from nbm.config import from_dict
from nbm.data import ImageTrainSet
from nbm.tasks import make_task
from nbm.tasks.base import EvalContext
from nbm.tasks.nqueens import data as D
from nbm.tasks.nqueens import render as V
from nbm.tasks.nqueens import rules as R


def random_board(rng: random.Random, n: int | None = None) -> str:
    """A partial board: a random subset of the rows of a random solution."""
    n = n or rng.randint(5, 11)
    x = rng.choice(R.solutions(n)[0])
    rows = rng.sample(range(n), rng.randint(0, n - 1))
    return R.from_queens(n, [(r, R.DIGITS.index(x[r])) for r in rows])


# ------------------------------------------------------------------ rules
def test_solution_and_class_counts():
    for n in range(4, 13):
        assert len(R.solutions(n)[0]) == R.KNOWN_COUNTS[n]
        assert len(R.classes(n)) == R.KNOWN_CLASSES[n]
        assert all(len(v) in (1, 2, 4, 8) for v in R.classes(n).values())
    for n in range(4, 10):
        assert R.count_completions_bitmask(R.empty(n)) == R.KNOWN_COUNTS[n]


def test_completions_and_extendable_cells_against_brute_force():
    rng = random.Random(0)
    for _ in range(150):
        s = random_board(rng, rng.randint(5, 10))
        assert R.is_partial(s)
        assert R.n_completions(s) == R.count_completions_bitmask(s)
        brute = [(r, c) for r, c in R.safe_cells(s) if R.count_completions_bitmask(R.place(s, r, c)) > 0]
        assert R.extendable_cells(s) == brute
        for g in R.SYMMETRIES:  # a symmetric image of a board has as many completions
            assert R.n_completions(R.sym_state(g, s)) == R.n_completions(s)


def test_classify():
    s = "1.3....."  # N = 8, queens on (0, 1) and (2, 3)
    assert R.classify(s, (0, 1)) == "occupied"
    assert R.classify(s, (0, 5)) == "attacked"  # a second queen in row 0
    assert R.classify(s, (1, 2)) == "attacked"  # diagonal
    assert R.classify(s, (3, 8)) == "off_board" and R.classify(s, (9, 0)) == "off_board"
    for cell in R.safe_cells(s):
        assert R.classify(s, cell) == ("correct" if cell in R.extendable_cells(s) else "dead")


# ------------------------------------------------------------------ render / decode
def test_render_decode_round_trip_every_cell():
    rng = random.Random(1)
    for t in range(6):
        s = random_board(rng, rng.randint(4, 12))
        assert V.decode(V.render(s), s)["alpha"] == pytest.approx(0, abs=1e-3)
        cells = [(r, c) for r in range(V.GRID) for c in range(V.GRID)] if t < 2 else rng.sample(
            [(r, c) for r in range(V.GRID) for c in range(V.GRID)], 30)
        for cell in cells:
            d = V.decode(V.render(s, cell), s)
            assert d["cell"] == cell and abs(d["alpha"] - 1) < 0.02


def test_decode_blur_noise_faint():
    rng = random.Random(2)
    for t in range(20):
        s = random_board(rng)
        cell = rng.choice(R.safe_cells(s) or [(0, 0)])
        tgt = V.render(s, cell)
        assert V.decode(Image.fromarray(tgt).filter(ImageFilter.GaussianBlur(1.2)), s)["cell"] == cell
        noisy = np.clip(tgt.astype(np.float32) + np.random.default_rng(t).normal(0, 12, tgt.shape), 0, 255).astype(np.uint8)
        assert V.decode(noisy, s)["cell"] == cell
        faint = (0.75 * V.render(s).astype(np.float32) + 0.25 * tgt.astype(np.float32)).astype(np.uint8)
        d = V.decode(faint, s)
        assert d["cell"] == cell and 0.15 < d["alpha"] < 0.35  # threshold-free: a faint mark still decides


# ------------------------------------------------------------------ data
def test_class_splits():
    counts = {n: pd.Series(D.class_split(n)).value_counts().to_dict() for n in (7, 8, 9, 10, 11)}
    assert counts[7] == {"train": 6}
    assert counts[8] == {"train": 8, "val": 2, "test": 2}
    assert counts[9] == {"train": 30, "val": 8, "test": 8}
    assert counts[10] == {"train": 62, "val": 15, "test": 15}
    assert counts[11] == {"ood": 341}
    assert D.waterfill({0: 1, 1: 10, 2: 10}, 9) == {0: 1, 1: 4, 2: 4}


def small_table(tmp_path, name="val", n=8, rows=12, seed=0):
    """A table of `rows` boards of N = n drawn from its val classes, written as <tmp_path>/nqueens/<name>.parquet."""
    rng = random.Random(seed)
    can, sp = D.sol_split(n)
    sols = [x for x, s in zip(R.solutions(n)[0], sp) if s == "val"]
    items = []
    while len(items) < rows:
        x = rng.choice(sols)
        s = D.sub_state(x, set(rng.sample(range(n), rng.randint(1, n - 1))))
        items.append((n, s, x))
    df = D.finish(D.make_rows(name, items, {n: (can, sp)}), name)
    (tmp_path / "nqueens").mkdir(exist_ok=True)
    df.to_parquet(tmp_path / "nqueens" / f"{name}.parquet", index=False)
    return df


def config(tmp_path, monkeypatch):
    monkeypatch.setenv("NBM_DATA", str(tmp_path))
    import yaml
    with open(Path(__file__).parents[1] / "configs" / "nqueens" / "image.yaml") as fh:
        return from_dict(yaml.safe_load(fh))


def test_image_example_and_labels(tmp_path, monkeypatch):
    df = small_table(tmp_path)
    task = make_task(config(tmp_path, monkeypatch))
    assert len(task.load_split("val")) == len(df)
    item = ImageTrainSet(task, df)[0]
    ex = task.image_example(df.iloc[0].to_dict(), None)
    row = df.iloc[0]
    assert ex.prompt == "Continue." and ex.cond.size == ex.target.size == (384, 384)
    x0, y0, x1, y1 = ex.boxes[0]
    assert (x0, y0) == (32 * row.label_c, 32 * row.label_r) and item["boxes"] == [tuple(ex.boxes[0])]
    diff = np.argwhere((np.asarray(ex.cond) != np.asarray(ex.target)).any(-1))
    assert diff[:, 1].min() >= x0 and diff[:, 1].max() < x1 and diff[:, 0].min() >= y0 and diff[:, 0].max() < y1
    for r in df.itertuples():
        assert r.label_r == R.top_row(r.state) and R.is_extendable(r.state, r.label_r, r.label_c)
        assert sorted(12 * a + b for a, b in R.extendable_cells(r.state)) == list(r.ext)


# ------------------------------------------------------------------ evaluation loop
class FakeModel:
    """Draws the label queen (or, with `wrong`, an attacked cell) on the condition picture."""

    def __init__(self, df, wrong=False):
        self.by_state = {r.state: (r.label_r, r.label_c) for r in df.itertuples()}
        self.wrong = wrong

    def generate_seeded(self, conds, prompts, steps, seeds):
        out = []
        for c in conds:
            s = next(s for s in self.by_state if np.array_equal(np.asarray(c), V.render(s)))
            cell = self.by_state[s]
            if self.wrong:
                r, c0 = R.queens(s)[0]
                cell = (r, (c0 + 1) % len(s))
            out.append(V.render_pil(s, cell))
        return out


def test_evaluate_with_fake_models(tmp_path, monkeypatch):
    df = small_table(tmp_path)
    cfg = config(tmp_path, monkeypatch)
    task = make_task(cfg)
    ctx = EvalContext(cfg, FakeModel(df), "val", steps=1, votes=1, seed=0, batch_size=5, out_dir=tmp_path / "out")
    res = task.evaluate(ctx, df)
    assert res.metrics["score"] == 1.0 and res.metrics["n"] == len(df) and res.metrics["legal"] == 1.0
    assert [r["id"] for r in res.records] == list(df.id) and all(r["cat"] == "correct" for r in res.records)
    assert len(list((tmp_path / "out" / "strips").glob("*.png"))) == 8
    bad = task.evaluate(EvalContext(cfg, FakeModel(df, wrong=True), "val", 1, 3, 0, 4), df)
    assert bad.metrics["score"] == 0.0 and bad.metrics["cat/attacked"] == 1.0
    assert all(len(r["samples"]) == 3 for r in bad.records)
