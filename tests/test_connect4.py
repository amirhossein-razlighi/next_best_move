"""Connect Four task: rules, render -> decode round trips, training examples, the evaluation loop with fake models (CPU)."""
import json
import random

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from nbm.config import load_config
from nbm.models.video import N_BLOCKS, blocks_to_frames
from nbm.tasks import make_task
from nbm.tasks.base import EvalContext, write_result
from nbm.tasks.connect4 import data as D
from nbm.tasks.connect4 import decode as DEC
from nbm.tasks.connect4 import game as G
from nbm.tasks.connect4 import render as R
from nbm.tasks.connect4 import solver as S


def random_position(rng: random.Random, max_plies: int = 30) -> str:
    b = G.EMPTY
    for _ in range(rng.randrange(0, max_plies)):
        if G.terminal(b):
            break
        b = G.play(b, rng.choice(G.legal(b)))
    return b


def test_rules():
    b = G.EMPTY
    for c in (3, 3, 2, 2, 1, 1):
        b = G.play(b, c)
    assert G.mover(b) == "x" and G.landing(b, 3) == 2 and G.immediate_wins(b) == [0, 4]
    assert G.kind_of(b) == "win" and G.winner(G.play(b, 0)) == "x" and G.terminal(G.play(b, 4))
    assert G.kind_of(G.play(b, 6)) == "block"
    assert G.mirror(G.mirror(b)) == b and G.canon(b) == G.canon(G.mirror(b)) and G.valid(b)
    floating = "." + "x" + "." * 40  # a disc above an empty cell
    assert not G.valid(floating)
    full = "xoxoxo" + "." * 36
    assert G.landing(full, 0) is None and 0 not in G.legal(full)
    with pytest.raises(ValueError):
        G.play(full, 0)


def test_score_labels():
    sc = (-3, None, 0, 5, 5, 0, -10)
    assert G.value(sc) == 1 and G.best_set(sc) == [3, 4] and G.vp_set(sc) == [3, 4] and G.canonical_best(sc) == 3
    assert G.critical(sc) and not G.critical((0, 0, None, 0, 0, 0, 0))
    assert G.canonical_best((2, 2, 2, -1, 2, 2, 2)) == 2


def test_render_decode_round_trip():
    rng = random.Random(0)
    for _ in range(40):
        b = random_position(rng)
        pic = R.render(b)
        assert pic.shape == (256, 256, 3) and pic.dtype == np.uint8
        assert DEC.read_cells(pic) == b and DEC.read_turn(pic) == ("." if G.terminal(b) else G.mover(b))
        if G.terminal(b):
            continue
        for c in G.legal(b):
            mv = DEC.read_move(R.render(b, decide=c), b)
            assert (mv["col"], mv["row"]) == (c, G.landing(b, c)) and mv["valid"] and abs(mv["strength"] - 1) < 0.05
        assert not DEC.read_move(pic, b)["drawn"]  # no mark at all: no move


def test_wrong_landing_row_is_invalid():
    b = G.play(G.EMPTY, 3)  # column 3 holds one disc; a disc drawn two rows up floats
    pic = R.render(b).copy()
    m = R.masks()
    i = G.idx(3, 2)
    pic.reshape(-1, 3)[m["disc"][i]] = R.BLUE
    pic.reshape(-1, 3)[m["tint"][i]] = R.GREEN
    mv = DEC.read_move(pic, b)
    assert (mv["col"], mv["row"]) == (3, 2) and mv["legal"] and not mv["landing_ok"] and not mv["valid"]


def test_line_blocks_and_read_line():
    rng = random.Random(1)
    for _ in range(20):
        b = random_position(rng, 20)
        if G.terminal(b):
            continue
        line, cur = [], b
        while len(line) < R.MAX_PLIES and not G.terminal(cur):
            c = rng.choice(G.legal(cur))
            line.append(c)
            cur = G.play(cur, c)
        blocks = R.line_blocks(b, line)
        assert blocks.shape == (N_BLOCKS, 256, 256, 3)
        first = DEC.read_move(blocks[1], b)
        got = DEC.read_line(list(blocks), b, first)
        assert got["moves"] == line and got["imagine_ok"] in (1.0, None)
        assert got["stop"] == ("game_over" if G.terminal(cur) and len(line) < R.MAX_PLIES else "max_plies")
        broken = blocks.copy()
        broken[2] = blocks[0]  # the IMAGINE block shows the position before the move
        assert DEC.read_line(list(broken), b, first)["stop"] == "imagine_mismatch"


# ---------------------------------------------------------------- task with a tiny synthetic data set (fake labels)
def tiny_data(tmp_path, n=6):
    """Rows with the data's schema; scores are invented (the canonical column scores 1, every other column -1)."""
    rng = random.Random(2)
    rows = []
    while len(rows) < n:
        b = random_position(rng, 16)
        if G.terminal(b) or len(G.legal(b)) < 2:
            continue
        choice = G.legal(b)[len(rows) % len(G.legal(b))]
        scores = [S.FULL if c not in G.legal(b) else (1 if c == choice else -1) for c in range(G.W)]
        line, cur = [], b
        while len(line) < R.MAX_PLIES and not G.terminal(cur):
            c = rng.choice(G.legal(cur)) if line else choice
            line.append(c)
            cur = G.play(cur, c)
        rows.append({"id": f"c4_test_{len(rows):06d}", "board": b, "ply": G.plies(b), "mover": G.mover(b), "scores": scores,
                     "value": 1, "best": [choice], "choice": choice, "vp": [choice], "critical": True, "kind": G.kind_of(b),
                     "hard": G.kind_of(b) == "quiet", "n_legal": len(G.legal(b)), "line": line, "line_len": len(line),
                     "line_end": "open"})
    d = tmp_path / "connect4"
    d.mkdir()
    for split in ("train", "test"):
        pd.DataFrame(rows).to_parquet(d / f"{split}.parquet", index=False)
    return pd.DataFrame(rows)


class FakeImageModel:
    """Draws the DECIDE picture of the row's label (looked up from the decoded board)."""

    def __init__(self, choice_of):
        self.choice_of = choice_of

    def generate_seeded(self, conds, prompts, steps, seeds):
        out = []
        for c in conds:
            b = DEC.read_cells(np.asarray(c))
            out.append(Image.fromarray(R.render(b, decide=self.choice_of[b][0])))
        return out


class FakeVideoModel(FakeImageModel):
    def generate_seeded(self, firsts, prompts, steps, shift, seeds):
        clips = []
        for c in firsts:
            b = DEC.read_cells(np.asarray(c))
            clips.append(np.stack([np.asarray(x) for x in blocks_to_frames(list(R.line_blocks(b, self.choice_of[b][1])))]))
        return torch.from_numpy(np.stack(clips))


@pytest.mark.parametrize("family", ["image", "video"])
def test_examples_and_evaluate_with_fake_model(family, tmp_path, monkeypatch):
    df = tiny_data(tmp_path)
    monkeypatch.setenv("NBM_DATA", str(tmp_path))
    cfg = load_config(f"configs/connect4/{family}.yaml")
    task = make_task(cfg)
    rows = task.load_split("train")
    assert rows.line.tolist() == df.line.tolist() and isinstance(rows.scores[0][0], int)
    row = rows.to_dict("records")[0]
    if family == "image":
        ex = task.image_example(row, None)
        assert ex.cond.size == ex.target.size == (256, 256) and ex.prompt == "Continue."
        x0, y0, x1, y1 = ex.boxes[0]
        assert DEC.read_move(np.asarray(ex.target), row["board"])["col"] == row["choice"] and x1 - x0 == y1 - y0 == 32
        model = FakeImageModel({r.board: (r.choice, r.line) for r in df.itertuples()})
    else:
        ex = task.video_example(row, None)
        assert ex.blocks.shape == (N_BLOCKS, 256, 256, 3) and ex.blocks.dtype == np.uint8 and task.prompts() == ["Continue."]
        model = FakeVideoModel({r.board: (r.choice, r.line) for r in df.itertuples()})
    test = task.load_split("test")
    ctx = EvalContext(cfg, model, "test", steps=1, votes=2, seed=0, batch_size=4, out_dir=tmp_path / "out")
    (tmp_path / "out").mkdir()
    res = task.evaluate(ctx, test)
    m = res.metrics
    assert m["n"] == len(df) and m["legal_landing"] == m["best"] == m["keeps_value"] == 1.0 and m["score"] >= 1.0
    assert all(r["col"] == r["choice"] and r["samples"] == [r["choice"]] * 2 for r in res.records)
    if family == "video":
        assert all(r["imagined"]["moves"] == line for r, line in zip(res.records, df.line))
    write_result(res, tmp_path / "out", {"split": "test"})
    assert json.load(open(tmp_path / "out" / "metrics.json"))["best"] == 1.0
    assert len(list((tmp_path / "out" / "strips").iterdir())) == len(df)


def test_data_digest_and_eval_pick():
    df = pd.DataFrame({"board": [G.EMPTY], "choice": [3], "line": [[3, 3]]})
    assert len(D.table_digest(df)) == 64
    b = G.play(G.EMPTY, 3)
    pos = [(G.EMPTY, (0,) * 7), (b, (1, -1, -1, -1, -1, -1, -1))]
    assert D.pick_eval_position("val", 5, pos, set()) == (1, b, pos[1][1])  # the only critical position
    assert D.pick_eval_position("val", 5, pos, {G.canon(b)}) is None


@pytest.mark.skipif(not (S.solver_dir() / "bin" / "c4board").exists(), reason="Pons solver not built (pons/build.sh)")
def test_solver_scores():
    sc = S.solver().scores(G.EMPTY)
    assert sc == (-2, -1, 0, 1, 0, -1, -2)  # the centre column wins for the first player
    assert S.best_line(G.EMPTY, 2) == [3, 3]
