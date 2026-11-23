"""Go task: rules, render -> mark -> decode round trips, the evaluation loop with a fake model, the data builder's pieces (CPU)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from nbm.config import from_dict
from nbm.tasks import make_task
from nbm.tasks.base import EvalContext, write_result
from nbm.tasks.go import build as B
from nbm.tasks.go import render, rules
from nbm.tasks.go import prompt_for, side_batches

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "go" / "image.yaml"
SIZE = 384


def position(moves: list[str]) -> dict:
    """A position row after the given moves ("B D4" ...), with the next move of the list as its label when given."""
    states = rules.replay([(m.split()[0], rules.from_gtp(m.split()[1])) for m in moves])
    b, col = states[-1]
    black, white = rules.stones(b.g)
    return {"black": black, "white": white, "side": rules.side_name(col), "ko": rules.idx_of(b.ko) if b.ko else -1}


def gtp_moves(points: list[tuple[int, int]]) -> list[str]:
    """Alternating moves (Black first) on the given points."""
    return [f"{'BW'[k % 2]} {rules.gtp(p)}" for k, p in enumerate(points)]


def test_gtp_and_index_round_trip():
    for i in range(361):
        assert rules.idx_of(rules.from_gtp(rules.gtp(rules.pt_of(i)))) == i
    assert rules.gtp((3, 15)) == "D4" and rules.from_gtp("T19") == (18, 0) and rules.gtp(None) == "pass"


def test_capture_suicide_and_ko():
    # capture: White surrounds the black stone on D4
    states = rules.replay([(m.split()[0], rules.from_gtp(m.split()[1])) for m in
                           ["B D4", "W C4", "B Q16", "W D5", "B Q4", "W E4", "B Q10", "W D3"]])
    final, col = states[-1]
    assert final.g[15, 3] == rules.EMPTY and col == rules.BLACK
    # suicide: Black may not play A1 between White's A2 and B1
    assert not rules.legal_mask(position(["B Q16", "W A2", "B Q4", "W B1"]))[18, 0]
    # simple ko: Black captures the white stone on (6, 6) with (7, 6); White may not retake at once
    pts = [(6, 5), (7, 5), (5, 6), (8, 6), (6, 7), (7, 7), (15, 15), (6, 6)]
    before = position(gtp_moves(pts))
    assert before["side"] == "black" and rules.legal_mask(before)[6, 7]
    after = position(gtp_moves(pts + [(7, 6)]))
    assert after["side"] == "white" and after["ko"] == rules.idx_of((6, 6)) and rules.idx_of((6, 6)) not in after["white"]
    assert not rules.legal_mask(after)[6, 6]
    with pytest.raises(rules.IllegalMove):
        rules.board_of_row(after).play(rules.WHITE, (6, 6))


def test_render_mark_decode_round_trip():
    row = position(["B Q16", "W D4", "B Q4", "W D16", "B C3"])
    cond = render.render_board(row, SIZE)
    assert cond.size == (SIZE, SIZE) and cond.mode == "RGB"
    a = render.red_strengths(cond, cond)
    assert a.max() == 0
    for pt in [(0, 0), (18, 18), (9, 9), (3, 15), (16, 2)]:
        tgt = render.draw_move(cond, pt)
        x0, y0, x1, y1 = render.cell_box(pt, SIZE)
        changed = np.argwhere((np.asarray(tgt) != np.asarray(cond)).any(-1))
        assert changed[:, 0].min() >= y0 and changed[:, 0].max() < y1 and changed[:, 1].min() >= x0 and changed[:, 1].max() < x1
        move, strength = render.read_move(tgt, cond)
        assert move == pt and 0.95 < strength < 1.05


def test_readout_takes_the_strongest_disc():
    cond = render.render_board(position(["B Q16"]), SIZE)
    c = np.asarray(cond, np.float32)
    weak, strong = (4, 4), (12, 7)
    arr = c + 0.4 * (np.asarray(render.draw_move(cond, weak), np.float32) - c) + 0.8 * (np.asarray(render.draw_move(cond, strong), np.float32) - c)
    out = Image.fromarray(arr.round().clip(0, 255).astype(np.uint8))
    assert render.read_move(out, cond)[0] == strong


def toy_split(n: int, seed: int) -> pd.DataFrame:
    """Rows of random short games: id, side, stones, ko, label = a random legal point, phase."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        b = rules.Board()
        col = rules.BLACK
        for _ in range(int(rng.integers(4, 30))):
            ys, xs = np.nonzero(b.legal(col))
            j = int(rng.integers(len(xs)))
            b.play(col, (int(xs[j]), int(ys[j])))
            col = rules.WHITE if col == rules.BLACK else rules.BLACK
        black, white = rules.stones(b.g)
        ys, xs = np.nonzero(b.legal(col))
        j = int(rng.integers(len(xs)))
        rows.append({"id": f"g{seed}_{k}", "side": rules.side_name(col), "black": black, "white": white,
                     "ko": rules.idx_of(b.ko) if b.ko else -1, "correct": [rules.idx_of((int(xs[j]), int(ys[j])))],
                     "phase": rules.PHASES[k % 4][0]})
    return pd.DataFrame(rows)


@pytest.fixture()
def task(tmp_path, monkeypatch):
    d = tmp_path / "data" / "go"
    d.mkdir(parents=True)
    for s, n in (("train", 6), ("val", 8), ("test", 7)):
        toy_split(n, len(s)).to_parquet(d / f"{s}.parquet")
    monkeypatch.setenv("NBM_DATA", str(tmp_path / "data"))
    cfg = yaml.safe_load(open(CONFIG))
    cfg["data"]["dir"] = "go"
    return make_task(from_dict(cfg))


def test_image_example(task):
    df = task.load_split("train")
    groups = task.batch_groups(df)
    assert ((groups == 1) == (df.side.values == "black")).all()
    row = df.to_dict("records")[0]
    ex = task.image_example(row, None)
    pt = rules.pt_of(row["correct"][0])
    assert ex.prompt == prompt_for(row["side"]) and list(ex.boxes) == [render.cell_box(pt, SIZE)]
    assert render.read_move(ex.target, ex.cond)[0] == pt


class FakeModel:
    """Draws the disc on the label point; records the seeds it was asked for."""

    def __init__(self, labels):
        self.labels, self.calls = labels, []

    def generate_seeded(self, conds, prompts, steps, seeds):
        self.calls.append((list(prompts), list(seeds)))
        return [render.draw_move(c, self.labels[np.asarray(c).tobytes()]) for c in conds]


def test_evaluate_with_fake_model(task, tmp_path):
    df = task.load_split("test")
    labels = {np.asarray(render.render_board(r, SIZE)).tobytes(): rules.pt_of(r["correct"][0]) for r in df.to_dict("records")}
    model = FakeModel(labels)
    ctx = EvalContext(task.cfg, model, "test", steps=1, votes=2, seed=3, batch_size=2)
    res = task.evaluate(ctx, df)
    assert res.metrics["score"] == res.metrics["best_move"] == res.metrics["legal_move"] == 1.0 and res.metrics["n"] == 7
    assert all(r["samples"] == [r["label"]] * 2 and r["move"] == r["label"] for r in res.records)
    # batches: white rows first, then black; noise seed of row i, sample j = (seed + j) * 1,000,003 + i
    batches = side_batches(df.side.values, 2)
    assert [s for _, s in model.calls[:len(batches)]] == [[3 * 1_000_003 + i for i in b] for b in batches]
    assert model.calls[len(batches)][1] == [4 * 1_000_003 + i for i in batches[0]]
    assert all(len(set(p)) == 1 for p, _ in model.calls)
    write_result(res, tmp_path / "out", {"split": "test"})
    assert json.load(open(tmp_path / "out" / "metrics.json"))["best_move"] == 1.0


def test_val_order_is_phase_stratified(task):
    val = task.load_split("val")
    assert sorted(val.id) == sorted(pd.read_parquet(task.data_dir / "val.parquet").id)
    assert set(val.head(4).phase) == {p for p, _, _ in rules.PHASES}


# ---------------------------------------------------------------- data builder
SGF = "(;GM[1]FF[4]SZ[19]PB[alice]PW[bob]BR[7d]WR[7d]KM[6.50]RE[B+R]RU[Japanese]DT[2010-01-01]{moves})"


def sgf_game(n_moves: int, seed: int = 0) -> str:
    b, col, rng, out = rules.Board(), rules.BLACK, np.random.default_rng(seed), []
    for _ in range(n_moves):
        ys, xs = np.nonzero(b.legal(col))
        j = int(rng.integers(len(xs)))
        b.play(col, (int(xs[j]), int(ys[j])))
        out.append(f";{'B' if col == rules.BLACK else 'W'}[{chr(97 + int(xs[j]))}{chr(97 + int(ys[j]))}]")
        col = rules.WHITE if col == rules.BLACK else rules.BLACK
    return SGF.format(moves="".join(out))


def test_load_game_filters(tmp_path):
    cases = {"ok.sgf": sgf_game(100), "short.sgf": sgf_game(30), "bot.sgf": sgf_game(100).replace("PB[alice]", "PB[GnuGoBot]"),
             "hcap.sgf": sgf_game(100).replace("SZ[19]", "SZ[19]HA[2]AB[dd][pp]"), "old.sgf": sgf_game(100).replace("2010", "2017"),
             "size.sgf": sgf_game(100).replace("SZ[19]", "SZ[13]")}
    for name, text in cases.items():
        (tmp_path / name).write_text(text)
    reasons = {n: B.load_game((str(tmp_path / n), n))["reason"] for n in cases}
    assert reasons == {"ok.sgf": "ok", "short.sgf": "short", "bot.sgf": "bot", "hcap.sgf": "handicap_or_setup", "old.sgf": "year",
                       "size.sgf": "size"}
    g = B.load_game((str(tmp_path / "ok.sgf"), "ok.sgf"))
    assert len(g["moves"]) == 100 and g["year"] == 2010


def test_plies_and_positions():
    _, nodes = rules.parse_sgf(sgf_game(120, seed=4))
    moves = [f"{c} {rules.gtp(rules.sgf_pt(n[c][0]))}" for n in nodes for c in ("B", "W") if c in n]
    g = {"game": "x/y.sgf", "game_rank": 0, "moves": moves, **{k: "" for k in B.META}}
    ev = B.eval_plies(g["game"], moves)
    assert ev == B.eval_plies(g["game"], moves) and len(ev) == B.EVAL_PER_GAME and all(20 <= t < 115 for t in ev)
    tr = B.train_plies(g["game"], moves, ev)
    assert set(ev) <= set(tr) and len(tr) == B.TRAIN_PER_GAME and tr == sorted(tr)
    for r in B.position_rows(g, ev):
        assert r["phase"] == rules.phase_of(r["ply"]) and r["moves"] == moves[:r["ply"]] and r["game_move"] == moves[r["ply"]]
        assert r["side"] == ("black" if r["ply"] % 2 == 0 else "white") and len(r["black"]) + len(r["white"]) <= r["ply"]


def test_labelled_row_and_eval_splits():
    base = position(["B Q16", "W D4"])
    lab = {"best": "Q4", "best_order0": "Q4", "root_winrate": 0.5, "root_score": 0.1, "root_visits": 401,
           "cands": [{"move": "Q4", "visits": 300, "winrate": 0.52, "scoreLead": 0.1, "prior": 0.4, "order": 0, "lcb": 0.5}]}
    row, why = B.labelled_row(base, lab)
    assert why == "kept" and row["correct"] == [rules.idx_of(rules.from_gtp("Q4"))] and row["best_winrate"] == 0.52
    assert B.labelled_row(base, {**lab, "root_winrate": 0.99})[1] == "decided"
    assert B.labelled_row(base, {**lab, "best": "Q16"})[1] == "best_illegal_ours"
    assert B.labelled_row(base, {**lab, "best": "pass"})[1] == "best_pass"
    assert B.labelled_row(base, None)[1] == "unlabelled"
    # games of 3 positions; game 1 repeats a board of game 0 (shared: never in test / val)
    rows = []
    for g in range(1500):
        for t in range(3):
            stones = [g * 0 + t] if g in (0, 1) else [(g * 7 + t) % 361, 360 - g % 300]
            rows.append({"id": f"{g}#{t}", "game": f"g{g}", "game_rank": g, "ply": 20 + t, "side": "black", "black": stones, "white": []})
    sp = B.eval_splits(pd.DataFrame(rows))
    assert len(sp["test"]) == B.N_TEST and len(sp["val"]) == B.N_VAL
    assert not ({"g0", "g1"} & set(sp["test"].game)) and not (set(sp["test"].game) & set(sp["val"].game))
    assert sp["test"].game_rank.max() < sp["val"].game_rank.min()
