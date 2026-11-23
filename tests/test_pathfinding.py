"""CPU tests of the pathfinding tasks: data builder, render -> decode round trips, training examples, and both evaluation loops
with a fake model that plans on the decoded picture (python -m pytest -q tests/test_pathfinding.py)."""
import numpy as np
import pytest
import torch
from PIL import Image

from nbm.config import load_config
from nbm.models.video import blocks_to_frames
from nbm.tasks import get_task
from nbm.tasks.base import AccessRNG, EvalContext
from nbm.tasks.pathfinding import data as PD
from nbm.tasks.pathfinding import evaluation as E
from nbm.tasks.pathfinding import maps
from nbm.tasks.pathfinding import render as R


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    """Tiny data tables: the first rows of the paper's splits (the builder reproduces them exactly)."""
    root = tmp_path_factory.mktemp("data")
    for costs, d, splits in (("uniform", "shortest_path", ("val", "test_id")), ("terrain", "terrain", ("val",))):
        (root / d).mkdir()
        keys: set = set()
        for split in splits:
            df = PD.build_split(costs, split, keys, workers=1, limit=6)
            df.to_parquet(root / d / f"{split}.parquet", index=False)
            keys |= {PD.map_key(PD.row_map(r)) for r in df.to_dict("records")}
            if split == "val":
                df.to_parquet(root / d / "train.parquet", index=False)
    probe = PD.probe_rows()[:4]
    import pandas as pd
    pd.DataFrame(probe).to_parquet(root / "terrain" / "probe.parquet", index=False)
    return root


def make(name, family, data_root, monkeypatch):
    monkeypatch.setenv("NBM_DATA", str(data_root))
    cfg = load_config(f"configs/{name}/{family}.yaml", [f"task=nbm.tasks.pathfinding.{name}"])
    return get_task(cfg.task)(cfg)


def test_builder_reproduces_paper_rows(data_root):
    import pandas as pd
    u = pd.read_parquet(data_root / "shortest_path" / "val.parquet")
    t = pd.read_parquet(data_root / "terrain" / "val.parquet")
    p = pd.read_parquet(data_root / "terrain" / "probe.parquet")
    assert u.id.tolist()[:3] == ["sp_u_val_000000", "sp_u_val_000001", "sp_u_val_000002"]
    assert u.moves.tolist()[:3] == ["DDDRRRR", "RRU", "RU"] and u.row_seed.tolist()[:3] == [94858632, 1510918691, 1650432765]
    assert t.moves.tolist()[:3] == ["RRRRD", "DRR", "LDD"] and t.cost.tolist()[:3] == [8, 3, 5]
    assert p.moves.tolist()[:3] == ["UUU", "RRDDDDDDDDLLL", "LUUUU"] and p.label.tolist()[:3] == ["cross", "detour", "cross"]


def test_targets_are_optimal_and_tie_broken():
    for seed in range(20):
        for terrain in (False, True):
            m = maps.make_map(8, maps.FAMILIES[seed % 3], seed, terrain=terrain)
            info = maps.map_info(m)
            path = maps.path_from_moves(m["start"], info["moves"])
            D = maps.cost_to_go(m["grid"], tuple(m["goal"]), terrain)
            sc = maps.score_path(m, path, D)
            assert sc["path_valid"] and sc["optimal"] and info["cost"] == int(D[tuple(m["start"])])
    m, label = maps.probe_map(0)
    assert label in ("cross", "detour") and m["terrain"]


def test_render_decode_round_trip():
    m = maps.make_map(10, "rooms", 3, terrain=True, dmin=10)
    path = maps.path_from_moves(m["start"], maps.map_info(m)["moves"])
    D = maps.cost_to_go(m["grid"], tuple(m["goal"]), True)
    for k in range(len(path) - 1):
        fr = R.decide_picture(m, path, k)
        dec = R.decode_cells(fr, m["n"])
        assert R.mismatches(dec, m, path[k], path[:k], path[k + 1]) == 0
        cell, strength = R.read_move(fr, dec, path[k])
        assert cell == path[k + 1] and abs(strength - 1.0) < 1e-4
        assert E.step_record(m, D, path[k], path[:k], cell)["optimal"]
        faint = (R.state_picture(m, path, k) * 0.85 + fr * 0.15).astype(np.uint8)  # a faint mark still decides the move
        assert R.read_move(faint, R.decode_cells(faint, m["n"]), path[k])[0] == path[k + 1]
    blocks = R.segment_blocks(m, path, 0)
    assert blocks.shape == (17, R.S, R.S, 3) and blocks.dtype == np.uint8


def test_training_examples(data_root, monkeypatch):
    task = make("terrain", "image", data_root, monkeypatch)
    row = task.load_split("train").to_dict("records")[0]
    ex = task.image_example(row, None)
    assert ex.cond.size == ex.target.size == (R.S, R.S) and ex.prompt == "Continue."
    m, path = E.target(row)
    assert np.array_equal(np.asarray(ex.cond), R.render(m, path[0])) and list(ex.boxes[0]) == list(R.cell_box(*path[1], m["n"]))
    ex2 = task.image_example(row, AccessRNG(torch.Generator().manual_seed(0)))
    assert ex2.cond.size == (R.S, R.S)
    vt = make("terrain", "video", data_root, monkeypatch)
    v = vt.video_example(row, None)
    assert v.blocks.shape == (17, R.S, R.S, 3) and vt.prompts() == ["Continue."]
    with pytest.raises(ValueError):
        task.load_split("test_id")


# ------------------------------------------------------------------ fake models: read the picture, plan, draw
def plan(frame):
    """Decode a state picture and return (map, agent cell, trail cells, the cheapest route from the agent)."""
    a = np.asarray(frame.convert("RGB"))
    o = next(i for i in range(R.S) if a[i, i].any())  # the map's offset (black outside)
    n = (R.S - 2 * o) // R.CELL
    dec = R.decode_cells(a, n)
    grid = dec["base"].astype(np.int8).copy()
    trail = [tuple(int(x) for x in p) for p in np.argwhere(grid == maps.TRAIL)]
    pos = tuple(int(x) for x in np.argwhere(dec["centre"] == 1)[0])
    goal = tuple(int(x) for x in np.argwhere(dec["goal"])[0])
    plan_grid = grid.copy()
    plan_grid[grid == maps.TRAIL] = maps.WALL
    m = {"n": n, "grid": plan_grid, "start": pos, "goal": goal, "terrain": True}
    route = maps.target_path(plan_grid, pos, goal, terrain=True)
    draw = dict(m, grid=np.where(grid == maps.TRAIL, maps.FLOOR, grid).astype(np.int8))
    return draw, pos, trail, route


class FakeImageModel:
    def generate_seeded(self, conds, prompts, steps, seeds):
        out = []
        for c in conds:
            m, pos, trail, route = plan(c)
            out.append(Image.fromarray(R.render(m, pos, trail, mark=route[1])))
        return out


class FakeVideoModel:
    def generate_seeded(self, firsts, prompts, steps, shift, seeds):
        clips = []
        for f in firsts:
            m, pos, trail, route = plan(f)
            full = [tuple(p) for p in trail] + route  # the trail order does not matter for the pictures
            k0 = len(trail)
            blocks = [np.asarray(f.convert("RGB"))]
            k = k0
            while len(blocks) < 17:
                if k < len(full) - 1 and k < k0 + 8:
                    blocks.append(R.render(m, full[k], full[:k], mark=full[k + 1]))
                    k += 1
                    blocks.append(R.render(m, full[k], full[:k]))
                else:
                    blocks.append(R.render(m, full[k], full[:k]))
            clips.append(np.stack(blocks_to_frames(blocks)))
        return torch.from_numpy(np.stack(clips))


@pytest.mark.parametrize("name,family,split", [("shortest_path", "image", "test"), ("terrain", "image", "probe"),
                                               ("shortest_path", "video", "val"), ("terrain", "video", "val")])
def test_closed_loop_with_fake_model(name, family, split, data_root, monkeypatch):
    task = make(name, family, data_root, monkeypatch)
    df = task.load_split(split).head(4)
    model = FakeImageModel() if family == "image" else FakeVideoModel()
    ctx = EvalContext(task.cfg, model, split, steps=1, votes=1, seed=0, batch_size=3)
    res = task.evaluate(ctx, df)
    assert res.metrics["score"] == 1.0 and res.metrics["n"] == 4
    assert all(r["valid"] and r["optimal"] and r["stop"] == "goal" and not r["edits"] for r in res.records)
    if split == "probe":
        assert res.metrics["probe_correct"] == 1.0
    ctx.in_training = True
    res = task.evaluate(ctx, df)
    assert res.metrics["score"] >= 1.0


def test_noise_seed_formula():
    assert E.noise_seed(0, 5, 0) == 5 and E.noise_seed(1, 5, 2) == 1_000_003 + 5 + 2 * 100_000_007
