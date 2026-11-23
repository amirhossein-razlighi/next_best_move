"""Fast CPU tests of the billiards task: simulator, solver, rendering, the threshold-free readout, training examples, the data
builder and the evaluation loop with fake models (no GPU, no model weights, no dataset files)."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from nbm.config import load_config
from nbm.models.video import N_BLOCKS, blocks_to_frames
from nbm.tasks import make_task
from nbm.tasks import billiards as T
from nbm.tasks.base import EvalContext
from nbm.tasks.billiards import data as D
from nbm.tasks.billiards import physics as P
from nbm.tasks.billiards import render as RD
from nbm.tasks.billiards import solver as S
from nbm.tasks.billiards.decode import read_shot

REPO = __import__("pathlib").Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def _outputs(tmp_path_factory, monkeypatch):
    """The simulator is compiled into $NBM_OUTPUTS/cache: one temporary root for the whole session."""
    monkeypatch.setenv("NBM_OUTPUTS", str(tmp_path_factory.getbasetemp() / "nbm_outputs"))


@pytest.fixture(scope="module")
def rows():
    """Five solved layouts (seeds of the test split), as dataset rows."""
    out = [D.label_seed((s, "test")) for s in range(3_000_000, 3_000_008)]
    return [r for r in out if r is not None][:5]


def cfg(family, tmp_path, rows):
    pd.DataFrame(rows).to_parquet(tmp_path / "test.parquet", index=False)
    return load_config(REPO / "configs" / "billiards" / f"{family}.yaml", [f"data.dir={tmp_path}"])


def test_simulator_straight_pot():
    """Cue ball, red ball and the bottom-left pocket on one line: a firm straight shot pots the red, the cue ball stays."""
    xy = np.array([[0.5, 0.5], [0.3, 0.3]])
    s = P.simulate(xy, None, 225.0, 0.6)
    assert s["pocket"][1] == 0 and s["pocket"][0] == -1 and s["info"]["first_hit"] == 1
    assert P.shot_ok(xy, None, 225.0, 0.6) and not P.shot_ok(xy, None, 45.0, 0.6)
    pm, info = P.batch(xy, None, [225.0, 45.0], [0.6, 0.6])
    assert P.success_mask(pm).tolist() == [True, False] and P.direct_mask(pm, info).tolist() == [True, False]
    # frames are recorded without changing the outcome
    s2 = P.simulate(xy, None, 225.0, 0.6, rec_t=[0.1, 0.2])
    assert s2["pocket"] == s["pocket"] and s2["frames"].shape == (2, 2, 2)


def test_solver_label_succeeds(rows):
    for r in rows:
        xy = np.asarray(r["xy"]).reshape(-1, 2)
        assert P.shot_ok(xy, None, r["angle"], r["power"]) and r["width"] > 0
        assert r["shot_type"] in ("direct", "bank", "kick", "kiss", "combo")
        assert 2 <= len(r["colors"]) <= 6 and r["colors"][:2] == [0, 1] and r["id"] == f"test_{r['seed']}"


def test_render_and_readout_round_trip(rows):
    """The readout of a clean target picture recovers the label shot (angle within 0.05 deg, power within one track column)."""
    for r in rows:
        xy, cols = T.layout(r)
        ref = RD.render(xy, cols)
        tgt = RD.render(xy, cols, None, (r["angle"], r["power"]))
        assert ref.shape == (RD.CH, RD.CW, 3) and ref.dtype == np.uint8 and (tgt != ref).any()
        d = read_shot(tgt, ref, xy[0])
        assert d["drawn"] and abs(P.ang_diff(d["angle"], r["angle"])) < 0.05 and abs(d["power"] - r["power"]) <= 1.5 / RD.TRACK_W
        assert not read_shot(ref, ref, xy[0])["drawn"]  # nothing drawn: no shot


def test_mirror_is_a_table_symmetry():
    xy = np.array([[0.5, 0.5], [0.3, 0.3]])
    m = P.mirror_xy(xy, "lr")
    assert np.allclose(m, [[1.5, 0.5], [1.7, 0.3]]) and P.shot_ok(m, None, 315.0, 0.6)


def test_vote_circular_median():
    a, p = T.vote([(359.0, 0.5), (1.0, 0.7), (3.0, 0.6)])
    assert abs(a - 1.0) < 1e-9 and p == 0.6
    a, _ = T.vote([(10.0, 0.5), (350.0, 0.5)])
    assert abs(a % 360.0) < 1e-9


@pytest.mark.parametrize("family", ["image", "video"])
def test_training_examples(family, tmp_path, rows):
    task = make_task(cfg(family, tmp_path, rows))
    df = task.load_split("test")
    row = df.to_dict("records")[0]
    assert isinstance(row["xy"], list) and isinstance(row["colors"][0], int)
    xy, cols = T.layout(row)
    if family == "image":
        ex = task.image_example(row, None)
        assert ex.cond.size == ex.target.size == (RD.CW, RD.CH) and ex.prompt == T.PROMPT and ex.boxes == [T.cue_box(xy)]
        d = read_shot(np.asarray(ex.target), np.asarray(ex.cond), xy[0])
        assert abs(P.ang_diff(d["angle"], row["angle"])) < 0.05
    else:
        ex = task.video_example(row, None)
        assert ex.blocks.shape == (N_BLOCKS, RD.CH, RD.CW, 3) and ex.blocks.dtype == np.uint8 and task.prompts() == [T.PROMPT]
        assert np.array_equal(ex.blocks[0], RD.render(xy, cols))
        assert np.array_equal(ex.blocks[1], RD.render(xy, cols, None, (row["angle"], row["power"])))
        assert not (RD.render(xy, cols) == ex.blocks[-1]).all()  # the balls moved (the red was potted)


class FakeImageModel:
    """Draws the label shot of the layout it is given (looked up by its picture)."""

    def __init__(self, rows):
        self.by_pic = {RD.render(*T.layout(r)).tobytes(): r for r in rows}

    def label_pic(self, cond):
        r = self.by_pic[np.asarray(cond).tobytes()]
        return RD.render(*T.layout(r), None, (r["angle"], r["power"]))

    def generate_seeded(self, conds, prompts, steps, seeds):
        return [Image.fromarray(self.label_pic(c)) for c in conds]


class FakeVideoModel(FakeImageModel):
    def generate_seeded(self, firsts, prompts, steps, shift, seeds):
        clips = []
        for f in firsts:
            blocks = [np.asarray(f), self.label_pic(f)] + [np.asarray(f)] * (N_BLOCKS - 2)
            clips.append(np.stack(blocks_to_frames(blocks)))
        return torch.from_numpy(np.stack(clips))


@pytest.mark.parametrize("family,votes", [("image", 1), ("video", 1), ("image", 3)])
def test_evaluate_with_fake_model(family, votes, tmp_path, rows):
    c = cfg(family, tmp_path, rows)
    task = make_task(c)
    df = task.load_split("test")
    model = FakeImageModel(rows) if family == "image" else FakeVideoModel(rows)
    ctx = EvalContext(c, model, "test", steps=1, votes=votes, seed=0, batch_size=2, out_dir=tmp_path / "out")
    res = task.evaluate(ctx, df)
    assert res.metrics["score"] == res.metrics["success"] == 1.0 and res.metrics["n"] == len(rows)
    assert all(r["success"] and r["angle_err"] < 0.05 for r in res.records)
    assert (len(res.records[0]["samples"]) == votes) if votes > 1 else ("line_score" in res.records[0])
    assert len(list((tmp_path / "out" / "strips").glob("*.png"))) == len(rows)


def test_builder_is_deterministic(rows):
    """label_seed is a pure function of the seed; the first kept seeds of a split are its first rows."""
    again = [D.label_seed((s, "test")) for s in range(3_000_000, 3_000_008)]
    again = [r for r in again if r is not None][:5]
    assert [r["id"] for r in again] == [r["id"] for r in rows] and [r["angle"] for r in again] == [r["angle"] for r in rows]
    m = D.mirror_row((0, rows[0]))
    assert m["orig_row"] == 0 and m["variant"] in D.MIRRORS and m["id"] == f"{rows[0]['id']}_{m['variant']}"
    assert D.table_digest(pd.DataFrame(rows)) == D.table_digest(pd.DataFrame(again))
