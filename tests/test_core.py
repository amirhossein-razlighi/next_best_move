"""Fast CPU tests of the core: configs, marks / readout, noise, batch streams, clip layout, registry, and the toy task's
training examples and evaluation loop with fake models (no GPU, no model weights)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from nbm import marks, paths, readout
from nbm.config import Config, apply_override, from_dict, load_config
from nbm.data import GroupBatchSampler, ImageTrainSet, ShardedBatchSampler, VideoTrainSet, collate_image, collate_video
from nbm.models.noise import batch_noise, row_noise
from nbm.models.video import HOLD, N_BLOCKS, N_FRAMES, blocks_to_frames, frames_to_blocks, video_to_blocks
from nbm.tasks import available_tasks, get_task, make_task
from nbm.tasks import _example as EX
from nbm.tasks.base import AccessRNG, EvalContext, RowStore, jsonable, row_seed, write_result


def toy_config(family="image", **over) -> Config:
    cfg = {
        "task": "_example", "family": family,
        "data": {"dir": "toy", "train_limit": 100},
        "model": {"base": "unused", "resolution": 256, "lora": {"rank": 4, "alpha": 4.0, "dropout": 0.0, "targets": ["to_q"]},
                  **({"split_devices": False} if family == "image" else {"flow_shift": 3.0})},
        "train": {"output_dir": "runs/toy", "processes": 1, "batch_size": 2, "grad_accum": 1, "learning_rate": 1.0e-4,
                  "weight_decay": 1.0e-4, "lr_scheduler": "cosine", "warmup_steps": 0, "max_steps": 10, "max_grad_norm": 1.0,
                  "gradient_checkpointing": False, "seed": 0, "num_workers": 0, "log_every": 1, "save_every": 5,
                  "eval_every": 5, "eval_n": 8, "eval_steps": 2, "eval_batch_size": 4, "dist_timeout_min": 10, "resume": None,
                  **({"highlight_loss_weight": 1.0} if family == "image" else {})},
        "eval": {"steps": 20, "batch_size": 4, "seed": 0},
        "options": {"n_val": 8, "n_test": 8},
    }
    for k, v in over.items():
        cfg[k] = v
    return cfg


def write_cfg(tmp_path, cfg) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


# ------------------------------------------------------------------ config / paths
def test_config_overrides_and_validation(tmp_path):
    p = write_cfg(tmp_path, toy_config())
    cfg = load_config(p, ["train.max_steps=20", "train.learning_rate=2.0e-4", "eval.mode=closed_loop", "options.style=step"])
    assert cfg.train.max_steps == 20 and cfg.train.learning_rate == 2e-4
    assert cfg.eval.mode == "closed_loop" and cfg.options.style == "step"
    assert cfg.get_path("eval.votes", 1) == 1 and cfg.get_path("model.lora.rank") == 4
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(p, ["train.typo_key=1"])
    bad = toy_config()
    del bad["train"]["seed"]
    with pytest.raises(ValueError, match="missing"):
        load_config(write_cfg(tmp_path, bad))
    with pytest.raises(ValueError, match="family"):
        load_config(p, ["family=audio"])
    c = Config()
    apply_override(c, "a.b.c=[1, 2]")
    assert c.a.b.c == [1, 2]


def test_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("NBM_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("NBM_MODELS", str(tmp_path / "m"))
    assert paths.data_path("chess") == tmp_path / "d" / "chess"
    assert paths.model_path("Qwen-Image-2.1") == tmp_path / "m" / "Qwen-Image-2.1"
    assert paths.data_path("/abs/x") == Path("/abs/x")


# ------------------------------------------------------------------ marks / readout
def test_marks_fit_recovers_strengths():
    cond = EX.render(5)
    boxes = [EX.cell_box(k) for k in range(16)]
    tgt = marks.tint(cond, [(boxes[5], marks.RED), (boxes[9], marks.BLUE)])
    coef, res = marks.fit_strengths(marks.as_array(tgt), marks.as_array(cond), boxes, [marks.RED, marks.BLUE], margin=4)
    s = marks.to_strength(coef)
    assert abs(s[5, 0] - 1) < 0.02 and abs(s[9, 1] - 1) < 0.02
    assert np.delete(s[:, 0], 5).max() < 0.02 and np.delete(s[:, 1], 9).max() < 0.02
    assert res.max() < 1.0
    # a soft mark through its coverage map
    cov = np.zeros((256, 256), np.float32)
    cov[16:48, 16:48] = 1.0
    dotted = marks.blend(cond, cov, marks.GREEN)
    c2, _ = marks.fit_strengths(marks.as_array(dotted), marks.as_array(cond), boxes[:1], [marks.GREEN], coverages=[cov])
    assert abs(marks.to_strength(c2)[0, 0] - 1) < 0.02
    assert marks.to_strength(np.array([-1.0, 5.0])).tolist() == [0.0, marks.MAX_STRENGTH]


def test_readout_rules():
    assert readout.ranked([0.1, 0.9, 0.9, 0.5]) == [1, 2, 3, 0]  # ties keep the order
    assert readout.strongest([0.1, 0.9, 0.9]) == 1
    assert readout.strongest([0.1, 0.9, 0.3], candidates=[0, 2]) == 2
    assert readout.majority_vote(["a", "b", "b", "a"]) == "a"  # tie -> the earliest
    assert readout.majority_vote([None, "c", None]) == "c"
    assert readout.majority_vote([None, None]) is None
    assert readout.majority_vote(["e7e8q", "e7e8", "d2d4"], key=lambda m: m[:4]) == "e7e8q"


# ------------------------------------------------------------------ noise
def test_row_noise_is_per_row():
    a = row_noise([3, 7, 11], (2, 1, 4, 4))
    b = row_noise([11, 3], (2, 1, 4, 4))
    assert a.shape == (3, 2, 1, 4, 4) and a.dtype == torch.float32
    assert torch.equal(a[0], b[1]) and torch.equal(a[2], b[0])
    assert torch.equal(a[1], torch.randn((2, 1, 4, 4), generator=torch.Generator().manual_seed(7)))
    n1, n2 = batch_noise(0, 3, 2, (2, 1, 4, 4), "cpu"), batch_noise(0, 3, 2, (2, 1, 4, 4), "cpu")
    assert torch.equal(n1, n2) and not torch.equal(n1, batch_noise(0, 4, 2, (2, 1, 4, 4), "cpu"))
    assert row_seed(1, 5, 2) == 1_000_003 + 5 + 2 * 10_000_019


# ------------------------------------------------------------------ batch streams
def take(it, n):
    return [next(it) for _ in range(n)]


def test_group_sampler_homogeneous_sharded_resumable():
    groups = np.array([0] * 30 + [1] * 10)
    one = take(iter(GroupBatchSampler(groups, 4, seed=0)), 40)
    assert all(len({int(groups[i]) for i in b}) == 1 for b in one)
    frac1 = np.mean([groups[b[0]] == 1 for b in take(iter(GroupBatchSampler(groups, 4, seed=1)), 2000)])
    assert abs(frac1 - 0.25) < 0.04
    # two processes split the same global stream
    r0, r1 = take(iter(GroupBatchSampler(groups, 4, 0, 0, 2)), 20), take(iter(GroupBatchSampler(groups, 4, 0, 1, 2)), 20)
    assert r0 == one[0::2] and r1 == one[1::2]
    # resume: skipping the global batches of the steps already trained continues the stream
    assert take(iter(GroupBatchSampler(groups, 4, 0, 0, 2, skip=6)), 5) == one[6::2][:5]


def test_sharded_sampler():
    one = take(iter(ShardedBatchSampler(10, 3, seed=0)), 12)
    assert all(len(b) == 3 and len(set(b)) == 3 for b in one)
    assert sorted(i for b in one[:3] for i in b) == sorted(set(i for b in one[:3] for i in b))  # no repeats within a pass
    assert take(iter(ShardedBatchSampler(10, 3, 0, 1, 2)), 4) == one[1::2][:4]
    assert take(iter(ShardedBatchSampler(10, 3, 0, 0, 1, skip=5)), 3) == one[5:8]


# ------------------------------------------------------------------ models (CPU-only parts)
def test_box_mask_latent_cells():
    from nbm.models.image import box_mask
    m = box_mask([[(32, 64, 64, 96)], [(0, 0, 8, 8), (250, 250, 256, 256)]], 16, 16, 256, 256, "cpu")
    assert m.shape == (2, 1, 1, 16, 16)
    assert m[0, 0, 0].sum() == 4 and m[0, 0, 0, 4:6, 2:4].all()
    assert m[1, 0, 0, 0, 0] == 1 and m[1, 0, 0, 15, 15] == 1 and m[1].sum() == 2


def test_clip_layout_round_trip():
    blocks = [Image.new("RGB", (64, 64), (10 * k, 5 * k, 200 - 10 * k)) for k in range(N_BLOCKS)]
    frames = blocks_to_frames(blocks)
    assert len(frames) == N_FRAMES == 1 + HOLD * (N_BLOCKS - 1)
    back = frames_to_blocks(frames)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(blocks, back))
    vid = np.stack([np.asarray(f) for f in frames])
    assert np.array_equal(np.asarray(video_to_blocks(vid)[7]), np.asarray(blocks[7]))


# ------------------------------------------------------------------ registry + toy task
def test_registry():
    assert "_example" not in available_tasks()
    assert get_task("_example") is EX.DarkCell
    assert get_task("nbm.tasks._example") is EX.DarkCell
    with pytest.raises(ValueError, match="unknown task"):
        get_task("no_such_task")


def test_toy_training_examples():
    task = make_task(from_dict(toy_config()))
    df = task.load_split("train")
    row = RowStore(df)[0]
    ex = task.image_example(row, None)
    assert EX.decode(ex.target, ex.cond)["cell"] == row["cell"] and ex.boxes == [EX.cell_box(row["cell"])]
    ex2 = task.image_example(row, AccessRNG(torch.Generator().manual_seed(0)))
    ex3 = task.image_example(row, AccessRNG(torch.Generator().manual_seed(0)))
    assert np.array_equal(np.asarray(ex2.cond), np.asarray(ex3.cond))
    batch = collate_image([ImageTrainSet(task, df)[i] for i in range(3)])
    assert batch["cond"].shape == (3, 4, 256, 256) and len(batch["boxes"]) == 3 and len(batch["prompts"]) == 3
    vtask = make_task(from_dict(toy_config("video")))
    vb = collate_video([VideoTrainSet(vtask, df)[i] for i in range(2)])
    assert vb["blocks"].shape == (2, N_BLOCKS, 256, 256, 3) and vb["blocks"].dtype == torch.uint8 and vb["block_weights"] is None


class FakeImageModel:
    """A perfect 'model': marks the dark cell it finds in the condition picture."""

    def generate_seeded(self, conds, prompts, steps, seeds):
        out = []
        for c in conds:
            dark = [k for k in range(16) if np.asarray(c)[EX.cell_box(k)[1] + 8, EX.cell_box(k)[0] + 8].tolist() == list(EX.DARK)]
            out.append(marks.tint(c, [(EX.cell_box(dark[0]), marks.RED)]))
        return out


class FakeVideoModel(FakeImageModel):
    def generate_seeded(self, firsts, prompts, steps, shift, seeds):
        decides = FakeImageModel.generate_seeded(self, firsts, prompts, steps, seeds)
        clips = [blocks_to_frames([f, d] + [EX.render(None)] * (N_BLOCKS - 2)) for f, d in zip(firsts, decides)]
        return torch.from_numpy(np.stack([np.stack([np.asarray(x) for x in c]) for c in clips]))


@pytest.mark.parametrize("family", ["image", "video"])
def test_toy_evaluate_with_fake_model(family, tmp_path):
    cfg = from_dict(toy_config(family))
    task = make_task(cfg)
    df = task.load_split("val")
    model = FakeImageModel() if family == "image" else FakeVideoModel()
    ctx = EvalContext(cfg, model, "val", steps=1, votes=3, seed=0, batch_size=3)
    res = task.evaluate(ctx, df)
    assert res.metrics["score"] == 1.0 and res.metrics["n"] == len(df) == 8
    assert all(r["samples"] == [r["cell"]] * 3 for r in res.records)
    write_result(res, tmp_path / "out", {"split": "val", "arr": np.arange(3)})
    m = json.load(open(tmp_path / "out" / "metrics.json"))
    assert m["arr"] == [0, 1, 2] and m["score"] == 1.0 and (tmp_path / "out" / "_DONE").exists()
    assert len((tmp_path / "out" / "predictions.jsonl").read_text().splitlines()) == 8


def test_eval_context_sharding():
    cfg = from_dict(toy_config())
    ctxs = [EvalContext(cfg, None, "val", 1, 1, 0, 4, rank=r, world=3) for r in range(3)]
    got = sorted(i for c in ctxs for _, b in c.my_batches(23) for i in b)
    assert got == list(range(23))
    assert [k for k, _ in ctxs[1].my_batches(23)] == [1, 4]
    assert jsonable({"a": np.float32(1.5), "b": (np.int64(2), torch.tensor([1, 2]))}) == {"a": 1.5, "b": [2, [1, 2]]}


@pytest.mark.parametrize("family", ["image", "video"])
def test_example_configs_validate(family):
    cfg = load_config(Path(__file__).parents[1] / "configs" / "_example" / f"{family}.yaml")
    assert cfg.family == family and make_task(cfg).family == family
