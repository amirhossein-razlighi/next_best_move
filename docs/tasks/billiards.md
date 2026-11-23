# Billiards

Paper: Table 2 ("Billiards") and Appendix A.5. Image model (Qwen-Image-2.1 + LoRA) and video model (Wan2.2-TI2V-5B + LoRA).
Code: `nbm/tasks/billiards/`, configs: `configs/billiards/image.yaml`, `configs/billiards/video.yaml`.

## Task

"Pot the red": a table holds the cue ball, a red target ball and up to four other balls. The model chooses one shot, a
continuous cue direction and power, and succeeds if the red ball is potted and the cue ball is not. Physics is never given:
the prompt is `Continue.` and the models learn from solver-labelled shots only.

* **Simulator** (`physics.py`, C source `sim/bilsim.c`): deterministic 2D pool without spin. Table 2 x 1 (table widths),
  ball radius 0.03, constant rolling deceleration, equal-mass ball collisions (restitution 0.95), cushions (restitution
  0.8), six point pockets (capture radius 0.085 corner / 0.07 side). A shot is (angle in degrees, power in [0, 1]); full
  power rolls a lone cue ball 6 table widths. The C file is compiled with `gcc` the first time it is used, into
  `$NBM_OUTPUTS/cache/billiards/` (flags without fused multiply-add or fast math: bit-identical results on every x86-64
  machine). Only `gcc` is needed.
* **Picture** (`render.py`): 512 x 288 px (`model.resolution: 288` is the height). Brown rails, green cloth, black pockets,
  flat balls with a thin rim, a power track below the table.
* **Shot** (the mark M(x, a)): a cyan aiming line from the cue ball to where its centre would first meet a cushion, drawn
  *under* the balls; a magenta cue stick behind the cue ball on the same axis; the power track filled cyan up to the power.
  Image model: input = the layout, target = the layout with the label shot. Video model: a 17-block clip = the layout, the
  DECIDE block (= the image target), the simulated balls at t = 0.2, 0.4, ..., 2.8 s, and the resting state. Only the DECIDE
  block is read; its shot is executed (the imagined motion is not scored).
* **Readout** (`decode.py`, threshold-free): per pixel the least-squares alpha of cyan and magenta over the input picture;
  the angle is the one whose rendered line + stick correlates best with these alpha maps (coarse 0.25 deg grid, then
  refined to 0.002 deg), the power is the bar length whose template fits the cyan alpha of the track best. No pixel or
  length threshold; an output with no positive cyan / magenta evidence at all is a miss. On clean renders the angle error
  has a median of about 0.003 deg. The decoded shot is executed in the simulator.

Metrics (one shot per layout): `success` = red potted and cue ball not (paper: "ball potted"; `score`), `red_potted`,
`scratch` (cue ball potted), `angle_err_median` / `power_err_median` (against the label shot), `success_hard` /
`success_easy` (rows with / without the `hard` flag).

## Data

The data is built by the repository itself (no external tool): random layouts labelled by the offline solver
(`solver.py`): every angle on a 0.25 deg grid at 9 powers (12,960 simulated shots); the label is the centre of the widest
successful angle range that is robust to +- 0.05 power, at a firm but controlled power, re-centred at that power. Shots
include direct pots, banks, kicks, kisses and combinations.

```bash
python -m nbm.tasks.billiards.data --out $NBM_DATA/billiards --workers 32          # ~9 CPU-hours (~20 min on 32 cores)
python -m nbm.tasks.billiards.data --out /tmp/bil_sample --limit 20 --workers 8    # quick sample: the first 20 rows per split
```

The build is deterministic: layout of seed s = `numpy.random.default_rng(s)`, kept iff the solver finds a successful shot.
It writes `<split>.parquet` and `digest.json` (sha256 per split over every row's positions and label shot). The paper's data:

| split | rows | content | digest (first 16 hex) |
|---|---|---|---|
| `train` | 200,000 | the first kept layouts of the seeds 1,000,000+ | `9edaca984df42169` |
| `val` | 1,000 | seeds 2,000,000+ | `5aad29e205260548` |
| `test` | 2,000 | seeds 3,000,000+ (Table 2) | `97f43d4e334905ea` |
| `test_hard` | 1,000 | the first HARD layouts of the seeds 4,000,000+ (Appendix A.5) | `79b0087d5d3f6095` |
| `test_mirror` | 2,000 | every test layout mirrored once and solved again (Appendix A.5) | `fbd79b27fd7a2479` |

**Hard** = the solver finds no *direct* pot anywhere on its grid of 12,960 shots (a direct pot: the cue ball hits the red
first without touching a cushion, and the red drops without a cushion or another contact), so a bank, kick or combination
is needed. No other criterion. 387 of the 2,000 `test` layouts are hard; `test_hard` holds only hard layouts.
**Mirror**: per test row i an exact table symmetry (left-right 664, top-bottom 635, both 701 rows), drawn with
`default_rng([i, 99])`; `orig_row` = i, `variant` = the symmetry. Layout positions are continuous: no layout is shared
between splits, also up to the table's symmetries.

**Columns**: `id`, `seed`, `split`, `k_others` (0-4 other balls), `xy` (2n floats: cue, red, the others, table widths, y up),
`colors` (n ints: 0 cue, 1 red, 2 yellow, 3 blue, 4 purple, 5 orange), and the solver label: `angle`, `power`, `width` (the
successful angle range at the label power), `ang_lo`, `ang_hi`, `grid_power`, `grid_width`, `power_robust`,
`grid_power_robust`, `p_lo`, `p_hi`, `pocket`, `shot_type` (direct / bank / kick / kiss / combo), `n_succ`, `frac_succ`,
`n_direct`, `hard`, `cue_final`, `n_potted_obj`, `t_rest`; `test_mirror` also `orig_row`, `variant`.

## Train

```bash
torchrun --standalone --nproc_per_node 4 -m nbm.train --config configs/billiards/image.yaml   # 4 processes x 2 GPUs
torchrun --standalone --nproc_per_node 8 -m nbm.train --config configs/billiards/video.yaml   # 8 processes x 1 GPU
```

Both: rank-64 LoRA, learning rate 1e-4 (cosine, 200 warm-up steps), global batch 16; image 20,000 steps (1.6 passes over the
200,000 layouts), video 12,000 steps; about 32 h (image) and 22 h (video) on one node of 8 A5000 GPUs.

## Evaluate

```bash
torchrun --standalone --nproc_per_node 1 -m nbm.evaluate --config configs/billiards/image.yaml \
    --checkpoint checkpoints/billiards_image --split test                 # full sampling, 20 Euler steps (2 GPUs per process)
torchrun --standalone --nproc_per_node 1 -m nbm.evaluate --config configs/billiards/image.yaml \
    --checkpoint checkpoints/billiards_image --split test --steps 1       # one step
torchrun --standalone --nproc_per_node 4 -m nbm.evaluate --config configs/billiards/video.yaml \
    --checkpoint checkpoints/billiards_video --split test [--steps 1]     # 30 UniPC steps / one step
```

Splits: `test` (Table 2), `test_hard`, `test_mirror` (Appendix A.5), `val`, `train`. One generation per layout in batches of
consecutive rows (image 8, video 2); the noise of row i comes from the seed `seed * 1,000,003 + i`, so the result does not
depend on the number of processes. `--votes n`: sample k uses the seed `(seed + k) * 1,000,003 + i` (sample k is the
single-sample evaluation with seed + k), and the n decoded shots are combined into one, the circular median of the angles
and the median power (the paper's billiards vote). `predictions.jsonl` has one record per layout: the decoded `angle`,
`power`, `line_score`, `bar_score`, `drawn`, the outcome (`success`, `red_potted`, `scratch`, `pocket`, `angle_err`,
`power_err`) and the label (`label_angle`, `label_power`, `label_pocket`, `hard`, `shot_type`); with votes also `samples`.
The main process writes pictures of the first rows to `strips/` (image: layout / label shot / output; video: label shot,
then the 17 generated blocks).

## Expected numbers (released checkpoints, seed 0, threshold-free readout)

Ball potted (%):

| model | split | full sampling | one step |
|---|---|---|---|
| image | `test` (2,000) | **18.6** (372) | **28.0** (560) |
| video | `test` (2,000) | **34.6** (693) | **47.9** (959) |
| image | `test_hard` (1,000) | 3.2 (32) | 4.2 (42) |
| video | `test_hard` (1,000) | 3.8 (38) | 6.0 (60) |
| image | `test_mirror` (2,000) | 19.25 (385) | 28.25 (565) |
| video | `test_mirror` (2,000) | 34.8 (696) | 49.1 (982) |

Bold = the paper's Table 2.
