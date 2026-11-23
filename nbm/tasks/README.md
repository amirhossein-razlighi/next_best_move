# Writing a task

A task is everything that is specific to one problem: its data, how a state and an action are drawn, how a generated picture
or clip is read back, and how it is evaluated. The core (`nbm/models`, `nbm/train.py`, `nbm/evaluate.py`) owns the two
backbones, the optimisation and the files, and talks to a task only through the interface in `nbm/tasks/base.py`.

A task lives in four places, and only there:

| path | content |
|---|---|
| `nbm/tasks/<task>/` | the package: rules / environment, rendering, data builder, decoding, metrics, evaluation loop |
| `configs/<task>/image.yaml`, `video.yaml` | the paper model's hyper-parameters |
| `docs/tasks/<task>.md` | the task, data format, how to build data / train / evaluate, expected numbers |
| `tests/test_<task>*.py` | fast CPU tests (render / mark / decode round trips) |

The registry finds the package by its directory name (`nbm/tasks/__init__.py`), so no shared file is edited to add a task.
`nbm/tasks/_example/` is a complete toy task ("mark the dark cell") used by `tests/test_core.py`; copy its structure.

## 1. The interface

```python
# nbm/tasks/<task>/__init__.py
from nbm.tasks.base import Task, ImageExample, VideoExample, EvalContext, EvalResult, AccessRNG

class MyTask(Task):
    families = ("image", "video")          # the models this task supports

    def load_split(self, split) -> pd.DataFrame: ...
    def batch_groups(self, df) -> np.ndarray: ...            # optional (image training), default: one group
    def image_example(self, row: dict, rng: AccessRNG | None) -> ImageExample: ...
    def video_example(self, row: dict, rng: AccessRNG | None) -> VideoExample: ...
    def prompts(self) -> list[str]: ...                        # video: every prompt used
    def evaluate(self, ctx: EvalContext, df: pd.DataFrame) -> EvalResult: ...

TASK = MyTask
```

`TASK(cfg)` is constructed with the full config (`self.cfg`; helpers `self.family`, `self.data_dir` = `data.dir` under
`$NBM_DATA`, `self.resolution` = `model.resolution`, `self.options` = the config's `options` section).

| method | called by | contract |
|---|---|---|
| `load_split(split)` | trainer (`"train"`, `"val"`), evaluator (`--split`, any name the task defines) | all rows of the split as a DataFrame, **in evaluation order**. The core takes prefixes: `data.train_limit` rows for training, `train.eval_n` val rows for the in-training evaluation, `--limit N` for `nbm.evaluate`. A task may expand rows (e.g. one training row per (map, segment)). |
| `batch_groups(df)` | image trainer | an int label per training row; a batch holds rows of one label only, because the prompts of one batch must share a token layout in the Qwen3-VL encoder (chess / Go: 0 = white to move, 1 = black). Per batch the label is drawn with probability proportional to its row count. Default: all 0. Video batches mix freely (fixed-length text embeddings). |
| `image_example(row, rng)` | image trainer (DataLoader workers) | `ImageExample(cond, target, prompt, boxes)`: RGB condition and target pictures of the same size (multiples of 32 px), the prompt, and the pixel boxes `(x0, y0, x1, y1)` of the highlight region (the latents they touch get `train.highlight_loss_weight`; e.g. the two squares of a chess move, the cell of a Go stone; may be empty when that weight is 1). |
| `video_example(row, rng)` | video trainer | `VideoExample(blocks, prompt, block_weights=None)`: `blocks` uint8 `(17, H, W, 3)`, block 0 = the condition picture, blocks 1..16 = the clip (each shown for 4 frames, see `nbm/models/video.py`); `block_weights` an optional `(16,)` loss weight per block (`None` = the plain mean, as in every paper run). |
| `prompts()` | video trainer / evaluator | every prompt the clips use; they are embedded once with UMT5 (cached under `$NBM_OUTPUTS/cache`) before the DiT is loaded. |
| `evaluate(ctx, df)` | evaluator, and the trainer every `train.eval_every` steps on the val prefix (a training monitor: its numeric metrics go to TensorBoard) | run the task's evaluation (one-shot, closed loop, open-loop clip, receding horizon, ...) and return `EvalResult(records, metrics)` **on every process**: `records` = one JSON-serialisable dict per item (`predictions.jsonl`), `metrics` = summary (`metrics.json`). |

`row` (training) is a plain dict `{column: value}` with Python values, like `df.to_dict("records")`.

### Per-access randomness (`rng`)

`rng` is `None` when the core needs the row's deterministic example. In training it is an `AccessRNG` whose `uniform()` is
`torch.rand(()).item()` and `randint(n)` is `int(torch.randint(0, n, ()).item())` on torch's default RNG, which torch seeds per
DataLoader worker, so the training stream is reproducible for a given seed. Examples: a random state along the target path
(`seed = f"{row_seed}:{rng.randint(2**31 - 1)}"`), one of several correct answers (`u = rng.uniform()`). For a deterministic
test pass `AccessRNG(torch.Generator().manual_seed(0))`.

## 2. Calling the models (`EvalContext`)

`evaluate` receives `ctx` (`nbm/tasks/base.py`):

| field / method | meaning |
|---|---|
| `ctx.model` | `ImageModel` (`nbm/models/image.py`) or `VideoModel` (`nbm/models/video.py`), LoRA loaded, eval mode |
| `ctx.family` | `"image"` / `"video"` |
| `ctx.steps`, `ctx.votes`, `ctx.seed`, `ctx.batch_size`, `ctx.split` | `--steps` (default `eval.steps`: 20 image / 30 video), `--votes` (default 1), `--seed`, `--batch_size` |
| `ctx.cfg` | the full config: task-specific evaluation keys come from `ctx.cfg.eval` (overridable: `eval.mode=closed_loop`) and `ctx.cfg.options` |
| `ctx.rank`, `ctx.world`, `ctx.is_main` | the process layout |
| `ctx.global_batches(items, bs)`, `ctx.my_batches(items, bs)` | consecutive global batches; global batch `k` goes to process `k % world` |
| `ctx.gather(list)` | every process's list concatenated, on every process |
| `ctx.barrier()`, `ctx.log(msg)`, `ctx.progress(total, desc, unit)` | sync; print on rank 0; one progress line per minute (with ETA) on rank 0 |
| `ctx.out_dir` | directory for pictures / strips (rank 0 only, `None` elsewhere) |
| `ctx.generate(inputs, prompts, seeds)` | one output per input with per-row noise and `ctx.steps` steps: image -> list of PIL pictures; video -> uint8 tensor `(B, 65, H, W, 3)` |

The model calls underneath, for when a task needs other noise:

```python
# image (Qwen-Image-2.1, Euler sampler)
pics = ctx.model.generate(conds, prompts, steps, noise)           # noise (B, C, 1, h, w); conds: PIL list; one DiT call per step
shape = ctx.model.latent_shape(width, height)                     # (C, 1, h, w)
# video (Wan2.2-TI2V-5B, UniPC sampler, flow shift ctx.flow_shift)
clips = ctx.model.generate(first_frames, prompts, steps, ctx.flow_shift, noise)   # noise (B, C, 17, h, w) -> uint8 (B, 65, H, W, 3)
blocks = nbm.models.video.video_to_blocks(clips[i])               # 17 PIL blocks (each = mean of its 2 central frames)
```

`generate` treats the given lists as **one batch** (one DiT call per step). Outputs depend at bfloat16 rounding level on the
batch composition, so keep a task's batching fixed (same rows together, same batch size) to reproduce its numbers. Image
batches must share a prompt layout (see `batch_groups`).

## 3. Seeds and noise per row

`nbm/models/noise.py`:

* `row_noise(seeds, shape)`: row `i` gets `torch.randn(shape)` from a CPU generator seeded with `seeds[i]`. A row's noise does not
  depend on its batch or on the number of processes. Every task except chess (image model) uses it (`ctx.generate` does).
* `batch_noise(seed, k, n, shape, device)`: the whole batch from one generator on the DiT device seeded with
  `seed * 1_000_003 + k` (`k` = index of the batch in the global batch list). The chess image model uses this.

The common seed of row `i` and vote sample `j` is `nbm.tasks.base.row_seed(seed, i, j) = seed * 1_000_003 + i + 10_000_019 * j`.
A task may use its own formula (e.g. the shortest-path closed loop adds `100_000_007 * round`); say it in a docstring.

## 4. Readout helpers

* `nbm/marks.py`: the paper's colours (`BLUE`, `RED`, `GREEN`, `ALPHA = 0.6`), `tint` (box blend, rounded once), `blend` (soft
  coverage, e.g. a disc), `fit_strengths(out, cond, boxes, colors, margin, coverages)` (per-box least squares of `out - cond`
  on `(colour - cond)`; returns blend coefficients and residuals), `to_strength` (divide by alpha, clip to [0, 1.5]). A task
  may also keep its own decoder.
* `nbm/readout.py`: `strongest` / `ranked` (the threshold-free readout: the strongest mark wins, ties to the first), and
  `majority_vote(decisions, key)` (most frequent, ties to the earliest sample, `None` ignored).

## 5. Configs

`configs/<task>/<image|video>.yaml`, keys in `nbm/config.py` (templates: `configs/_example/`). All `model` and `train` keys are
required (no hidden defaults), unknown keys there are an error; `data`, `eval` and `options` may hold task keys.

| training option | config key |
|---|---|
| `data_dir` | `data.dir` (relative to `$NBM_DATA`) |
| `train_limit` | `data.train_limit` |
| `model` | `model.base` (`Qwen-Image-2.1` / `Wan2.2-TI2V-5B-Diffusers`) |
| `resolution` | `model.resolution` |
| `split_devices` (image) | `model.split_devices` |
| `flow_shift` (video) | `model.flow_shift` |
| `rank`, `lora_alpha`, `lora_dropout`, `lora_targets` | `model.lora.{rank, alpha, dropout, targets}` |
| `output_dir` | `train.output_dir` (relative to `$NBM_OUTPUTS`, e.g. `runs/<task>_image`) |
| process count of the launch (`scale_data_order.json` `world`, or the job script) | `train.processes` |
| `train_batch_size`, `gradient_accumulation_steps` | `train.batch_size`, `train.grad_accum` |
| `learning_rate`, `weight_decay`, `lr_scheduler`, `lr_warmup_steps`, `max_train_steps`, `max_grad_norm` | `train.learning_rate`, `weight_decay`, `lr_scheduler`, `warmup_steps`, `max_steps`, `max_grad_norm` |
| `highlight_loss_weight` (image) | `train.highlight_loss_weight` |
| `gradient_checkpointing`, `seed`, `num_workers`, `log_every`, `save_every`, `dist_timeout_min` | same names under `train` |
| `eval_every`, `eval_n`, `eval_steps`, `eval_batch_size` | same names under `train` (the in-training evaluation) |
| `target_style`, `game`, task flags | `options.*` |

The core supports exactly the paper runs' recipe: image `weighting_scheme none`, video `weighting_scheme uniform`, no
guidance, no task mixing. Write floats with a dot (`1.0e-4`): YAML reads `1e-4` as a string. Then:

```bash
torchrun --standalone --nproc_per_node 1 -m nbm.train --config configs/<task>/image.yaml train.max_steps=20 train.output_dir=runs/smoke
torchrun --standalone --nproc_per_node 1 -m nbm.evaluate --config configs/<task>/image.yaml --checkpoint $NBM_OUTPUTS/runs/smoke/checkpoint-20 --split val --limit 4
```

(with `model.split_devices: true` each image process needs 2 GPUs.)

## 6. Tests

`tests/test_<task>*.py`, CPU only, < 2 min: render -> mark -> decode round trips, the evaluation loop with a fake model (see
`tests/test_core.py`: `FakeImageModel` / `FakeVideoModel` return the correct target). Run `python -m pytest -q tests/test_<task>.py`
from the repository root.
