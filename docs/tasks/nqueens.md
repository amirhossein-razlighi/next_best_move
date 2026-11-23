# N-queens

Paper: Appendix A.6. Image model only (Qwen-Image-2.1 + LoRA). Code: `nbm/tasks/nqueens/`, config: `configs/nqueens/image.yaml`.

## Task

A partly filled N-queens board (N = 5 to 12) with k non-attacking queens. The model places one more queen, and the placement is
**correct** if the board can still be completed to a full solution (no two queens share a row, column or diagonal). Any such
"extendable" cell counts, in any row. The rules are never stated: the prompt is `Continue.`.

* **Picture** (`render.py`): 384 x 384 px, a fixed canvas of 12 x 12 cells of 32 px. The N x N board is a checkerboard in the
  top-left corner; the other canvas cells are dark grey (off-board). A queen is a black crown glyph.
* **Target**: the same picture with the next queen drawn as a red glyph. The training label is the topmost empty row, in the
  column of the solution the board was drawn from.
* **Readout** (`render.decode`, threshold-free): for each of the 144 canvas cells, the least-squares strength of a red glyph
  over the glyph's interior pixels, relative to the input picture. The strongest cell is the decision, whatever its strength.
  The readout knows neither the board size nor the rules; `rules.classify` scores the decision as `correct`, `dead` (legal but
  no completion), `attacked`, `occupied` or `off_board`. *Legal* means correct or dead.

## Data

Build it (deterministic, about 30 s on one CPU, no external tools: labels come from exact enumeration of all solutions):

```bash
python -m nbm.tasks.nqueens.data build --out $NBM_DATA/nqueens
```

This writes the tables below as parquet files, plus `build_stats.json`, `digest.json` (one sha256 per table over its
(N, board, label) rows) and `checks.json` (the leakage and label checks; the build fails if one fails). The paper's build has
these digests:

| table | rows | content | digest (first 16 hex) |
|---|---|---|---|
| `train` | 200,071 | N = 5..10, boards of training-family solutions | `8b7745b9c0fe9abe` |
| `seen` | 1,000 | N = 8..10, training families, boards (and their 8 symmetric images) never trained on | `d7b446053b046391` |
| `val` | 1,000 | N = 8..10, validation families | `73d0aaa24d3cda9e` |
| `test_a` | 2,000 | N = 8..10, held-out test families | `4d2d68ba3c8dc56a` |
| `test_b` | 2,000 | N = 11, 12 (never-trained sizes) | `5965284b6d702e95` |
| `test_hard` | 1,000 | N = 8..12, "trap" and "few" boards (see below) | `acfe4e9f4ccb1a3d` |

**Splits by solution family.** A family is a solution together with its rotations and reflections. The families of each N are
ordered by a hash and the first ones are held out: 2 validation + 2 test of the 12 families at N = 8, 8 + 8 of 46 at N = 9 and
15 + 15 of 92 at N = 10 (N = 5, 6, 7 are all training). No training board is contained in a held-out solution, so no
evaluation board, nor any of its rotations or reflections, occurs in training. N = 11 and 12 are never trained on.
`test_hard`: *trap* = the row to fill has at least 3 safe cells of which exactly one is extendable, and at most half of all
safe cells are extendable; *few* = a unique completion with at least 3 queens still to place.

**Columns**: `id`, `table`, `N`, `k`, `state` (one character per row: `.` = empty, otherwise the base-36 column of its queen),
`label_r`, `label_c`, `src_sol`, `src_class`, `class_split`, `n_comp` (number of completions), `comp_train` / `comp_val` /
`comp_test`, `n_ext` / `n_safe` / `n_empty` (extendable / non-attacked empty / empty cells), `ext` (extendable cells as
`12 r + c`), `row_ext` / `row_safe`, `subtype` (`trap` / `few` in `test_hard`).

## Train

```bash
torchrun --standalone --nproc_per_node 4 -m nbm.train --config configs/nqueens/image.yaml
```

Rank-64 LoRA, learning rate 1e-4 (cosine, 200 warm-up steps), 24,000 steps, global batch 8, highlight weight 1.
`model.split_devices: true` puts the text encoder + VAE and the DiT of each process on two 24 GB GPUs, so 4 processes need one
node of 8 GPUs. The paper run used 8 processes x batch 1 on two such nodes; the config reaches the same global batch with
`train.grad_accum: 2` (every optimizer step sees the same 8 training rows as the paper run; only the per-process noise draws
differ). On 16 GPUs use `train.processes=8 train.grad_accum=1`. The paper run took about 10 hours on 16 A5000 GPUs.

## Evaluate

```bash
torchrun --standalone --nproc_per_node 1 -m nbm.evaluate --config configs/nqueens/image.yaml \
    --checkpoint checkpoints/nqueens_image --split test_a --steps 1            # the paper's readout (one Euler step)
torchrun --standalone --nproc_per_node 1 -m nbm.evaluate --config configs/nqueens/image.yaml \
    --checkpoint checkpoints/nqueens_image --split test_a                     # full sampling, 20 steps
```

Splits: `val`, `seen`, `test_a`, `test_b`, `test_hard` (and `train`). One generation per board, batches of 8 consecutive
boards (`eval.batch_size`), noise of board i from seed `seed * 1,000,003 + i`; `--votes n` takes the majority cell of n
samples. `metrics.json` holds `acc` (= correct, the score), `legal`, the share of each category (`cat/...`), `acc/N<n>` per
board size, `random_safe` (the expected accuracy of a uniformly random non-attacked empty cell on the same boards) and
`faint` (share of decisions whose red mark is weaker than half a glyph: one-step outputs are blurry, the readout still
decides). `predictions.jsonl` has one record per board (decided cell, category, mark strength).

## Expected numbers (paper checkpoint, seed 0)

| split | boards | steps | correct (%) | legal (%) |
|---|---|---|---|---|
| `test_a` (held-out families) | 2,000 | 1 | 85.2 | 99.4 |
| `test_a` | 2,000 | 20 | 78.8 | |
| `test_b` (N = 11, 12) | 2,000 | 1 | 79.2 | 97.0 |
| `test_hard` | 1,000 | 1 | 62.0 | 97.3 |
| `seen` (training families) | 1,000 | 1 | 79.1 | 98.6 |
| `val` | 1,000 | 1 | 86.1 | 99.5 |


