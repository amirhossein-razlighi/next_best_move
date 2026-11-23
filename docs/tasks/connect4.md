# Connect Four

Paper: Table 2 and Appendix A.3. Image model (Qwen-Image-2.1 + LoRA) and video model (Wan2.2-TI2V-5B + LoRA). Code:
`nbm/tasks/connect4/`, configs: `configs/connect4/image.yaml`, `configs/connect4/video.yaml`.

## Task

Standard 7 x 6 Connect Four. The model chooses a column for the side to move by drawing the disc where it lands. That a disc
falls to the lowest free row is never stated: the prompt is `Continue.`, with no rule text and no legend.

* **Picture** (`render.py`): 256 x 256 px. A dark slate board of 7 x 6 cells of 32 px (2 x 2 image tokens, 2 x 2 video latent
  cells); Red moves first, Blue second; a disc in the top margin shows whose turn it is (none once the game is over).
* **Move** (the mark M(x, a)): the mover's disc drawn in the landing cell of the chosen column, the cell's square tinted green
  around it. Image model: input = the position, target = this picture for the training label (an exact-best column, the most
  central one on ties). Video model: a 17-block clip of the solver's best line from the position, both sides, up to 8 plies:
  block 0 = the position, then per ply a DECIDE block (the move mark) and an IMAGINE block (the board after the move, turn
  marker flipped); when the game ends the final board is held. The decision is the first DECIDE block.
* **Readout** (`decode.py`, threshold-free): per cell, the least-squares strength of the green tint on the cell's square
  outside the hole, relative to the input picture. The strongest cell is the move, whatever its strength. The move is
  **valid** iff its column is not full and the cell is that column's landing row (a floating disc is an impossible move and
  counts as wrong). For the video model the record also reads the imagined continuation block by block against a bookkept
  board (`decode.read_line`: are the imagined boards right, are the later moves legal); later moves count when their
  strongest mark exceeds 0.5, and the reading stops at the first failure.

Metrics (one decision per position, labels from the exact solver, any exact-best column counts):

| metric | meaning |
|---|---|
| `legal_landing` | the move is valid (paper: "legal, correct landing row") |
| `keeps_value` | valid and value-preserving: a won position stays won, a drawn one drawn (paper: "keeps the win") |
| `best` | valid and an exact-best column (fastest win / slowest loss) |
| `*_hard` | the same on the rows with `hard` = true (936 of the 1,000 `test_hard` positions are won) |
| video: `imagined_plies`, `imagined_imagine_ok`, `imagined_later_legal`, `imagined_stops` | the imagined continuation |

## Data

Labels come from Pascal Pons' exact solver with its 7 x 6 opening book, which are **not** part of this repository (the
solver is AGPL-3.0): <https://github.com/PascalPons/connect4> (commit `d6ba50d8`) and the book
<https://github.com/PascalPons/connect4/releases/download/book/7x6.book> (sha256 `f346cd44...3aec53`). The build script
clones and downloads both and compiles the small batch driver `nbm/tasks/connect4/pons/c4board.cpp` (one board per line in,
the 7 move scores out); it needs git, curl and g++. The solver is only needed to build the data; evaluation reads the labels
from the data files.

```bash
bash nbm/tasks/connect4/pons/build.sh               # -> ./tools/connect4 (or $NBM_CONNECT4_SOLVER): bin/c4board, book/7x6.book
python -m nbm.tasks.connect4.data --out $NBM_DATA/connect4 --workers 32
```

The build is deterministic (about 10 minutes with 32 workers) and writes `train`, `val`, `test`, `test_hard` (`.parquet`)
and `digest.json` (sha256 per split over its (board, label, line) rows). The paper's data has these digests:

| split | rows | content | digest (first 16 hex) |
|---|---|---|---|
| `train` | 200,000 | the first distinct critical positions of the self-play games with seeds 1,000,000+ | `62ae3a56db6cb392` |
| `val` | 1,000 | one critical position per game, seeds 2,000,000+ | `3e4a4d578d99eb55` |
| `test` | 2,000 | one critical position per game, seeds 3,000,000+ | `1f7df4f04703513d` |
| `test_hard` | 1,000 | one hard position per game, seeds 4,000,000+ | `7b02e47e55ed27c3` |

**Games**: each player plays a uniformly random legal column with a probability drawn from {0, 0.05, 0.1, 0.2, 0.35, 0.5},
and otherwise (with equal chance) a random exact-best or a random value-preserving column. **Critical** positions (not every
column has the same game-theoretic value) are the only ones kept. **Hard** = critical with neither an immediate win for the
mover nor an immediate winning threat of the opponent. Positions are deduplicated up to the left-right mirror within and
across splits, and each evaluation split takes at most one position per game.

**Columns**: `id`, `split`, `game_seed`, `game_ply`, `eps_x` / `eps_o` (the players' noise levels), `board` (42 characters,
column-major, each column bottom to top: `.` empty, `x` Red, `o` Blue), `ply`, `mover`, `scores` (the 7 exact move scores
for the side to move in Pons' convention, > 0 win, 0 draw, < 0 loss, larger = faster win / slower loss, -1000 = full column),
`value` (+1 / 0 / -1), `best`, `choice` (the training label), `vp` (value-preserving columns), `critical`, `kind` (`win` /
`block` / `quiet`), `hard`, `n_legal`, `line` (the video's continuation: both sides play the canonical best column, up to
8 plies), `line_len`, `line_end` (`win_x` / `win_o` / `draw` / `open`).

## Train

```bash
torchrun --standalone --nproc_per_node 4 -m nbm.train --config configs/connect4/image.yaml   # 4 processes x 2 GPUs
torchrun --standalone --nproc_per_node 8 -m nbm.train --config configs/connect4/video.yaml   # 8 processes x 1 GPU
```

Both: rank-64 LoRA, learning rate 1e-4 (cosine, 200 warm-up steps), global batch 16, one pass over about the 200,000 training
positions (image 12,000 steps, video 10,000 steps), about 8 hours each on one node of 8 A5000 GPUs.

## Evaluate

```bash
torchrun --standalone --nproc_per_node 1 -m nbm.evaluate --config configs/connect4/image.yaml \
    --checkpoint checkpoints/connect4_image --split test_hard                  # full sampling, 20 Euler steps
torchrun --standalone --nproc_per_node 1 -m nbm.evaluate --config configs/connect4/image.yaml \
    --checkpoint checkpoints/connect4_image --split test_hard --steps 1        # one step
torchrun --standalone --nproc_per_node 1 -m nbm.evaluate --config configs/connect4/video.yaml \
    --checkpoint checkpoints/connect4_video --split test_hard [--steps 1]      # 30 UniPC steps / one step
```

Splits: `test_hard` (the paper's table), `test`, `val`, `train`. One generation per position in batches of consecutive rows
(image 8, video 2), noise of row i from seed `seed * 1,000,003 + i`, so the result does not depend on the number of processes;
`--votes n` takes the majority of the valid sampled columns. `predictions.jsonl` has one record per position: the decided
column `col` (null if no valid move), the strongest mark (`mark_col`, `mark_row`, `mark_strength`, `legal`, `landing_ok`),
`best`, `keeps_value`, and for the video model the imagined line (`imagined`). The first 8 rows' pictures go to `strips/`.

## Expected numbers (released checkpoints, seed 0, threshold-free readout)

| model | split | steps | legal_landing | keeps_value | best |
|---|---|---|---|---|---|
| image | `test_hard` (1,000) | 20 | **100.0** | **67.5** | 48.0 |
| image | `test_hard` | 1 | 100.0 | **76.4** | 57.8 |
| video | `test_hard` (1,000) | 30 | **99.6** | **64.5** | 44.4 |
| video | `test_hard` | 1 | 100.0 | **72.0** | 53.1 |
| image | `test` (2,000) | 20 / 1 | 100.0 / 100.0 | 73.3 / 80.4 | 56.5 / 64.0 |
| video | `test` (2,000) | 30 / 1 | 99.9 / 100.0 | 65.8 / 73.6 | 45.3 / 53.4 |

Bold = the paper's Table 2.
