# Go (19 x 19)

**Task.** A position from a strong human game is shown as a picture of the whole board (no coordinates, no move history). The
model marks the next best move with a red dot; the reference is KataGo's move after a 400-visit search. Image model only
(Qwen-Image-2.1 LoRA, 384 px). Code: `nbm/tasks/go/` (`rules.py` the game, `render.py` pictures and readout, `build.py` the data,
`katago.py` the labelling engine), config `configs/go/image.yaml`.

| | |
|---|---|
| condition | the board: wood background, grid with thick edges and star points, black / white stones (`render.render_board`) |
| prompt | `Black to play. Mark the best move with a red dot.` (or `White to play. ...`) |
| target | the condition with a red disc (radius 0.38 of a cell, alpha 0.9) on the KataGo move; its cell is the highlight region of the loss (weight 5) |
| readout | per point, the least-squares strength of the red disc in (output - condition); the strongest of all 361 points is the move (threshold-free). It counts as **legal** if it is an empty point that is not suicide and not the ko point, and as **best** if it is KataGo's move. |

## Data

Paper data: `go/` under `$NBM_DATA` (`scripts/download_data.sh`): `test.parquet` 2,000 positions (908 games),
`val.parquet` 1,000 (426 games), `train.parquet` 1,000,000 (91,253 games). Splits are by game: no game and no board (stones +
side to move) occurs in two splits.

| column | meaning |
|---|---|
| `id` | `<archive dir>/<file>.sgf#<ply>` (the position after `ply` moves of that game) |
| `side` | `black` / `white`, the side to move |
| `black`, `white` | indices of the stones, `y * 19 + x` with x = column from the left, y = row from the top |
| `ko` | index of the point the side to move may not play because of the simple-ko rule, -1 if none |
| `correct` | `[index of KataGo's move]` (also `best`, in GTP text, e.g. `D17`) |
| `phase` (paper files: `level`) | moves already played: `p20_59`, `p60_119`, `p120_179`, `p180_` |
| val / test also | `game`, `ply`, `game_move` (the move the human played), `best_winrate`, `root_winrate`, `root_score`, `cands` (every searched root move: visits, winrate, score lead, prior), the move history (`moves`, paper files: `moves_json`), game metadata |

### Building it from scratch

1. **Game records.** The KGS archive of [u-go.net](https://u-go.net/gamerecords/) (games where one player is 7 dan or stronger,
   or both are 6 dan; distributed with KGS's permission for research, not redistributed here). Download the 55 archives listed in
   `nbm/tasks/go/kgs_archives.sha256` from `https://dl.u-go.net/gamerecords/<file>`, check them (`sha256sum -c`), and unpack
   them all into one directory (`tar xjf <file> -C <kgs dir>`; 228,849 SGF files).
2. **KataGo.** The official release [v1.18.2](https://github.com/lightvector/KataGo/releases/tag/v1.18.2)
   (`katago-v1.18.2-cuda12.1-cudnn9.8.0-linux-x64.zip`, sha256 `3e30b486...a3bc9c5d`) and the network
   `kata1-tf3-b11c768-s11003M-d5973M-7gres.bin.gz` from [katagotraining.org/networks](https://katagotraining.org/networks/)
   (sha256 `93bdb63a3bfae4a70db0cb5265287495ecfc10b1ba1cc6814feeba1cdf055871`). Set `KATAGO=<path of the katago binary>` (any
   wrapper that runs it, e.g. in a container, works). Settings (`nbm/tasks/go/analysis.cfg`, `katago.py`): Chinese rules, komi
   7.5 for every position (the picture shows neither rules nor komi; most KGS games were played with Japanese rules and 6.5),
   the game's full move history in every query (ko and superko exact), winrates for the side to move, 400 visits, 32 analysis
   threads x 2 search threads, batch 64.
3. **The steps** (`python -m nbm.tasks.go.build <step>`, details in the module docstring; `<work>` holds the intermediate files):

```bash
B="python -m nbm.tasks.go.build"; W=work/go; OUT=$NBM_DATA/go; NET=<dir>/kata1-tf3-b11c768-s11003M-d5973M-7gres.bin.gz
$B games --kgs <kgs dir> --work $W                         # 106,318 valid games, seeded order (CPU, minutes)
$B positions --work $W --round eval                        # 400,000 positions: 5 from each of the first 80,000 games
$B value --work $W --round eval --net $NET                 # raw network value, 1 visit (GPU)
$B select --work $W --round eval                           # first 125,000 not-decided positions (value in [0.05, 0.95])
$B label --work $W --round eval --net $NET --part 0 --parts 16   # 400-visit labels; one GPU job per part (~1 h each on an A5000)
$B split --work $W --round eval --out $OUT                 # -> test.parquet, val.parquet
$B positions --work $W --round train --out $OUT            # ~3.15M positions: up to 30 per game, val / test games excluded
for k in $(seq 0 31); do $B value --work $W --round train --net $NET --shard $k; done      # one GPU job per shard
$B select --work $W --round train                          # first 1,100,000 not-decided positions
for k in $(seq 0 31); do $B label --work $W --round train --net $NET --shard $k; done      # ~4 h per shard on an A5000
$B split --work $W --round train --out $OUT                # -> train.parquet (1,000,000 rows)
```

The labels cost about 130 GPU hours. The game filter, the game order, the drawn positions, the selection and the splits are
deterministic and reproduce the paper's files (same rows, same contents); KataGo's search is not bit-for-bit repeatable
(random symmetries, multi-threaded search), so relabelled positions agree with the paper's labels on most but not all rows.

## Train

```bash
torchrun --standalone --nproc_per_node 4 -m nbm.train --config configs/go/image.yaml
```

Setup: 4 processes x 2 GPUs (24 GB each: text encoder + VAE on one, DiT on the other), batch 2 per process (global 8),
125,000 steps = one pass over the 1M positions (about 3.7 days on 8 A5000), LoRA rank 64, lr 1e-4 cosine, highlight weight 5.
Every 12,500 steps the first 256 validation positions (phase-stratified order) are evaluated with 20 sampling steps.

## Evaluate

```bash
torchrun --standalone --nproc_per_node 2 -m nbm.evaluate --config configs/go/image.yaml --checkpoint checkpoints/go_image --split test            # 20 steps
torchrun --standalone --nproc_per_node 2 -m nbm.evaluate --config configs/go/image.yaml --checkpoint checkpoints/go_image --split test --steps 1  # one step
torchrun --standalone --nproc_per_node 2 -m nbm.evaluate --config configs/go/image.yaml --checkpoint checkpoints/go_image --split test --votes 8  # majority vote
```

`scripts/download_checkpoints.sh go_image` downloads the released checkpoint. Each process uses 2 GPUs. Batches hold
one side to move (white rows first, then black, 8 per batch); the noise of row i, sample j is seeded (seed + j) * 1,000,003 + i,
so the records do not depend on the number of processes. Writes `metrics.json` (`legal_move`, `best_move`, per phase, `chance` =
mean 1 / number of legal moves) and `predictions.jsonl` (per position: label, move, legal, best, the samples, the disc strength).

## Expected numbers (released checkpoint, test, seed 0)

| readout | legal move | best move (= KataGo) |
|---|---|---|
| 20 steps (default) | 99.8 | 20.9 |
| one step (`--steps 1`) | 99.5 | 31.1 |
| majority vote of 8 samples, 20 steps (`--votes 8`) | 99.7 | 27.9 |

One evaluation of the 2,000 test positions takes about 30 min at 20 steps and
5 min at one step on 8 A5000 (4 processes).
