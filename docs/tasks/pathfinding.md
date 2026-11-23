# Pathfinding: shortest path and terrain with hidden costs

Paper: Table 2 ("Shortest path", "Cost-aware path") and Appendix A.2. Two tasks share one package, `nbm/tasks/pathfinding/`:

| task name | maps | target | configs |
|---|---|---|---|
| `pathfinding.shortest_path` | floor / wall, every step costs 1 | a shortest route | `configs/shortest_path/{image,video}.yaml` |
| `pathfinding.terrain` | floor / mud / sand / wall, entering costs 1 / 2 / 4, never shown | a cheapest route | `configs/terrain/{image,video}.yaml` |

Image model: Qwen-Image-2.1 + LoRA, run in a closed loop, one move per call. Video model: Wan2.2-TI2V-5B + LoRA, draws
"decide, then imagine" clips of up to 8 moves that are executed.

## Task

An agent on an N x N grid (N <= 12) must reach a goal with 4-neighbour moves; walls cannot be entered and cells cannot be
revisited. The prompt is always `Continue.`: no rule, cost or legend is ever given, the costs of mud and sand must be inferred
from the training routes.

* **Picture** (`render.py`): 384 x 384 px, 32 px cells (one video latent token, 2 x 2 image tokens), the map centred on a black
  canvas. Floor light grey, wall dark grey, mud brown with stripes, sand yellow with dots. The agent is a blue disc, the goal a
  red ring, visited cells are green.
* **Move** (the mark M(x, a)): a red disc on the next cell. Image model: input = a state, target = the same picture + the red
  disc on the next cell of the target route. Video model: a 17-block clip; block 0 = a state, then per move a DECIDE block (the
  red disc) and an IMAGINE block (the agent moved, its old cell green), up to 8 moves, then the last state held.
* **Readout** (`render.read_move`, threshold-free): a fixed decoder reads every cell by matching the mean colours of its centre,
  ring band and corners to the known palette; the move is the cell with the strongest red disc (least-squares strength against
  the cell's unmarked centre), whatever its strength. A decided cell that is not a free, unvisited 4-neighbour of the agent is
  illegal and ends the solve.
* **Targets** (`maps.target_path`): a shortest (terrain: cheapest) route, ties broken by fewest turns, then by the direction
  order right, down, left, up. Evaluation accepts any optimal route.

### Evaluation loops (`evaluation.py`)

* **Image, closed loop**: every call is one generation from a clean picture of the true state; the decided cell is applied by
  the code and the new state is drawn for the next call. Stop at the goal, at an illegal move, or after 2 x target length + 8
  calls.
* **Video, clip execution**: clip 0 starts from the start state, clip k >= 1 from the model's own last picture of clip k - 1.
  The DECIDE blocks of a clip are executed in order (the IMAGINE blocks are only checked). Stop at the goal, at an illegal move,
  when a clip makes no move, or after ceil((2 x target length + 8) / 8) clips.
* Noise of row i in call (clip) t: seed `seed * 1,000,003 + i + t * 100,000,007`, one CPU generator per row, so records do
  not depend on the number of processes. Each round generates for all unfinished maps in consecutive batches (image 8,
  video 2, as in the paper; at bfloat16 the outputs depend slightly on the batch composition).

Metrics per map (`predictions.jsonl`; `metrics.json` has their means and breakdowns per map size `/n<N>`, family, route length
`/len<a-b>` and probe label):

| metric | meaning |
|---|---|
| `optimal` | valid and the route's cost is optimal (uniform: its length is the BFS distance). Table 2 "optimal" / "cheapest route" |
| `valid` | reached the goal by legal moves, the route is simple, and nothing was changed off the route (image: no call's output changed another cell; video: the last picture equals the solved state outside the route) |
| `probe_correct` | probe maps: the agent reached the goal by legal moves and its route crosses mud / sand exactly when the cheapest routes do (the off-route edit check of `valid` is not applied). Table 2 "detour or cross" |
| `reached`, `path_valid`, `edits`, `cost`, `gap`, `stop` (`goal` / `illegal` / `stall` / `cap`) | parts of the above |
| `steps` | every decided move: `from`, `to`, `legal`, `optimal` (the cost-to-go drops by the step's cost), `strength` of the red disc; video also `imagine_ok` (the IMAGINE block shows exactly the state after the move) |
| `calls` (image), `segments` / `n_segments`, `last_exact`, `moves_after_goal` (video) | loop bookkeeping |
| `crossed` (terrain) | the route enters mud or sand |

`metrics.json` `score` = mean `optimal`.

## Data

The data is generated, no external tool is needed (pure Python, deterministic: a rebuild is identical row by row to the
released tables; both sets together take about 2 minutes with 16 workers).

```bash
python -m nbm.tasks.pathfinding.data --costs uniform --out $NBM_DATA/shortest_path --workers 16
python -m nbm.tasks.pathfinding.data --costs terrain --out $NBM_DATA/terrain --workers 16
```

| task | split | maps | content |
|---|---|---|---|
| shortest path | `train` | 200,000 | N 5..10, families blocks / walls / rooms, BFS distance 2..20 |
| | `val` | 600 | same distribution |
| | `test_id` (alias `test`) | 1,000 | same distribution ("held-out maps") |
| | `test_ood_long` | 500 | N 9..10, distance 21..32 ("longer paths") |
| | `test_ood_size` | 500 | N 11..12 ("larger maps") |
| | `test_ood_maze` | 500 | N 5..10, braided depth-first mazes, a family never seen in training (appendix) |
| terrain | `train` | 200,000 | N 5..10, the three families with fewer walls and mud / sand on 25-45 % of the free cells, route length <= 20 |
| | `val` | 600 | same distribution |
| | `test` | 1,000 | same distribution ("cheapest route") |
| | `probe` | 500 | 10 x 10 open rooms with one band of sand or mud in the way: 250 maps where every cheapest route crosses it, 250 where every cheapest route goes around it ("detour or cross") |

Row i of a split takes the i-th (map size, family) pair of a fixed cycle (every prefix is balanced) and a map seed from a
per-split base; no evaluation map appears in training and the evaluation splits are pairwise disjoint. Columns: `id`, `split`,
`n`, `family`, `terrain`, `seed`, `grid` (row-major digits: 0 floor, 1 wall, 2 mud, 3 sand), `sr`, `sc` / `gr`, `gc` (start /
goal), `d` (BFS distance), `cost` (optimal cost), `len`, `turns`, `moves` (the target route as `R` / `D` / `L` / `U`),
`uniform_moves` / `uniform_cost` (terrain: the step-shortest route and its true cost), `label`
(probe: `cross` / `detour`), `seg_di` (the clip of the target route used for video training), `seg_walk` (unused), `row_seed`.
`build.json` holds a hash per table; the released tables:

| table | sha1 (first 12 hex) |
|---|---|
| uniform `train` / `val` / `test_id` | `9b3ea80a126f` / `17bf5b87a7b0` / `afde9eedadde` |
| uniform `test_ood_long` / `test_ood_size` / `test_ood_maze` | `900e9321b753` / `b0b64ff1f335` / `4536c083dee4` |
| terrain `train` / `val` / `test` / `probe` | `1ec4338844f0` / `b91a2b23f96b` / `0c8a220613d9` / `a1ade87fbde3` |

## Train

```bash
torchrun --standalone --nproc_per_node 2 -m nbm.train --config configs/shortest_path/image.yaml   # 2 processes x 2 GPUs
torchrun --standalone --nproc_per_node 4 -m nbm.train --config configs/terrain/image.yaml         # 4 processes x 2 GPUs
torchrun --standalone --nproc_per_node 8 -m nbm.train --config configs/shortest_path/video.yaml   # 8 processes x 1 GPU
torchrun --standalone --nproc_per_node 8 -m nbm.train --config configs/terrain/video.yaml
```

All: rank-64 LoRA, learning rate 1e-4 (cosine, 200 warm-up steps). Image: global batch 8, 8,000 steps (uniform) / 24,000 steps
(terrain); each training access draws a random state along the target route (every state a closed-loop solve visits), target =
that state + the red disc on the next cell. Video: global batch 16 clips, 5,000 steps, one clip per map (the row's `seg_di`
segment of the target route). During training, `eval/step_*` holds a quick validation on the first validation maps (image:
one call from the start state; video: one teacher-forced clip).

## Evaluate

```bash
torchrun --standalone --nproc_per_node 2 -m nbm.evaluate --config configs/shortest_path/image.yaml \
    --checkpoint checkpoints/shortest_path_image --split test_id        # also test_ood_long, test_ood_size, test_ood_maze
torchrun --standalone --nproc_per_node 4 -m nbm.evaluate --config configs/shortest_path/video.yaml \
    --checkpoint checkpoints/shortest_path_video --split test_id
torchrun --standalone --nproc_per_node 2 -m nbm.evaluate --config configs/terrain/image.yaml \
    --checkpoint checkpoints/terrain_image --split test                 # and --split probe
torchrun --standalone --nproc_per_node 4 -m nbm.evaluate --config configs/terrain/video.yaml \
    --checkpoint checkpoints/terrain_video --split probe
```

Add `--steps 1` for the one-step readout, `--limit N` for the first N maps. One sample per call (`--votes` must be 1).
A full test split takes a few GPU hours (image: about 25 s per batch of 8 pictures at 20 steps on an A5000; a 1,000-map split
needs about 15,000 pictures).

## Expected numbers (released checkpoints, seed 0, threshold-free readout, % of maps)

| task | split | model | steps | optimal | valid | Table 2 |
|---|---|---|---|---|---|---|
| shortest path | `test_id` (1,000) | image | 20 | **88.8** | 94.7 | optimal, held-out maps |
| | | video | 30 | **94.6** | 97.4 | |
| | `test_ood_long` (500) | image | 20 | **47.0** | 70.0 | optimal, longer paths |
| | | video | 30 | **66.6** | 76.0 | |
| | `test_ood_size` (500) | image | 20 | **79.0** | 89.8 | optimal, larger maps |
| | | video | 30 | **79.8** | 84.6 | |
| | `test_ood_maze` (500) | image | 20 | 78.4 | 89.2 | appendix A.2 |
| | | video | 30 | 84.4 | 92.0 | |
| terrain | `test` (1,000) | image | 20 | **87.5** | 98.5 | cheapest route |
| | | video | 30 | **82.8** | 95.0 | |
| | `probe` (500) | image | 20 | 79.8 (probe_correct **85.0**) | 99.8 | detour or cross |
| | | video | 30 | 69.8 (probe_correct **76.8**) | 95.4 | |

Bold = Table 2. One step (`--steps 1`, appendix D): shortest path `test_id` image 89.0 / video 83.0 optimal, `test_ood_long`
47.8 / 14.2, `test_ood_size` 81.8 / 61.0; terrain `test` image 87.3 / video 62.6, `probe` probe_correct 79.0 / 30.2. The
untrained base models never produce a valid route (0.0).
