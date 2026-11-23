"""Evaluation loops of the pathfinding tasks and their scoring (Table 2: optimal / cheapest route, detour or cross).

Every generated picture is read with the fixed cell decoder (render.decode_cells) against the TRUE map, with the paper's
threshold-free readout: the move of a DECIDE picture is the cell with the strongest red disc (render.read_move), at any strength.
A decided cell that is not a free, unvisited 4-neighbour of the agent is ILLEGAL and ends the solve.

image closed loop   every call = one generation from a clean picture of the true state (agent at its cell, visited cells green);
                    the decided cell is applied by the code; stop at the goal, at an illegal move, or after the move budget of
                    2 x target length + 8 calls. Noise of row i in call t: seed * 1,000,003 + i + t * 100,000,007.
video clip execution  clip 0 from the start state; clip k >= 1 from the model's OWN last picture of clip k - 1 (its pixels, not a
                    re-render); the up to 8 DECIDE pictures of a clip are executed in order (the bookkept agent moves; the IMAGINE
                    picture after each move is only checked); stop at the goal, at an illegal move, when a clip makes no move, or
                    after ceil(move budget / 8) clips. Same noise formula with t = the clip index.
Each round generates for all unfinished rows in global batches (ctx.my_batches), so with per-row noise the records do not
depend on the number of processes.

Row metrics: reached (the agent reached the goal by legal moves), path_valid, edits (image: some call's output changed a cell
outside its mark; video: the last picture differs from the solved state outside the route), valid = reached + path_valid + no
edits, optimal = valid and the route's cost equals the optimal cost (uniform: its length equals the BFS distance), gap, step
legality / optimality (an optimal step lowers the cost-to-go by exactly its own cost); terrain extras: crossed (the route enters
mud or sand), probe_correct (on probe maps: crossed exactly when the cheapest routes cross).

In training (ctx.in_training) a quicker validation runs instead of full solves: image = one call from the start state per row;
video = one teacher-forced clip per row (from the state at the start of the row's seg_di segment of the target route).
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from PIL import Image

from nbm.models.video import video_to_blocks
from nbm.tasks.base import EvalContext, EvalResult, row_seed
from nbm.tasks.pathfinding import maps
from nbm.tasks.pathfinding import render as R

ROUND_SEED_STRIDE = 100_000_007
LEN_BUCKETS = ((1, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 99))


def noise_seed(seed: int, row: int, call: int) -> int:
    """Noise seed of row `row` in call / clip `call` of a closed loop."""
    return row_seed(seed, row) + call * ROUND_SEED_STRIDE


def move_budget(row: dict) -> int:
    """Moves allowed in a solve: 2 x target length + 8 (never shown to the model)."""
    return 2 * int(row["len"]) + 8


def max_clips(row: dict) -> int:
    return -(-move_budget(row) // R.MOVES_PER_CLIP)


def to_map(row: dict) -> dict:
    """The map of a data row."""
    n = int(row["n"])
    return {"n": n, "family": row["family"], "grid": maps.grid_from_str(row["grid"], n), "start": (int(row["sr"]), int(row["sc"])),
            "goal": (int(row["gr"]), int(row["gc"])), "terrain": bool(row["terrain"])}


def target(row: dict) -> tuple[dict, list]:
    """(map, target route start .. goal) of a data row."""
    m = to_map(row)
    return m, maps.path_from_moves(m["start"], row["moves"])


# ------------------------------------------------------------------------------------------------ one move
def step_record(m: dict, D: np.ndarray, pos, visited, cell, clean: bool = True) -> dict:
    """A decided move pos -> cell: legal (a free, unvisited 4-neighbour) and optimal (cost-to-go drops by the step's cost)."""
    g = m["grid"]
    cell = (int(cell[0]), int(cell[1]))
    adjacent = abs(cell[0] - pos[0]) + abs(cell[1] - pos[1]) == 1
    wall = bool(g[cell] == maps.WALL)
    revisit = cell in set(map(tuple, visited)) or cell == tuple(pos)
    legal = adjacent and not wall and not revisit
    cc = maps.cell_cost(g, m["terrain"])
    optimal = legal and int(D[cell]) + int(cc[cell]) == int(D[tuple(pos)])
    return {"from": [int(pos[0]), int(pos[1])], "to": [cell[0], cell[1]], "legal": bool(legal), "adjacent": bool(adjacent),
            "wall": wall, "revisit": bool(revisit), "optimal": bool(optimal), "clean": bool(clean)}


def edit_count(dec: dict, m: dict, pos, visited, cell) -> int:
    """Cells of a DECIDE output that differ from the state + its red disc. When the decided disc is faint (the cell does not
    decode as the clearest red disc) that cell itself is not counted."""
    dcell, drawn, _ = R.red_disc_cell(dec)
    skip = () if drawn and tuple(dcell) == tuple(cell) else (cell,)
    return R.mismatches(dec, m, pos, visited, mark=cell, skip=skip)


def image_call(rec: dict, m: dict, D: np.ndarray, frame: np.ndarray, cap: int) -> None:
    """Fold one image-model output (the DECIDE picture of the current state) into the row record."""
    dec = R.decode_cells(frame, m["n"])
    pos, vis = tuple(rec["pos"]), [tuple(p) for p in rec["visited"]]
    cell, strength = R.read_move(frame, dec, pos)
    _, _, n_red = R.red_disc_cell(dec)
    rec["calls"] += 1
    rec["edit_calls"] += int(edit_count(dec, m, pos, vis, cell) > 0)
    s = step_record(m, D, pos, vis, cell, clean=n_red == 1)
    s["strength"] = round(strength, 4)
    rec["steps"].append(s)
    if not s["legal"]:
        rec["done"], rec["stop"] = True, "illegal"
        return
    rec["visited"] = [list(p) for p in vis] + [list(pos)]
    rec["pos"] = list(cell)
    if tuple(cell) == tuple(m["goal"]):
        rec["done"], rec["stop"] = True, "goal"
    elif rec["calls"] >= cap:
        rec["done"], rec["stop"] = True, "cap"


def decode_clip(blocks: list, m: dict, D: np.ndarray, pos0, visited0) -> dict:
    """Execute the DECIDE pictures of one generated clip (17 uint8 blocks) from the bookkept state (pos0, visited0).
    Returns steps, end_pos / end_visited, stop (None, or "illegal"), reached, n_moves, moves_after_goal (DECIDE marks drawn after
    the goal), last_exact (the last picture equals the solved state) and last_off_path (its wrong cells outside the route)."""
    n, goal = m["n"], tuple(m["goal"])
    pos, vis = tuple(pos0), [tuple(p) for p in visited0]
    steps, stop, after = [], None, 0
    for j in range(R.MOVES_PER_CLIP):
        decide, imagine = blocks[1 + 2 * j], blocks[2 + 2 * j]
        dd = R.decode_cells(decide, n)
        _, drawn, n_red = R.red_disc_cell(dd)
        if pos == goal:
            after += int(drawn)
            continue
        if stop:
            continue
        cell, strength = R.read_move(decide, dd, pos)
        s = step_record(m, D, pos, vis, cell, clean=n_red == 1)
        s["strength"] = round(strength, 4)
        di = R.decode_cells(imagine, n)
        a, a_drawn, _ = R.agent_cell(dd)
        s["decide_agent_ok"] = bool(a_drawn and a == pos)
        s["imagine_ok"] = bool(s["legal"] and R.mismatches(di, m, cell, vis + [pos]) == 0)
        ia, ia_drawn, _ = R.agent_cell(di)
        s["imagine_agent_ok"] = bool(ia == cell and ia_drawn)
        steps.append(s)
        if not s["legal"]:
            stop = "illegal"
            continue
        vis.append(pos)
        pos = tuple(s["to"])
    last = R.decode_cells(blocks[-1], n)
    route = set(vis) | {pos, goal}
    return {"steps": steps, "end_pos": [int(pos[0]), int(pos[1])], "end_visited": [[int(a), int(b)] for a, b in vis], "stop": stop,
            "reached": pos == goal, "n_moves": len([s for s in steps if s["legal"]]), "moves_after_goal": after,
            "last_exact": R.mismatches(last, m, pos, vis) == 0, "last_off_path": R.mismatches(last, m, pos, vis, skip=route)}


def video_clip(rec: dict, clip: dict, cap: int) -> None:
    """Fold one executed clip into the row record."""
    rec["segments"].append({k: clip[k] for k in ("stop", "reached", "n_moves", "moves_after_goal", "last_exact", "last_off_path")})
    rec["steps"] += clip["steps"]
    rec["pos"], rec["visited"] = clip["end_pos"], clip["end_visited"]
    if clip["reached"]:
        rec["done"], rec["stop"] = True, "goal"
    elif clip["stop"] == "illegal":
        rec["done"], rec["stop"] = True, "illegal"
    elif clip["n_moves"] == 0:
        rec["done"], rec["stop"] = True, "stall"
    elif len(rec["segments"]) >= cap:
        rec["done"], rec["stop"] = True, "cap"


# ------------------------------------------------------------------------------------------------ records
def base_record(row: dict, i: int) -> dict:
    return {"row": int(i), "id": row["id"], "n": int(row["n"]), "family": row["family"], "terrain": bool(row["terrain"]),
            "d": int(row["d"]), "best_cost": int(row["cost"]), "target_len": int(row["len"]), "label": row["label"] or ""}


def init_record(row: dict, i: int) -> dict:
    return {**base_record(row, i), "pos": [int(row["sr"]), int(row["sc"])], "visited": [], "steps": [], "segments": [],
            "done": False, "stop": None, "calls": 0, "edit_calls": 0}


def terrain_metrics(m: dict, path: list | None, label: str = "") -> dict:
    """crossed: the route enters mud or sand (None = no route); probe_correct: crossed matches the probe map's label."""
    if not m.get("terrain"):
        return {}
    g = m["grid"]
    crossed = None if path is None else bool(any(g[tuple(p)] in (maps.MUD, maps.SAND) for p in path))
    out = {"crossed": crossed}
    if label:
        out["probe_correct"] = bool(crossed is not None and crossed == (label == "cross"))
    return out


def finish_record(rec: dict, m: dict, D: np.ndarray, family: str) -> dict:
    """The final row metrics of a solve (module doc)."""
    path = [tuple(m["start"])] + [tuple(s["to"]) for s in rec["steps"] if s["legal"]]
    reached = rec["stop"] == "goal"
    sc = maps.score_path(m, path, D) if reached else {"path_valid": False, "cost": None, "gap": None, "optimal": False}
    if family == "image":
        edits = rec["edit_calls"] > 0
    else:
        edits = bool(rec["segments"] and rec["segments"][-1]["last_off_path"] > 0)
    valid = bool(reached and sc["path_valid"] and not edits)
    st = rec["steps"]
    unused = ("segments",) if family == "image" else ("calls", "edit_calls")
    out = {k: v for k, v in rec.items() if k not in ("pos", "visited", "done") + unused}
    out.update(reached=reached, path_valid=bool(reached and sc["path_valid"]), edits=edits, valid=valid,
               optimal=bool(valid and sc["optimal"]), cost=sc.get("cost"), gap=sc.get("gap") if valid else None,
               wall_violation=any(s["wall"] for s in st), n_moves=len([s for s in st if s["legal"]]),
               step_legal=float(np.mean([s["legal"] for s in st])) if st else float("nan"),
               step_optimal=float(np.mean([s["optimal"] for s in st])) if st else float("nan"),
               first_optimal=bool(st and st[0]["optimal"]), path=[list(p) for p in path])
    if family == "video":
        out["imagine_ok"] = float(np.mean([s.get("imagine_ok", False) for s in st])) if st else float("nan")
        out["n_segments"] = len(rec["segments"])
        out["last_exact"] = bool(rec["segments"] and rec["segments"][-1]["last_exact"])
        out["moves_after_goal"] = int(sum(s["moves_after_goal"] for s in rec["segments"]))
    out.update(terrain_metrics(m, path if reached else None, out["label"]))
    return out


def _mean(xs) -> float:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def len_bucket(L: int) -> str:
    for a, b in LEN_BUCKETS:
        if a <= L <= b:
            return f"{a}-{b}" if b < 99 else f"{a}+"
    return "?"


SUMMARY_KEYS = ("valid", "path_valid", "optimal", "reached", "edits", "wall_violation", "step_legal", "step_optimal",
                "first_optimal", "imagine_ok", "n_moves", "n_segments", "calls", "last_exact", "moves_after_goal", "crossed",
                "probe_correct")


def summarize(recs: list[dict]) -> dict:
    """Means of the row metrics, stop reasons, and optimal / valid / probe_correct per map size, family, route length and label."""
    out: dict = {"n": len(recs)}
    for k in SUMMARY_KEYS:
        if any(k in r for r in recs):
            out[k] = _mean(r.get(k) for r in recs)
    gaps = [r["gap"] for r in recs if r.get("valid") and r.get("gap") is not None]
    out["gap_mean"] = _mean(gaps)
    for s in ("goal", "illegal", "stall", "cap"):
        out[f"stop_{s}"] = _mean(r.get("stop") == s for r in recs)
    groups: dict[str, list] = {}
    for r in recs:
        groups.setdefault(f"/n{r['n']}", []).append(r)
        groups.setdefault(f"/{r['family']}", []).append(r)
        groups.setdefault(f"/len{len_bucket(r['target_len'])}", []).append(r)
        if r.get("label"):
            groups.setdefault(f"/{r['label']}", []).append(r)
    for g, rs in sorted(groups.items()):
        for k in ("valid", "optimal", "probe_correct"):
            if any(k in r for r in rs):
                out[k + g] = _mean(r.get(k) for r in rs)
        out["n" + g] = len(rs)
    out["score"] = out.get("optimal", float("nan"))
    return out


# ------------------------------------------------------------------------------------------------ loops
def _png(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def _unpng(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def closed_loop(ctx: EvalContext, df: pd.DataFrame, prompt: str) -> EvalResult:
    """Full solves of every row: the image closed loop or the video clip execution (module doc)."""
    if ctx.votes != 1:
        raise ValueError("pathfinding solves use one sample per call (--votes 1)")
    video = ctx.family == "video"
    rows = df.to_dict("records")
    recs = [init_record(r, i) for i, r in enumerate(rows)]
    cache: dict[int, tuple] = {}
    last: dict[int, bytes] = {}  # video: the own last picture of every unfinished row (PNG, lossless)

    def MD(i: int) -> tuple[dict, np.ndarray]:
        if i not in cache:
            m = to_map(rows[i])
            cache[i] = (m, maps.cost_to_go(m["grid"], tuple(m["goal"]), m["terrain"]))
        return cache[i]

    def condition(i: int, t: int) -> np.ndarray:
        m = MD(i)[0]
        if not video:
            return R.render(m, recs[i]["pos"], recs[i]["visited"])
        return R.render(m, m["start"]) if t == 0 else _unpng(last[i])

    t = 0
    while True:
        act = [i for i in range(len(rows)) if not recs[i]["done"]]
        if not act:
            break
        mine = ctx.my_batches(act)
        prog = ctx.progress(len(mine), f"{ctx.split} {'clip' if video else 'call'} {t}: {len(act)} unfinished of {len(rows)}")
        out = []
        for _, idx in mine:
            outs = ctx.generate([Image.fromarray(condition(i, t)) for i in idx], [prompt] * len(idx),
                                [noise_seed(ctx.seed, i, t) for i in idx])
            for k, i in enumerate(idx):
                m, D = MD(i)
                r = dict(recs[i], steps=list(recs[i]["steps"]), segments=list(recs[i]["segments"]))
                if video:
                    blocks = [np.asarray(b) for b in video_to_blocks(outs[k])]
                    video_clip(r, decode_clip(blocks, m, D, r["pos"], r["visited"]), max_clips(rows[i]))
                    out.append((i, r, None if r["done"] else _png(blocks[-1])))
                else:
                    frame = np.asarray(outs[k].convert("RGB").resize((R.S, R.S)))
                    image_call(r, m, D, frame, move_budget(rows[i]))
                    out.append((i, r, None))
            prog.update(1, goal=f"{np.mean([x[1]['stop'] == 'goal' for x in out]):.2f}")
        prog.close()
        for i, r, png in ctx.gather(out):
            recs[i] = r
            if png is not None:
                last[i] = png
            else:
                last.pop(i, None)
        ctx.log(f"[{ctx.split}] round {t}: {len(act)} unfinished -> goal {sum(r['stop'] == 'goal' for r in recs)}, illegal "
                f"{sum(r['stop'] == 'illegal' for r in recs)}, stall {sum(r['stop'] == 'stall' for r in recs)}, cap "
                f"{sum(r['stop'] == 'cap' for r in recs)} of {len(recs)}")
        t += 1
    recs = [finish_record(r, *MD(i), ctx.family) for i, r in enumerate(recs)]
    return EvalResult(recs, summarize(recs))


def first_moves(ctx: EvalContext, df: pd.DataFrame, prompt: str) -> EvalResult:
    """In-training validation of the image model: one call from the start state per row; score = first move optimal."""
    rows = df.to_dict("records")
    mine = []
    for _, idx in ctx.my_batches(len(rows)):
        conds = [R.render(to_map(rows[i]), (int(rows[i]["sr"]), int(rows[i]["sc"]))) for i in idx]
        outs = ctx.generate([Image.fromarray(c) for c in conds], [prompt] * len(idx), [row_seed(ctx.seed, i) for i in idx])
        for i, o in zip(idx, outs):
            m = to_map(rows[i])
            D = maps.cost_to_go(m["grid"], tuple(m["goal"]), m["terrain"])
            frame = np.asarray(o.convert("RGB").resize((R.S, R.S)))
            dec = R.decode_cells(frame, m["n"])
            cell, strength = R.read_move(frame, dec, m["start"])
            s = step_record(m, D, m["start"], [], cell)
            mine.append({**base_record(rows[i], i), "move": s["to"], "strength": round(strength, 4), "first_legal": s["legal"],
                         "first_optimal": s["optimal"], "edits": edit_count(dec, m, m["start"], [], cell) > 0})
    recs = sorted(ctx.gather(mine), key=lambda r: r["row"])
    metrics = {"n": len(recs), **{k: _mean(r[k] for r in recs) for k in ("first_legal", "first_optimal", "edits")}}
    metrics["score"] = metrics["first_optimal"]
    return EvalResult(recs, metrics)


def clip_correct(clip: dict) -> bool:
    st = clip["steps"]
    return bool(clip["stop"] is None and st and all(s["optimal"] and s["imagine_ok"] for s in st)
                and (clip["reached"] or len(st) == R.MOVES_PER_CLIP))


def teacher_forced_clips(ctx: EvalContext, df: pd.DataFrame, prompt: str) -> EvalResult:
    """In-training validation of the video model: one clip per row from the state at the start of its seg_di segment of the
    target route; score = clip correct (every move optimal and its IMAGINE consistent, the clip's moves all made or the goal
    reached) + 0.001 x step optimal."""
    rows = df.to_dict("records")
    mine = []
    for _, idx in ctx.my_batches(len(rows)):
        items = []
        for i in idx:
            m, path = target(rows[i])
            seg = int(rows[i]["seg_di"])
            items.append((m, path, seg, R.segment_blocks(m, path, seg)[0]))
        clips = ctx.generate([Image.fromarray(it[3]) for it in items], [prompt] * len(idx), [row_seed(ctx.seed, i) for i in idx])
        for k, (i, (m, path, seg, _)) in enumerate(zip(idx, items)):
            k0 = R.MOVES_PER_CLIP * seg
            D = maps.cost_to_go(m["grid"], tuple(m["goal"]), m["terrain"])
            c = decode_clip([np.asarray(b) for b in video_to_blocks(clips[k])], m, D, path[k0], path[:k0])
            mine.append({**base_record(rows[i], i), "segment": seg, **c, "expected_moves": min(R.MOVES_PER_CLIP, len(path) - 1 - k0),
                         "segment_ok": clip_correct(c)})
    recs = sorted(ctx.gather(mine), key=lambda r: r["row"])
    st = [s for r in recs for s in r["steps"]]
    step_opt = float(np.mean([s["optimal"] for s in st])) if st else float("nan")
    metrics = {"n": len(recs), "segment_ok": _mean(r["segment_ok"] for r in recs), "step_optimal": step_opt,
               "step_legal": float(np.mean([s["legal"] for s in st])) if st else float("nan"),
               "reached": _mean(r["reached"] for r in recs)}
    metrics["score"] = metrics["segment_ok"] + 1e-3 * (step_opt if np.isfinite(step_opt) else 0.0)
    return EvalResult(recs, metrics)
