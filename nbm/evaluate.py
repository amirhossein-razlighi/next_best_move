"""Evaluate a checkpoint (or the zero-shot base model) on a split with the task's evaluation loop.

    torchrun --standalone --nproc_per_node <P> -m nbm.evaluate --config configs/<task>/<image|video>.yaml \
        --checkpoint <run>/best --split test [--steps 1] [--votes 8] [--limit N] [--seed 0] [--out <dir>] [key.path=value ...]

Defaults come from the config's eval section (eval.steps = 20 Euler steps for the image model, 30 UniPC steps for the video
model; votes 1). --steps 1 is the paper's one-step readout, --votes n the majority vote over n samples; every readout is the
threshold-free strongest-mark readout of the task. Overrides (e.g. eval.mode=closed_loop) set task-specific evaluation keys.

Processes: P processes share the work (Task.evaluate shards its batches); with per-row noise the result does not depend on P.
Image model with model.split_devices: process r uses GPUs 2r (text encoder + VAE) and 2r + 1 (DiT), so P processes need 2P
GPUs; otherwise (and for the video model) process r uses GPU r.

Writes <out>/metrics.json (settings + the task's metrics), <out>/predictions.jsonl (one record per item) and <out>/_DONE.
Default out: $NBM_OUTPUTS/eval/<task>_<family>/<split>-steps<k>-votes<n>-seed<s>[-limit<N>].
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from nbm import paths
from nbm.config import load_config
from nbm.tasks import make_task
from nbm.tasks.base import EvalContext, write_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on a task (nbm/evaluate.py docstring).")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None, help="LoRA checkpoint directory (default: the base model, zero-shot)")
    p.add_argument("--split", default="test")
    p.add_argument("--steps", type=int, default=None, help="sampling steps (default eval.steps)")
    p.add_argument("--votes", type=int, default=None, help="samples per item for the majority vote (default eval.votes or 1)")
    p.add_argument("--limit", type=int, default=None, help="first N rows of the split")
    p.add_argument("--seed", type=int, default=None, help="noise seed (default eval.seed)")
    p.add_argument("--batch_size", type=int, default=None, help="default eval.batch_size")
    p.add_argument("--out", default=None)
    p.add_argument("overrides", nargs="*", help="key.path=value")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    steps = args.steps if args.steps is not None else int(cfg.eval.steps)
    votes = args.votes if args.votes is not None else int(cfg.eval.get("votes", 1))
    seed = args.seed if args.seed is not None else int(cfg.eval.seed)
    bs = args.batch_size if args.batch_size is not None else int(cfg.eval.batch_size)
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    out = Path(args.out) if args.out else paths.output_path(
        f"eval/{cfg.task}_{cfg.family}/{args.split}-steps{steps}-votes{votes}-seed{seed}" + (f"-limit{args.limit}" if args.limit else ""))

    image = cfg.family == "image"
    if image and cfg.model.split_devices:
        dit, text = torch.device("cuda", 2 * local + 1), torch.device("cuda", 2 * local)
    else:
        dit = text = torch.device("cuda", local)
    if world > 1:
        torch.cuda.set_device(dit)
        dist.init_process_group("nccl", timeout=timedelta(minutes=int(cfg.train.dist_timeout_min)))
    barrier = dist.barrier if world > 1 else None

    task = make_task(cfg)
    df = task.load_split(args.split)
    df = (df if args.limit is None else df.head(args.limit)).reset_index(drop=True)
    base = paths.model_path(cfg.model.base)
    if image:
        from nbm.models.image import ImageModel
        model = ImageModel(base, dit, text_device=text)
    else:
        from nbm.models.video import VideoModel
        model = VideoModel(base, dit, task.prompts(), paths.output_path("cache"), is_main=rank == 0, barrier=barrier)
    if args.checkpoint:
        lora = cfg.model.lora
        model.load_lora(args.checkpoint, int(lora.rank), float(lora.alpha), list(lora.targets))
    model.transformer.eval()

    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        print(f"evaluate {cfg.task} {cfg.family} | checkpoint {args.checkpoint or 'none (base model)'} | {args.split} {len(df)} rows | "
              f"steps {steps} votes {votes} seed {seed} batch {bs} | {world} processes | out {out}", flush=True)
    ctx = EvalContext(cfg, model, args.split, steps, votes, seed, bs, rank, world, out if rank == 0 else None)
    t0 = time.time()
    res = task.evaluate(ctx, df)
    if rank == 0:
        meta = {"task": cfg.task, "family": cfg.family, "config": str(args.config), "checkpoint": args.checkpoint,
                "base_model": str(base), "split": args.split, "rows": len(df), "steps": steps, "votes": votes, "seed": seed,
                "batch_size": bs, "processes": world, "seconds": round(time.time() - t0, 1)}
        write_result(res, out, meta)
        print("metrics: " + " ".join(f"{k} {v}" for k, v in res.metrics.items() if isinstance(v, (int, float))), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
