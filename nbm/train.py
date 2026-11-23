"""Train a LoRA of the image (Qwen-Image-2.1) or video (Wan2.2-TI2V-5B) model on a task.

    torchrun --standalone --nproc_per_node <P> -m nbm.train --config configs/<task>/<image|video>.yaml [key.path=value ...]

Every hyper-parameter comes from the config (nbm/config.py); overrides such as train.max_steps=20 change single values. The
resolved config is written to <output_dir>/config.yaml.

What one optimizer step does: each process draws train.grad_accum batches of train.batch_size examples (nbm/data.py; global
batch = batch_size x grad_accum x processes), the task builds every example (Task.image_example / video_example), and the
flow-matching loss (nbm/models) is averaged and back-propagated into the LoRA (AdamW, cosine schedule with warm-up, gradient
clipping, bf16 autocast, DDP over the DiT via accelerate).
Image noise levels: u ~ U(0, 1), sigma = the scheduler's training sigma at floor(1000 u). Video: sigma = 3u / (1 + 2u) (flow
shift model.flow_shift). Image with model.split_devices: every process uses two GPUs (text encoder + VAE on one, DiT on the
other), so P processes need 2P GPUs.

Outputs in <output_dir> (train.output_dir, under $NBM_OUTPUTS): checkpoint-<step>/ every train.save_every steps and at the end
(the last two are kept), eval/step_<step>/ (validation monitor: Task.evaluate on the first train.eval_n validation rows with
train.eval_steps sampling steps, at step 0, every train.eval_every steps and at the end), tb/ (TensorBoard), and a log line
"step N loss X ..." every train.log_every steps. train.resume = latest | <checkpoint dir> continues a run: same data stream (the
batches already trained on are skipped), optimizer and schedule restored.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from nbm.config import load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a LoRA on a task (nbm/train.py docstring).")
    p.add_argument("--config", required=True, help="configs/<task>/<image|video>.yaml")
    p.add_argument("overrides", nargs="*", help="key.path=value, e.g. train.max_steps=20")
    return p.parse_args(argv)


def pin_gpu_pair() -> int:
    """model.split_devices: the process of local rank r gets the GPU pair (2r, 2r + 1) of its allocation (text encoder + VAE on
    2r, DiT on 2r + 1). CUDA_VISIBLE_DEVICES is reordered so that the DiT GPU is cuda:<r>, the device accelerate uses for the
    model, DDP and barriers. Must run before anything initialises CUDA. Returns the text encoder's index in the new order."""
    r = int(os.environ.get("LOCAL_RANK", 0))
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = vis.split(",") if vis else [str(i) for i in range(len(subprocess.check_output(["nvidia-smi", "-L"]).splitlines()))]
    if len(gpus) < 2 * r + 2:
        raise SystemExit(f"model.split_devices: local rank {r} needs GPUs {2 * r} and {2 * r + 1}, only {len(gpus)} visible")
    te, dit = gpus[2 * r], gpus[2 * r + 1]
    order = [g for g in gpus if g != dit]
    order.insert(r, dit)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(order)
    return order.index(te)


if __name__ == "__main__":  # the GPU pairing has to happen before torch is imported below
    _args = parse_args()
    _cfg = load_config(_args.config, _args.overrides)
    TEXT_GPU = pin_gpu_pair() if _cfg.family == "image" and _cfg.model.split_devices else None
else:
    TEXT_GPU = None

import random  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import InitProcessGroupKwargs, set_seed  # noqa: E402
from diffusers.optimization import get_scheduler  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from nbm import paths  # noqa: E402
from nbm.config import dump_config  # noqa: E402
from nbm.data import (GroupBatchSampler, ImageTrainSet, ShardedBatchSampler, VideoTrainSet, collate_image,  # noqa: E402
                      collate_video)
from nbm.models import lora as L  # noqa: E402
from nbm.progress import eta_text  # noqa: E402
from nbm.tasks import make_task  # noqa: E402
from nbm.tasks.base import EvalContext, write_result  # noqa: E402


def checkpoint_steps(out: Path) -> list[Path]:
    return sorted((p for p in out.glob("checkpoint-*") if p.name.split("-")[1].isdigit()), key=lambda p: int(p.name.split("-")[1]))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    T, M, image = cfg.train, cfg.model, cfg.family == "image"
    out = paths.output_path(T.output_dir)
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=int(T.grad_accum),
                              kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=int(T.dist_timeout_min)))])
    set_seed(int(T.seed) + accelerator.process_index)
    rank, world, is_main = accelerator.process_index, accelerator.num_processes, accelerator.is_main_process
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, out / "config.yaml")
    if world != int(T.processes):
        accelerator.print(f"note: the config was written for {T.processes} processes (global batch "
                          f"{T.batch_size * T.grad_accum * T.processes}); this launch has {world} "
                          f"(global batch {T.batch_size * T.grad_accum * world})")
    task = make_task(cfg)
    base = paths.model_path(M.base)
    lora = M.lora

    # model: frozen backbone + a fresh LoRA (float32 trainable parameters)
    if image:
        from nbm.models.image import ImageModel
        model = ImageModel(base, accelerator.device, text_device=torch.device("cuda", TEXT_GPU) if M.split_devices else None)
    else:
        from nbm.models.video import VideoModel
        model = VideoModel(base, accelerator.device, task.prompts(), paths.output_path("cache"), is_main,
                           barrier=accelerator.wait_for_everyone)
    if T.gradient_checkpointing:
        model.transformer.enable_gradient_checkpointing()
    model.add_trainable_lora(int(lora.rank), float(lora.alpha), list(lora.targets), float(lora.dropout))
    params = [p for p in model.transformer.parameters() if p.requires_grad]

    # data
    train_df = task.load_split("train").head(int(cfg.data.train_limit)).reset_index(drop=True)
    val_df = task.load_split("val").head(int(T.eval_n)).reset_index(drop=True)
    if image:
        dataset, collate = ImageTrainSet(task, train_df), collate_image
        sampler = GroupBatchSampler(task.batch_groups(train_df), int(T.batch_size), int(T.seed), rank, world)
    else:
        dataset, collate = VideoTrainSet(task, train_df), collate_video
        sampler = ShardedBatchSampler(len(train_df), int(T.batch_size), int(T.seed), rank, world)
    nw = int(T.num_workers)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=nw, persistent_workers=nw > 0,
                        prefetch_factor=4 if nw else None)

    optimizer = torch.optim.AdamW(params, lr=float(T.learning_rate), betas=(0.9, 0.999), weight_decay=float(T.weight_decay), eps=1e-8)
    # the prepared scheduler is stepped once per process per optimizer step, hence the factor `world`
    lr_sched = get_scheduler(T.lr_scheduler, optimizer=optimizer, num_warmup_steps=int(T.warmup_steps) * world,
                             num_training_steps=int(T.max_steps) * world)
    transformer, optimizer, lr_sched = accelerator.prepare(model.transformer, optimizer, lr_sched)

    step = 0
    if T.resume:
        ck = checkpoint_steps(out)[-1] if T.resume == "latest" else Path(T.resume)
        L.load_weights(accelerator.unwrap_model(transformer), ck)
        st = torch.load(ck / "state.pt", weights_only=False, map_location="cpu")
        optimizer.load_state_dict(st["optimizer"])
        lr_sched.load_state_dict(st["lr"])
        step = st["step"]
        lr_sched.scheduler.last_epoch = step * world  # re-based: resumable with another process count
        sampler.skip = step * int(T.grad_accum) * world  # continue the uninterrupted data stream
        accelerator.print(f"resumed from {ck} at step {step}")

    tb = SummaryWriter(out / "tb") if is_main else None
    n_trainable = sum(p.numel() for p in params)
    accelerator.print(f"task {cfg.task} | {cfg.family} model {base} | trainable {n_trainable / 1e6:.1f}M | train rows {len(train_df)} | "
                      f"val rows {len(val_df)} | {world} processes x batch {T.batch_size} x accum {T.grad_accum} = "
                      f"{T.batch_size * T.grad_accum * world} per step | output {out}")

    def save(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_lora(path, accelerator.unwrap_model(transformer))
        torch.save({"optimizer": optimizer.state_dict(), "lr": lr_sched.state_dict(), "step": step,
                    "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}}, path / "state.pt")

    def evaluate(at: int) -> None:
        """Validation monitor: Task.evaluate on the validation prefix, sharded over all processes (every process must call it)."""
        model.transformer.eval()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        t0 = time.time()
        ctx = EvalContext(cfg, model, "val", int(T.eval_steps), 1, 0, int(T.eval_batch_size), rank, world,
                          out / "eval" / f"step_{at:06d}" if is_main else None, in_training=True)
        if is_main:
            ctx.out_dir.mkdir(parents=True, exist_ok=True)
        res = task.evaluate(ctx, val_df)
        if is_main:
            write_result(res, ctx.out_dir, {"step": at, "split": "val", "steps": ctx.steps, "seed": 0})
            for k, v in res.metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    tb.add_scalar(f"val/{k}", v, at)
            tb.flush()
            accelerator.print(f"[eval step {at}] " + " ".join(f"{k} {v:.4f}" for k, v in res.metrics.items()
                                                             if isinstance(v, float)) + f" [{time.time() - t0:.0f}s]")
        model.transformer.train()
        torch.cuda.empty_cache()

    if step == 0:
        evaluate(0)
    accelerator.wait_for_everyone()

    transformer.train()
    it = iter(loader)
    running: dict[str, list[float]] = {"loss": [], "hl": [], "rest": []}
    t_last, seen, gnorm = time.time(), 0, torch.tensor(0.0)
    step0, t_run0 = step, time.time()
    shift = float(M.flow_shift) if not image else None
    while step < int(T.max_steps):
        for _ in range(int(T.grad_accum)):
            batch = next(it)
            with accelerator.accumulate(transformer):
                if image:
                    bsz = len(batch["prompts"])
                    sig = model.train_sigmas(model.scheduler, bsz).to(accelerator.device)
                    with accelerator.autocast():
                        tot, hl, rest = model.flow_loss(batch, sig, float(T.highlight_loss_weight), transformer=transformer)
                    loss = tot.mean()
                else:
                    bsz = batch["blocks"].shape[0]
                    sig = model.train_sigmas(bsz, shift).to(accelerator.device)
                    emb = model.embed(batch["prompts"])
                    with accelerator.autocast():
                        errp = model.flow_loss(batch["blocks"], emb, sig, transformer=transformer)  # (B, 16, h, w)
                    if batch["block_weights"] is None:
                        loss = errp.mean(dim=(2, 3)).mean()
                    else:
                        w = batch["block_weights"].to(errp.device)[:, :, None, None].expand_as(errp)
                        loss = (errp * w).sum() / w.sum().clamp_min(1e-8)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    gnorm = accelerator.clip_grad_norm_(params, float(T.max_grad_norm))
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)
            running["loss"].append(loss.item())
            if image:
                running["hl"].append(hl.mean().item())
                running["rest"].append(rest.mean().item())
            seen += bsz * world
        step += 1

        if is_main and step % int(T.log_every) == 0:
            dt = time.time() - t_last
            lr_now = lr_sched.get_last_lr()[0]
            tb.add_scalar("train/loss", np.mean(running["loss"]), step)
            tb.add_scalar("train/lr", lr_now, step)
            tb.add_scalar("train/grad_norm", float(gnorm), step)
            tb.add_scalar("perf/examples_per_sec", seen / dt, step)
            tb.add_scalar("perf/gpu_mem_peak_gb", torch.cuda.max_memory_allocated() / 1e9, step)
            if step % (10 * int(T.log_every)) == 0:
                accelerator.print(f"[train] step {step}/{T.max_steps} {100 * step / int(T.max_steps):.1f}% | "
                                  f"{eta_text(step - step0, int(T.max_steps) - step0, time.time() - t_run0)}", flush=True)
            parts = [f"step {step} loss {np.mean(running['loss']):.4f}"]
            if image:
                parts.append(f"hl {np.nanmean(running['hl']):.4f} rest {np.nanmean(running['rest']):.4f}")
            parts.append(f"lr {lr_now:.2e} gnorm {float(gnorm):.3f} {seen / dt:.2f} examples/s "
                         f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
            accelerator.print(" ".join(parts), flush=True)
            running = {k: [] for k in running}
            t_last, seen = time.time(), 0

        if step % int(T.save_every) == 0 or step == int(T.max_steps):
            if is_main:
                save(out / f"checkpoint-{step}")
                for old in checkpoint_steps(out)[:-2]:
                    shutil.rmtree(old)
            accelerator.wait_for_everyone()

        if step % int(T.eval_every) == 0 or step == int(T.max_steps):
            evaluate(step)
            accelerator.wait_for_everyone()

    if is_main:
        tb.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
