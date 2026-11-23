"""Wan2.2-TI2V-5B as a "decide, then imagine" clip model: prompt embeddings, clip <-> latents, TI2V forward, UniPC sampler, loss.

Clip layout (every video task): N_BLOCKS = 17 blocks; block 0 is the condition picture (one frame), blocks 1..16 are shown for
HOLD = 4 identical frames each, so a clip has 65 frames. The Wan2.2 VAE is causal and 4x temporal (frame 0 alone -> latent frame
0, then every 4 frames -> one latent frame), so latent frame k is exactly block k. A task's clip is any 17 pictures in this
layout (e.g. DECIDE = the marks of the next move, IMAGINE = the position after it, ...).

Model facts: VAE z_dim 48, 16x spatial; DiT patch (1, 2, 2) = one token per 32 x 32 pixels per latent frame; text = UMT5-XXL
embeddings zero-padded to 512 tokens. Image conditioning (expand_timesteps): the clean latent of frame 0 replaces latent frame 0
of the noisy clip at every step and gets timestep 0 (per-token timesteps); the output for frame 0 is ignored.

The text encoder is never resident: the task's few prompts are embedded once (UMT5-XXL loaded, used, freed) into a cache file
under <cache_dir>, before the DiT is loaded.

Numerics (must not change): bfloat16 DiT / VAE, float32 latents, VAE posterior mode, UniPC flow sampler with flow shift 3
(training uses the same shift for its noise levels), sampling under bfloat16 autocast with encode / decode outside it.
"""
from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
from PIL import Image

from nbm.models import lora as L
from nbm.models.noise import row_noise

LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2")
N_BLOCKS = 17
HOLD = 4
N_FRAMES = 1 + HOLD * (N_BLOCKS - 1)  # 65
TEXT_LEN = 512


# ---------------------------------------------------------------- clip layout
def blocks_to_frames(blocks: Sequence) -> list:
    """17 blocks -> 65 frames (block 0 once, every other block HOLD times)."""
    frames = [blocks[0]]
    for b in blocks[1:]:
        frames += [b] * HOLD
    return frames


def frames_to_blocks(frames: Sequence[Image.Image]) -> list[Image.Image]:
    """Generated 65 frames -> 17 blocks. Block j >= 1 owns frames [1 + HOLD (j - 1), 1 + HOLD j); it is read as the mean of the
    group's two central frames (the VAE's temporal blending with neighbouring latent frames sits at the group edges)."""
    assert len(frames) >= N_FRAMES, len(frames)
    blocks = [frames[0]]
    for j in range(1, N_BLOCKS):
        g = frames[1 + HOLD * (j - 1): 1 + HOLD * j]
        sel = g[HOLD // 2 - 1: HOLD // 2 + 1]
        arr = np.mean([np.asarray(f.convert("RGB"), dtype=np.float32) for f in sel], axis=0)
        blocks.append(Image.fromarray(arr.round().clip(0, 255).astype(np.uint8)))
    return blocks


def video_to_blocks(video_u8: torch.Tensor | np.ndarray) -> list[Image.Image]:
    """One generated clip (F, H, W, 3) uint8 -> its 17 blocks as PIL pictures."""
    return frames_to_blocks([Image.fromarray(f) for f in np.asarray(video_u8)])


# ---------------------------------------------------------------- prompt embeddings
@torch.no_grad()
def encode_prompts_once(base: str, prompts: Sequence[str], device: torch.device | str, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Load UMT5-XXL, embed the prompts exactly as WanPipeline does (512 tokens, zero padding), free it.
    Returns {prompt: (512, 4096) CPU tensor}."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    tok = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    shim = SimpleNamespace(tokenizer=tok, text_encoder=te, _execution_device=torch.device(device))
    out = {}
    for p in prompts:
        e = WanPipeline._get_t5_prompt_embeds(shim, prompt=p, max_sequence_length=TEXT_LEN, device=device, dtype=dtype)
        out[p] = e[0].cpu()
    del te
    torch.cuda.empty_cache()
    return out


def prompt_cache(base: str, prompts: Sequence[str], cache_dir: str | os.PathLike, device: torch.device | str,
                 is_main: bool = True) -> str:
    """Path of the embedding cache of this prompt set (keyed by the sorted prompts); only the main process computes it, the
    others must wait (barrier) before reading it."""
    prompts = sorted(set(prompts))
    key = hashlib.sha1("\n".join(prompts).encode()).hexdigest()[:12]
    path = os.path.join(str(cache_dir), f"wan_prompt_embeds_{key}.pt")
    if is_main and not os.path.exists(path):
        os.makedirs(str(cache_dir), exist_ok=True)
        embs = encode_prompts_once(base, prompts, device)
        torch.save(embs, path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


def token_timesteps(t: torch.Tensor, b: int, nt: int, per_frame: int) -> torch.Tensor:
    """Per-token timesteps (B, nt * per_frame) in the DiT's frame-major token order; frame-0 tokens get 0. t: (B,)."""
    t = torch.as_tensor(t)
    ts = t.view(b, 1).float().expand(b, nt * per_frame).clone()
    ts[:, :per_frame] = 0
    return ts


# ---------------------------------------------------------------- model
class VideoModel:
    family = "video"

    def __init__(self, base: str | os.PathLike, device: torch.device | str, prompts: Sequence[str], cache_dir: str | os.PathLike,
                 is_main: bool = True, barrier: Callable[[], None] | None = None, dtype: torch.dtype = torch.bfloat16):
        """prompts: every prompt the task uses (embedded once, before the DiT is loaded). In a multi-process job pass
        is_main (only the main process computes the cache) and a barrier."""
        base = str(base)
        self.device, self.dtype = torch.device(device), dtype
        cache = prompt_cache(base, prompts, cache_dir, self.device, is_main)
        if barrier is not None:
            barrier()
        self.embeddings = {k: v.to(self.device) for k, v in torch.load(cache).items()}
        self.vae = AutoencoderKLWan.from_pretrained(base, subfolder="vae", torch_dtype=dtype).to(self.device).eval()
        self.vae.requires_grad_(False)
        self.transformer = WanTransformer3DModel.from_pretrained(base, subfolder="transformer", torch_dtype=dtype).to(self.device)
        self.transformer.requires_grad_(False)
        self.sched_config = UniPCMultistepScheduler.load_config(base, subfolder="scheduler")
        cfg = self.vae.config
        self.t_down, self.s_down, self.z = cfg.scale_factor_temporal, cfg.scale_factor_spatial, cfg.z_dim
        self.lat_mean = torch.tensor(cfg.latents_mean).view(1, self.z, 1, 1, 1).to(self.device, torch.float32)
        self.lat_std = torch.tensor(cfg.latents_std).view(1, self.z, 1, 1, 1).to(self.device, torch.float32)
        self.patch = tuple(self.transformer.config.patch_size)

    # ---------- LoRA ----------
    def add_trainable_lora(self, rank: int, alpha: float, targets: Sequence[str], dropout: float) -> None:
        L.add_trainable_lora(self.transformer, rank, alpha, targets, dropout)

    def load_lora(self, checkpoint: str | os.PathLike, rank: int, alpha: float, targets: Sequence[str]) -> None:
        L.load_for_inference(self.transformer, checkpoint, rank, alpha, targets, self.dtype)

    @staticmethod
    def save_lora(path: str | os.PathLike, transformer: torch.nn.Module) -> None:
        WanPipeline.save_lora_weights(str(path), transformer_lora_layers=L.state_dict(transformer))

    def embed(self, prompts: Sequence[str]) -> torch.Tensor:
        """(B, 512, 4096) prompt embeddings (every prompt must be in the set given at construction)."""
        return torch.stack([self.embeddings[p] for p in prompts])

    # ---------- clips <-> latents ----------
    def latent_shape(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Shape (C, N_BLOCKS, h, w) of one clip's latents (and of its initial sampling noise)."""
        return (self.z, N_BLOCKS, height // self.s_down, width // self.s_down)

    def blocks_to_pixels(self, blocks_u8: torch.Tensor) -> torch.Tensor:
        """(B, NB, H, W, 3) uint8 blocks -> (B, 3, 1 + (NB - 1) HOLD, H, W) in [-1, 1] (block 0 one frame, the others HOLD)."""
        x = blocks_u8.to(self.device).permute(0, 4, 1, 2, 3).float() / 127.5 - 1
        x = torch.cat([x[:, :, :1], x[:, :, 1:].repeat_interleave(HOLD, dim=2)], dim=2)
        return x.to(self.vae.dtype)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor, chunk: int = 4) -> torch.Tensor:
        """(B, 3, F, H, W) in [-1, 1] -> normalized float32 latents (B, 48, 1 + (F - 1) / 4, H / 16, W / 16) (posterior mode)."""
        out = []
        for i in range(0, pixels.shape[0], chunk):
            lat = self.vae.encode(pixels[i:i + chunk].to(self.device, self.vae.dtype)).latent_dist.mode().float()
            out.append((lat - self.lat_mean) / self.lat_std)
        return torch.cat(out)

    @torch.no_grad()
    def decode(self, lat: torch.Tensor, chunk: int = 2) -> torch.Tensor:
        """Normalized latents -> uint8 clips (B, F, H, W, 3) on the CPU."""
        out = []
        for i in range(0, lat.shape[0], chunk):
            z = (lat[i:i + chunk].float() * self.lat_std + self.lat_mean).to(self.vae.dtype)
            v = self.vae.decode(z, return_dict=False)[0]
            out.append(((v.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 4, 1).cpu())
        return torch.cat(out)

    # ---------- DiT ----------
    def forward(self, transformer: torch.nn.Module, noisy: torch.Tensor, cond0: torch.Tensor, t: torch.Tensor,
                emb: torch.Tensor) -> torch.Tensor:
        """TI2V forward. noisy (B, C, T, h, w) (frame 0 is replaced by cond0 (B, C, 1, h, w), the clean first-frame latents),
        t (B,) timesteps in [0, 1000], emb (B, 512, 4096). Returns the velocity (B, C, T, h, w) (frame 0 meaningless)."""
        b, _, nt, h, w = noisy.shape
        x = torch.cat([cond0.to(noisy.dtype), noisy[:, :, 1:]], dim=2)
        ts = token_timesteps(t, b, nt, (h // self.patch[1]) * (w // self.patch[2]))
        return transformer(hidden_states=x.to(self.dtype), timestep=ts, encoder_hidden_states=emb.to(self.dtype), return_dict=False)[0]

    # ---------- sampling ----------
    @torch.no_grad()
    def sample(self, cond0: torch.Tensor, emb: torch.Tensor, steps: int, shift: float, noise: torch.Tensor,
               transformer: torch.nn.Module | None = None) -> torch.Tensor:
        """UniPC flow sampler of WanImageToVideoPipeline (expand_timesteps branch), batched over different first frames.
        noise: (B, C, N_BLOCKS, h, w) initial latents. Returns float32 latents, frame 0 = cond0."""
        transformer = self.transformer if transformer is None else transformer
        sched = UniPCMultistepScheduler.from_config(self.sched_config, flow_shift=shift)
        sched.set_timesteps(steps, device=self.device)
        lat = noise.to(self.device, torch.float32).clone()
        b = cond0.shape[0]
        for t in sched.timesteps:
            v = self.forward(transformer, lat, cond0, t.expand(b), emb).float()
            lat = sched.step(v, t, lat, return_dict=False)[0]
        lat[:, :, :1] = cond0.float()
        return lat

    @torch.no_grad()
    def generate(self, first_frames: Sequence[Image.Image], prompts: Sequence[str], steps: int, shift: float, noise: torch.Tensor,
                 transformer: torch.nn.Module | None = None) -> torch.Tensor:
        """One clip per (first frame, prompt): uint8 (B, 65, H, W, 3) on the CPU. The whole batch is one DiT call per step."""
        cond_u8 = torch.stack([torch.from_numpy(np.array(f.convert("RGB")))[None] for f in first_frames])
        cond0 = self.encode(self.blocks_to_pixels(cond_u8))
        emb = self.embed(prompts).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            lat = self.sample(cond0, emb, steps, shift, noise, transformer)
        return self.decode(lat)

    def generate_seeded(self, first_frames: Sequence[Image.Image], prompts: Sequence[str], steps: int, shift: float,
                        seeds: Sequence[int], transformer: torch.nn.Module | None = None) -> torch.Tensor:
        """generate() with per-row noise (nbm.models.noise.row_noise) from one seed per row."""
        w, h = first_frames[0].size
        return self.generate(first_frames, prompts, steps, shift, row_noise(seeds, self.latent_shape(w, h)), transformer)

    # ---------- training ----------
    @staticmethod
    def train_sigmas(n: int, shift: float) -> torch.Tensor:
        """Noise levels of a training batch: u ~ U(0, 1) on the CPU RNG, shifted like the sampler: s = shift u / (1 + (shift - 1) u)."""
        u = torch.rand(n)
        return shift * u / (1 + (shift - 1) * u)

    def flow_loss(self, blocks_u8: torch.Tensor, emb: torch.Tensor, sigmas: torch.Tensor, noise: torch.Tensor | None = None,
                  transformer: torch.nn.Module | None = None) -> torch.Tensor:
        """Flow-matching squared error per example, latent frame 1..16 and latent cell (mean over channels): (B, 16, h, w)
        float32. Frame 0 is the clean condition and has no loss. x_t = (1 - s) x0 + s eps, target eps - x0, timestep 1000 s."""
        transformer = self.transformer if transformer is None else transformer
        x0 = self.encode(self.blocks_to_pixels(blocks_u8))
        cond0 = x0[:, :, :1]
        noise = torch.randn_like(x0) if noise is None else noise.to(x0)
        s = sigmas.view(-1, 1, 1, 1, 1).to(x0)
        noisy = (1 - s) * x0 + s * noise
        pred = self.forward(transformer, noisy, cond0, sigmas.to(x0) * 1000, emb)
        err = (pred.float() - (noise - x0)) ** 2
        return err[:, :, 1:].mean(dim=1)
