"""Qwen-Image-2.1 as an image-edit model: load (+ LoRA), encode, flow-matching loss, Euler sampler.

The forward pass follows diffusers' QwenImage21Pipeline / its img2img LoRA recipe: the DiT sequence is [prompt tokens, with the
condition picture's vision tokens from the Qwen3-VL text encoder | clean condition latents | noisy target latents], and the
prediction is read on the target tail. The sampler is re-implemented so that a batch of *different* condition pictures is
denoised at once (the stock pipeline takes one condition per call).

Devices: `device` holds the DiT and everything it touches; `text_device` (default: the same GPU) holds the frozen Qwen3-VL text
encoder and the VAE. Splitting them over two 24 GB GPUs is what lets training and evaluation fit (text encoder ~17.5 GB + VAE
on one, DiT ~14 GB + LoRA state on the other).

Numerics (must not change: the released checkpoints and the paper's records depend on them): bfloat16 weights, the VAE
posterior mode for condition latents, a posterior sample for training targets, the Euler flow sampler with the pipeline's
schedule (linearly spaced sigmas from 1 to 1/steps, dynamic shift), one step = sigma 1 -> 0, and everything of `generate` under
bfloat16 autocast. Outputs depend (at bfloat16 rounding level) on the batch composition, so to reproduce a record use the same
batches.
"""
from __future__ import annotations

import copy
import os
from typing import Sequence

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from nbm.data import to_vae_tensor
from nbm.models import lora as L
from nbm.models.noise import row_noise

LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out")


def box_mask(boxes: Sequence[Sequence[Sequence[int]]], h: int, w: int, width: int, height: int,
             device: torch.device | str) -> torch.Tensor:
    """Per example a list of pixel boxes (x0, y0, x1, y1) of a width x height picture -> (B, 1, 1, h, w) mask of the latent
    cells they touch (a partly covered cell counts)."""
    m = torch.zeros(len(boxes), 1, 1, h, w, device=device)
    for b, bx in enumerate(boxes):
        for x0, y0, x1, y1 in bx:
            m[b, :, :, y0 * h // height:-(-y1 * h // height), x0 * w // width:-(-x1 * w // width)] = 1
    return m


class ImageModel:
    family = "image"

    def __init__(self, base: str | os.PathLike, device: torch.device | str, text_device: torch.device | str | None = None,
                 dtype: torch.dtype = torch.bfloat16):
        base = str(base)
        self.device, self.dtype = torch.device(device), dtype
        self.text_device = self.device if text_device is None else torch.device(text_device)
        # the VAE decoder's activations are large next to the resident text encoder (~19 GB of 24 when split): smaller chunks
        self.decode_chunk = 4 if self.text_device == self.device else 1
        self.processor = Qwen3VLProcessor.from_pretrained(base, subfolder="processor")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler")
        self.vae = AutoencoderKLQwenImage21.from_pretrained(base, subfolder="vae", torch_dtype=dtype).to(self.text_device)
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype).to(self.text_device)
        self.transformer = QwenImage21Transformer2DModel.from_pretrained(base, subfolder="transformer", torch_dtype=dtype).to(self.device)
        for m in (self.vae, self.text_encoder, self.transformer):
            m.requires_grad_(False)
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        z = self.vae.config.z_dim
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, z, 1, 1, 1).to(self.text_device, dtype)
        self.latents_std = torch.tensor(self.vae.config.latents_std).view(1, z, 1, 1, 1).to(self.text_device, dtype)
        # text-only pipeline object, used for its encode_prompt (prompt template, vision tokens, final-norm hook)
        self.text_pipe = QwenImage21Pipeline.from_pretrained(base, vae=None, transformer=None, processor=self.processor,
                                                             text_encoder=self.text_encoder, scheduler=None)

    # ---------- LoRA ----------
    def add_trainable_lora(self, rank: int, alpha: float, targets: Sequence[str], dropout: float) -> None:
        L.add_trainable_lora(self.transformer, rank, alpha, targets, dropout)

    def load_lora(self, checkpoint: str | os.PathLike, rank: int, alpha: float, targets: Sequence[str]) -> None:
        L.load_for_inference(self.transformer, checkpoint, rank, alpha, targets, self.dtype)

    @staticmethod
    def save_lora(path: str | os.PathLike, transformer: torch.nn.Module) -> None:
        QwenImage21Pipeline.save_lora_weights(str(path), transformer_lora_layers=L.state_dict(transformer))

    # ---------- pictures <-> latents ----------
    to_vae_tensor = staticmethod(to_vae_tensor)

    def latent_shape(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Shape (C, 1, h, w) of one picture's latents (and of its initial sampling noise)."""
        return (self.vae.config.z_dim, 1, height // self.vae_scale_factor, width // self.vae_scale_factor)

    @torch.no_grad()
    def encode_latents(self, pixels: torch.Tensor, sample: bool) -> torch.Tensor:
        """(B, 4, H, W) -> normalized latents (B, C, 1, h, w) on the DiT device; sample=False: posterior mode."""
        dist = self.vae.encode(pixels.to(self.text_device, self.dtype).unsqueeze(2)).latent_dist
        if not sample:
            lat = dist.mode()
        elif self.text_device == self.device:
            lat = dist.sample()
        else:
            # split devices: draw the posterior noise from the DiT device's RNG exactly as dist.sample() does on one GPU,
            # so a two-GPU run consumes the same random stream as a one-GPU run
            eps = torch.randn(dist.mean.shape, device=self.device, dtype=dist.parameters.dtype).to(self.text_device)
            lat = dist.mean + dist.std * eps
        return ((lat - self.latents_mean) / self.latents_std).to(self.dtype).to(self.device)

    @torch.no_grad()
    def decode_latents(self, lat: torch.Tensor) -> list[Image.Image]:
        out = []
        for i in range(0, lat.shape[0], self.decode_chunk):
            z = (lat[i:i + self.decode_chunk].to(self.text_device) * self.latents_std + self.latents_mean).to(self.vae.dtype)
            img = self.vae.decode(z, return_dict=False)[0][:, :, 0]
            arr = ((img[:, :3].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            out += [Image.fromarray(a) for a in arr]
        return out

    # ---------- prompts ----------
    @torch.no_grad()
    def encode_prompts(self, prompts: Sequence[str], cond_images: Sequence[Image.Image]):
        """Each prompt is encoded together with its own condition picture. All prompts of one call must share a token layout
        (same text length and picture size), e.g. one side to move per batch. Returns (emb, mask or None, image-slot mask)."""
        embs, masks, pads = [], [], []
        for p, img in zip(prompts, cond_images):
            e, m, pad = self.text_pipe.encode_prompt(prompt=p, image=[img.convert("RGB")], device=self.text_device)
            e = e.to(self.device)
            embs.append(e)
            masks.append(m.to(self.device) if m is not None else torch.ones(e.shape[:2], dtype=torch.long, device=self.device))
            pads.append(pad.to(self.device))
        if any(e.shape != embs[0].shape for e in embs) or any(not torch.equal(p, pads[0]) for p in pads):
            raise ValueError("prompt layouts differ inside a batch: group the rows so that one batch shares a prompt layout")
        emb, mask = torch.cat(embs), torch.cat(masks)
        return emb, (None if bool(mask.all()) else mask), pads[0].repeat(len(prompts), 1).to(self.device)

    # ---------- DiT ----------
    def forward(self, transformer: torch.nn.Module, noisy: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor,
                emb: torch.Tensor, emb_mask: torch.Tensor | None, pad_mask: torch.Tensor, kv_cache=None, kv_mode=None) -> torch.Tensor:
        """noisy / cond: (B, C, 1, h, w) normalized latents; timesteps in [0, 1000]. Returns the velocity (B, C, 1, h, w)."""
        b, c, _, h, w = noisy.shape
        ch, cw = cond.shape[3], cond.shape[4]
        packed_noisy = QwenImage21Pipeline._pack_latents(noisy, b, c, h, w)
        packed_cond = QwenImage21Pipeline._pack_latents(cond, b, c, ch, cw)
        if int(pad_mask[0].sum()) != (ch * cw) // 4:
            raise ValueError(f"prompt has {int(pad_mask[0].sum())} image slots, condition latents need {(ch * cw) // 4}")
        img_mask = torch.cat([pad_mask, pad_mask.new_ones(b, (h * w) // 4)], dim=1)
        out = transformer(
            hidden_states=torch.cat([packed_cond, packed_noisy], dim=1),
            encoder_hidden_states=emb,
            encoder_hidden_states_mask=emb_mask,
            timestep=timesteps.to(noisy.dtype) / 1000,
            img_shapes=[[(1, ch, cw), (1, h, w)]] * b,
            img_mask=img_mask,
            kv_cache=kv_cache,
            kv_cache_mode=kv_mode,
            return_dict=False,
        )[0]
        out = out[:, -packed_noisy.shape[1]:]
        return QwenImage21Pipeline._unpack_latents(out, h * self.vae_scale_factor, w * self.vae_scale_factor, self.vae_scale_factor)

    # ---------- sampling ----------
    @torch.no_grad()
    def sample(self, cond_lat: torch.Tensor, emb: torch.Tensor, emb_mask: torch.Tensor | None, pad_mask: torch.Tensor,
               steps: int, noise: torch.Tensor, transformer: torch.nn.Module | None = None) -> torch.Tensor:
        """Euler flow-matching sampler with QwenImage21Pipeline's schedule. noise: (B, C, 1, h, w) initial latents. The prompt /
        condition prefix is computed once and cached (the DiT's causal-condition KV cache)."""
        transformer = self.transformer if transformer is None else transformer
        b, c, _, h, w = cond_lat.shape
        sched = copy.deepcopy(self.scheduler)
        mu = calculate_shift(h * w, sched.config.get("base_image_seq_len", 256), sched.config.get("max_image_seq_len", 4096),
                             sched.config.get("base_shift", 0.5), sched.config.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(sched, steps, self.device, sigmas=np.linspace(1.0, 1 / steps, steps), mu=mu)
        if steps == 1:  # the pipeline's terminal-shift stretch divides 0 by 0 for a single sigma; one Euler step is sigma 1 -> 0
            sched.sigmas = torch.tensor([1.0, 0.0], device=sched.sigmas.device, dtype=sched.sigmas.dtype)
            sched.timesteps = (sched.sigmas[:1] * sched.config.num_train_timesteps).to(self.device)
            timesteps = sched.timesteps
        sched.set_begin_index(0)
        if tuple(noise.shape) != (b, c, 1, h, w):
            raise ValueError(f"noise {tuple(noise.shape)} != latents {(b, c, 1, h, w)}")
        lat = noise.to(self.device, torch.float32).to(self.dtype)
        cache = QwenImage21KVCache(len(transformer.transformer_blocks)) if transformer.config.causal_condition else None
        for i, t in enumerate(timesteps):
            mode = None if cache is None else ("extract" if i == 0 else "cached")
            v = self.forward(transformer, lat, cond_lat, t.expand(b), emb, emb_mask, pad_mask, cache, mode)
            # the scheduler works on packed (B, L, C) latents: pack, step, unpack
            pv = QwenImage21Pipeline._pack_latents(v, b, c, h, w)
            pl = QwenImage21Pipeline._pack_latents(lat, b, c, h, w)
            pl = sched.step(pv, t, pl, return_dict=False)[0]
            lat = QwenImage21Pipeline._unpack_latents(pl, h * self.vae_scale_factor, w * self.vae_scale_factor,
                                                      self.vae_scale_factor).to(self.dtype)
        return lat

    @torch.no_grad()
    def generate(self, conds: Sequence[Image.Image], prompts: Sequence[str], steps: int, noise: torch.Tensor,
                 transformer: torch.nn.Module | None = None) -> list[Image.Image]:
        """One output picture per (condition picture, prompt), denoised from `noise` (B, C, 1, h, w) in `steps` Euler steps.
        The whole batch is one DiT call per step: the caller chooses the batches (one prompt layout per batch)."""
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            emb, mask, pad = self.encode_prompts(prompts, conds)
            cond_lat = self.encode_latents(torch.stack([self.to_vae_tensor(c) for c in conds]), sample=False)
            lat = self.sample(cond_lat, emb, mask, pad, steps, noise, transformer)
            return self.decode_latents(lat)

    def generate_seeded(self, conds: Sequence[Image.Image], prompts: Sequence[str], steps: int, seeds: Sequence[int],
                        transformer: torch.nn.Module | None = None) -> list[Image.Image]:
        """generate() with per-row noise (nbm.models.noise.row_noise) from one seed per row."""
        w, h = conds[0].size
        return self.generate(conds, prompts, steps, row_noise(seeds, self.latent_shape(w, h)), transformer)

    # ---------- training ----------
    @staticmethod
    def train_sigmas(scheduler: FlowMatchEulerDiscreteScheduler, n: int) -> torch.Tensor:
        """Noise levels of a training batch: u ~ U(0, 1) on the CPU RNG, then the scheduler's training sigma at index
        floor(1000 u) (the diffusers img2img LoRA recipe with weighting scheme "none")."""
        u = torch.rand(size=(n,))
        idx = (u * scheduler.config.num_train_timesteps).long().clamp(max=scheduler.config.num_train_timesteps - 1)
        return scheduler.sigmas[idx]

    def flow_loss(self, batch: dict, sigmas: torch.Tensor, highlight_weight: float, noise: torch.Tensor | None = None,
                  transformer: torch.nn.Module | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rectified-flow velocity MSE on the target latents, per example: (total, highlight region, rest).

        batch: cond / target (B, 4, H, W) VAE tensors, cond_pil (condition pictures), prompts, boxes (per example a list of
        pixel boxes = the highlight region). total weights the latents of the highlight region by highlight_weight
        (1 = plain mean). x_t = (1 - s) x0 + s eps, target velocity eps - x0, timestep 1000 s."""
        transformer = self.transformer if transformer is None else transformer
        cond_lat = self.encode_latents(batch["cond"], sample=False)
        x0 = self.encode_latents(batch["target"], sample=True)
        emb, emb_mask, pad = self.encode_prompts(batch["prompts"], batch["cond_pil"])
        noise = torch.randn_like(x0) if noise is None else noise.to(x0)
        s = sigmas.view(-1, 1, 1, 1, 1).to(x0.dtype)
        noisy = (1 - s) * x0 + s * noise
        pred = self.forward(transformer, noisy, cond_lat, sigmas * 1000, emb, emb_mask, pad)
        err = (pred.float() - (noise - x0).float()) ** 2
        hm = box_mask(batch["boxes"], x0.shape[3], x0.shape[4], batch["cond"].shape[-1], batch["cond"].shape[-2], x0.device)
        w = 1 + (highlight_weight - 1) * hm
        total = (err * w).flatten(1).sum(1) / (w.expand_as(err)).flatten(1).sum(1)
        hl = (err * hm).flatten(1).sum(1) / hm.expand_as(err).flatten(1).sum(1)
        rest = (err * (1 - hm)).flatten(1).sum(1) / (1 - hm).expand_as(err).flatten(1).sum(1)
        return total, hl, rest
