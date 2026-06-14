import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.diffusion import (
    DiffusionOutput,
    EncoderBlock,
    TimestepEmbedding,
    cosine_alpha_bar,
)


@dataclass
class DiffuSeqConfig:
    vocab_size: int = 59513
    d_model: int = 512
    n_layers: int = 12      # single transformer — no separate enc/dec
    n_heads: int = 8
    ffn_dim: int = 2048
    max_src_len: int = 128
    max_tgt_len: int = 128
    T: int = 2000

    def to_dict(self):
        return dataclasses.asdict(self)


class DiffuSeqMT(nn.Module):
    """
    DiffuSeq-style seq2seq continuous diffusion.

    Instead of a separate encoder + cross-attention decoder, we concatenate the
    clean source embeddings with the noisy target embeddings and run a single
    bidirectional transformer over the full joint sequence.  Only the target
    positions are diffused; source positions are always clean.  The denoiser
    output at target positions predicts x0 (the clean content embedding), which
    is then rounded to tokens via the tied lm_head/embedding matrix.

    Reference: Gong et al. "DiffuSeq: Sequence to Sequence Text Generation
    with Diffusion Models" (2022).
    """

    def __init__(self, config: DiffuSeqConfig):
        super().__init__()
        self.config = config
        D = config.d_model

        self.embedding = nn.Embedding(config.vocab_size, D)
        self.src_pos = nn.Embedding(config.max_src_len, D)
        self.tgt_pos = nn.Embedding(config.max_tgt_len, D)
        self.time_embed = TimestepEmbedding(D)

        # Single bidirectional transformer — source and target attend to each other
        self.transformer = nn.ModuleList([
            EncoderBlock(D, config.n_heads, config.ffn_dim)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(D)

        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)
        # Tied weights: rounding lives in the same space as the diffusion process
        self.lm_head.weight = self.embedding.weight

        self.register_buffer("alpha_bar", cosine_alpha_bar(config.T))

        self.apply(self._init_weights)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> DiffusionOutput:
        B, device = input_ids.shape[0], input_ids.device
        src_len = input_ids.shape[1]

        # --- Source side (always clean) ---
        src_pos = torch.arange(src_len, device=device).clamp(max=self.config.max_src_len - 1)
        src_emb = self.embedding(input_ids) + self.src_pos(src_pos)  # [B, src_len, D]

        # --- Target side ---
        tgt_valid = labels != -100
        x0 = self.embedding(labels.masked_fill(~tgt_valid, 0))  # content only, no pos
        tgt_len = labels.shape[1]
        tgt_pos = torch.arange(tgt_len, device=device).clamp(max=self.config.max_tgt_len - 1)

        # Forward diffusion on target content only
        t = torch.randint(1, self.config.T + 1, (B,), device=device)
        ab = self.alpha_bar[t].view(B, 1, 1)
        noise = torch.randn_like(x0)
        noisy_tgt = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise  # [B, tgt_len, D]

        # Position and timestep are conditioning on top of the noisy content
        t_emb = self.time_embed(t)  # [B, D]
        noisy_tgt_in = noisy_tgt + self.tgt_pos(tgt_pos) + t_emb.unsqueeze(1)

        # --- Concatenate and run single transformer ---
        x = torch.cat([src_emb, noisy_tgt_in], dim=1)  # [B, src_len+tgt_len, D]

        # Padding mask: True = position to ignore
        src_pad = (attention_mask == 0) if attention_mask is not None else torch.zeros(
            B, src_len, dtype=torch.bool, device=device)
        tgt_pad = ~tgt_valid
        pad_mask = torch.cat([src_pad, tgt_pad], dim=1)  # [B, src_len+tgt_len]

        for layer in self.transformer:
            x = layer(x, key_padding_mask=pad_mask)
        x = self.norm(x)

        # Target positions only — these are the x0 predictions
        tgt_out = x[:, src_len:, :]  # [B, tgt_len, D]

        # DiffusionLM losses: MSE in content-embedding space + rounding CE
        mse = F.mse_loss(tgt_out[tgt_valid], x0[tgt_valid].detach())
        ce = F.cross_entropy(
            self.lm_head(tgt_out).view(-1, self.config.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )

        return DiffusionOutput(loss=mse + ce, logits=None)

    @torch.no_grad()
    def ddim_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tgt_len: int = 64,
        steps: int = 50,
    ) -> torch.Tensor:
        B, device = input_ids.shape[0], input_ids.device
        src_len = input_ids.shape[1]

        # Source embeddings are fixed throughout sampling
        src_pos = torch.arange(src_len, device=device).clamp(max=self.config.max_src_len - 1)
        src_emb = self.embedding(input_ids) + self.src_pos(src_pos)

        src_pad = (attention_mask == 0) if attention_mask is not None else torch.zeros(
            B, src_len, dtype=torch.bool, device=device)
        tgt_pad = torch.zeros(B, tgt_len, dtype=torch.bool, device=device)
        pad_mask = torch.cat([src_pad, tgt_pad], dim=1)

        tgt_pos = torch.arange(tgt_len, device=device).clamp(max=self.config.max_tgt_len - 1)

        # Start from pure noise in content-embedding space
        noisy_tgt = torch.randn(B, tgt_len, self.config.d_model, device=device)
        ts = torch.linspace(self.config.T, 1, steps, dtype=torch.long, device=device)

        x_scale = self.embedding.weight.detach().abs().max()

        logits = None
        for i, t_val in enumerate(ts):
            t = t_val.expand(B)
            t_emb = self.time_embed(t)

            noisy_tgt_in = noisy_tgt + self.tgt_pos(tgt_pos) + t_emb.unsqueeze(1)

            x = torch.cat([src_emb, noisy_tgt_in], dim=1)
            for layer in self.transformer:
                x = layer(x, key_padding_mask=pad_mask)
            x = self.norm(x)

            tgt_out = x[:, src_len:, :]       # [B, tgt_len, D]
            logits = self.lm_head(tgt_out)     # [B, tgt_len, V]

            # Soft rounding with temperature annealing (1.0 → 0.1 over sampling steps).
            # Hard argmax at each step causes a collapse cascade: once a high-frequency
            # token wins the trajectory is pulled toward its embedding, making it dominate
            # every subsequent step. The soft weighted-average x0 stays on the token
            # manifold while allowing the distribution to shift as denoising progresses.
            temperature = max(t_val.item() / self.config.T, 0.1)
            probs = F.softmax(logits / temperature, dim=-1)       # [B, tgt_len, V]
            x0_soft = probs @ self.embedding.weight               # [B, tgt_len, D]

            ab_t = self.alpha_bar[t_val].clamp(min=1e-8)
            ab_next = (self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else self.alpha_bar[0]).clamp(min=0.0)

            direction = (noisy_tgt - ab_t.sqrt() * x0_soft) / (1.0 - ab_t).clamp(min=1e-8).sqrt()
            noisy_tgt = ab_next.sqrt() * x0_soft + (1.0 - ab_next).clamp(min=0.0).sqrt() * direction
            noisy_tgt = noisy_tgt.clamp(-x_scale, x_scale)

        # Final hard argmax on the last logits (temperature ≈ 0.1, nearly deterministic)
        return logits.argmax(dim=-1)  # [B, tgt_len]
