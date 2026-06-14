import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput


@dataclass
class DiffusionConfig:
    vocab_size: int = 59513
    d_model: int = 512
    n_enc_layers: int = 6
    n_den_layers: int = 6
    n_heads: int = 8
    ffn_dim: int = 2048
    max_len: int = 128
    T: int = 2000  # forward diffusion timesteps

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class DiffusionOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos((t / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
    return (f / f[0]).clamp(0.0, 1.0)  # shape [T+1], index with timestep t


class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)    # [B, d_model]
        return self.proj(emb)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class DenoiserBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        enc_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x)
        x = x + self.cross_attn(h, encoder_out, encoder_out, key_padding_mask=enc_pad_mask, need_weights=False)[0]
        x = x + self.ffn(self.norm3(x))
        return x


class TextDiffusionMT(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        D = config.d_model

        self.embedding = nn.Embedding(config.vocab_size, D)
        self.enc_pos = nn.Embedding(config.max_len, D)
        self.den_pos = nn.Embedding(config.max_len, D)
        self.time_embed = TimestepEmbedding(D)

        self.encoder = nn.ModuleList([EncoderBlock(D, config.n_heads, config.ffn_dim) for _ in range(config.n_enc_layers)])
        self.enc_norm = nn.LayerNorm(D)

        self.denoiser = nn.ModuleList([DenoiserBlock(D, config.n_heads, config.ffn_dim) for _ in range(config.n_den_layers)])
        self.den_norm = nn.LayerNorm(D)

        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)
        self.x0_cond_proj = nn.Linear(D, D, bias=False)  # self-conditioning projection

        # DiffusionLM: tie lm_head and embedding weights so rounding lives in the same
        # space as the diffusion process. MSE pulls x0_pred toward embedding[label];
        # CE (via lm_head = embedding^T) pulls it toward high dot-product with embedding[label].
        # With separate weights these two objectives point in different directions.
        self.lm_head.weight = self.embedding.weight

        self.register_buffer("alpha_bar", cosine_alpha_bar(config.T))  # [T+1]

        self.apply(self._init_weights)
        # Re-init after weight tying so the shared matrix gets the embedding init (std=0.02)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).clamp(max=self.config.max_len - 1)
        x = self.embedding(input_ids) + self.enc_pos(pos)
        pad_mask = None if attention_mask is None else (attention_mask == 0)
        for layer in self.encoder:
            x = layer(x, key_padding_mask=pad_mask)
        return self.enc_norm(x), pad_mask

    def _denoise(self, x_t: torch.Tensor, t_emb: torch.Tensor, encoder_out: torch.Tensor, enc_pad_mask, x0_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x_t.shape
        pos = torch.arange(T, device=x_t.device).clamp(max=self.config.max_len - 1)
        x = x_t + self.den_pos(pos) + t_emb.unsqueeze(1)
        if x0_cond is not None:
            x = x + self.x0_cond_proj(x0_cond)
        for layer in self.denoiser:
            x = layer(x, encoder_out, enc_pad_mask)
        return self.den_norm(x)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> DiffusionOutput:
        B, device = input_ids.shape[0], input_ids.device

        enc_out, enc_pad_mask = self._encode(input_ids, attention_mask)

        # Embed targets; labels use -100 for padding, substitute 0 for safe lookup
        tgt_valid = labels != -100
        x0 = self.embedding(labels.masked_fill(~tgt_valid, 0))

        # Forward diffusion: x_t = sqrt(ab) * x0 + sqrt(1-ab) * noise
        t = torch.randint(1, self.config.T + 1, (B,), device=device)
        ab = self.alpha_bar[t].view(B, 1, 1)
        noise = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

        t_emb = self.time_embed(t)

        # Self-conditioning: 50% of batches get a detached first-pass x0 estimate as extra input
        x0_cond = torch.zeros_like(x0)
        if self.training and torch.rand(1).item() < 0.5:
            with torch.no_grad():
                x0_cond = self._denoise(x_t, t_emb, enc_out, enc_pad_mask, x0_cond=None).detach()

        x0_pred = self._denoise(x_t, t_emb, enc_out, enc_pad_mask, x0_cond=x0_cond)

        # DiffusionLM loss: MSE in embedding space + rounding CE in the same space.
        # lm_head.weight == embedding.weight (tied), so both losses agree on where x0_pred should land.
        mse = F.mse_loss(x0_pred[tgt_valid], x0[tgt_valid].detach())
        ce = F.cross_entropy(self.lm_head(x0_pred).view(-1, self.config.vocab_size),
                             labels.view(-1), ignore_index=-100)

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
        enc_out, enc_pad_mask = self._encode(input_ids, attention_mask)

        x = torch.randn(B, tgt_len, self.config.d_model, device=device)
        ts = torch.linspace(self.config.T, 1, steps, dtype=torch.long, device=device)

        # Clamp range derived from the embedding table so x stays near the token manifold.
        x_scale = self.embedding.weight.detach().abs().max()

        x0_cond = torch.zeros(B, tgt_len, self.config.d_model, device=device)
        logits = None
        for i, t_val in enumerate(ts):
            t = t_val.expand(B)
            x0_pred = self._denoise(x, self.time_embed(t), enc_out, enc_pad_mask, x0_cond=x0_cond)

            logits = self.lm_head(x0_pred)  # [B, tgt_len, V]

            # Soft rounding with temperature annealing: avoids the hard-argmax collapse
            # where the trajectory locks onto one token and never recovers.
            temperature = max(t_val.item() / self.config.T, 0.1)
            probs = F.softmax(logits / temperature, dim=-1)
            x0_soft = probs @ self.embedding.weight  # [B, tgt_len, D]
            x0_cond = x0_soft

            ab_t = self.alpha_bar[t_val].clamp(min=1e-8)
            ab_next = (self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else self.alpha_bar[0]).clamp(min=0.0)

            direction = (x - ab_t.sqrt() * x0_soft) / (1.0 - ab_t).clamp(min=1e-8).sqrt()
            x = ab_next.sqrt() * x0_soft + (1.0 - ab_next).clamp(min=0.0).sqrt() * direction
            x = x.clamp(-x_scale, x_scale)

        return logits.argmax(dim=-1)  # [B, tgt_len]
