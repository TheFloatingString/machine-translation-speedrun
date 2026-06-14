import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

from src.diffusion import EncoderBlock, DenoiserBlock, TimestepEmbedding


@dataclass
class MDLMConfig:
    vocab_size: int = 59513
    d_model: int = 512
    n_enc_layers: int = 6
    n_den_layers: int = 6
    n_heads: int = 8
    ffn_dim: int = 2048
    max_len: int = 128
    T: int = 1000  # masking schedule steps

    @property
    def mask_id(self) -> int:
        return self.vocab_size  # extra embedding slot; lm_head predicts over [0, vocab_size)

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class MDLMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class MaskedDiffusionMT(nn.Module):
    """
    Absorbing (masked) discrete diffusion for seq2seq MT.

    Forward process: linearly increase the fraction of masked target tokens as t→T.
    Reverse process: starting from all-[MASK], iteratively unmask tokens via the
    absorbing diffusion reverse kernel p(x_{t-1} | x_t, x0_pred).
    """

    def __init__(self, config: MDLMConfig):
        super().__init__()
        self.config = config
        D = config.d_model

        # vocab_size + 1 embeddings: indices [0, vocab_size-1] are real tokens,
        # index vocab_size is the [MASK] token used during diffusion.
        self.embedding = nn.Embedding(config.vocab_size + 1, D)
        self.enc_pos = nn.Embedding(config.max_len, D)
        self.den_pos = nn.Embedding(config.max_len, D)
        self.time_embed = TimestepEmbedding(D)

        self.encoder = nn.ModuleList([
            EncoderBlock(D, config.n_heads, config.ffn_dim)
            for _ in range(config.n_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(D)

        self.denoiser = nn.ModuleList([
            DenoiserBlock(D, config.n_heads, config.ffn_dim)
            for _ in range(config.n_den_layers)
        ])
        self.den_norm = nn.LayerNorm(D)

        self.lm_head = nn.Linear(D, config.vocab_size, bias=False)

        self.apply(self._init_weights)

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

    def _denoise(
        self,
        noisy_ids: torch.Tensor,
        t_emb: torch.Tensor,
        encoder_out: torch.Tensor,
        enc_pad_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, L = noisy_ids.shape
        pos = torch.arange(L, device=noisy_ids.device).clamp(max=self.config.max_len - 1)
        x = self.embedding(noisy_ids) + self.den_pos(pos) + t_emb.unsqueeze(1)
        for layer in self.denoiser:
            x = layer(x, encoder_out, enc_pad_mask)
        return self.lm_head(self.den_norm(x))  # [B, L, vocab_size]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> MDLMOutput:
        B, device = input_ids.shape[0], input_ids.device
        mask_id = self.config.mask_id

        enc_out, enc_pad_mask = self._encode(input_ids, attention_mask)

        # Linear masking schedule: at timestep t, each token is masked independently
        # with probability t/T.
        t = torch.randint(1, self.config.T + 1, (B,), device=device)
        mask_prob = t.float() / self.config.T  # [B]

        valid = labels != -100  # [B, L] — positions that are real target tokens
        rand_mask = torch.rand(labels.shape, device=device) < mask_prob.unsqueeze(1)

        noisy_labels = labels.clone()
        noisy_labels[~valid] = mask_id           # padding → always mask
        noisy_labels[valid & rand_mask] = mask_id  # random valid positions → mask

        t_emb = self.time_embed(t)
        logits = self._denoise(noisy_labels, t_emb, enc_out, enc_pad_mask)  # [B, L, vocab_size]

        # Loss only over masked valid positions
        loss_mask = valid & rand_mask
        if loss_mask.any():
            loss = F.cross_entropy(logits[loss_mask], labels[loss_mask])
        else:
            loss = (logits * 0).sum()  # rare edge case; keeps gradient graph alive

        return MDLMOutput(loss=loss, logits=None)

    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tgt_len: int = 64,
        steps: int = 50,
        repetition_penalty: float = 1.3,
    ) -> torch.Tensor:
        """
        Absorbing diffusion reverse with confidence-based unmasking (MaskGIT-style),
        temperature annealing, and a repetition penalty.

        Temperature t_val/T starts at 1.0 (diverse) and decays toward 0 (near-argmax)
        so early steps explore more and late steps commit confidently.  The repetition
        penalty discourages re-predicting already-committed tokens, preventing the
        all-same-token cascade common in early training.
        """
        B, device = input_ids.shape[0], input_ids.device
        mask_id = self.config.mask_id
        T = self.config.T
        stride = T // steps

        enc_out, enc_pad_mask = self._encode(input_ids, attention_mask)

        x = torch.full((B, tgt_len), mask_id, dtype=torch.long, device=device)

        for step in range(steps):
            t_val = T - step * stride
            temperature = max(t_val / T, 1e-3)  # 1.0 → ~0 over the course of sampling

            t_cur = torch.full((B,), t_val, dtype=torch.long, device=device)
            logits = self._denoise(x, self.time_embed(t_cur), enc_out, enc_pad_mask)  # [B, L, V]

            # Penalise already-committed tokens to break repetition cascades
            penalized = logits.clone()
            for b in range(B):
                committed = x[b][x[b] != mask_id].unique()
                if committed.numel() > 0:
                    tok_log = penalized[b, :, committed]
                    penalized[b, :, committed] = torch.where(
                        tok_log > 0, tok_log / repetition_penalty, tok_log * repetition_penalty
                    )

            probs = torch.softmax(penalized / temperature, dim=-1)  # [B, L, V]
            preds = probs.argmax(dim=-1)            # [B, L] greedy after penalty+temp
            confidence = probs.max(dim=-1).values   # [B, L] for ordering

            still_masked = x == mask_id

            # Number to unmask: stride/t_val fraction of remaining masked tokens
            n_masked = still_masked.sum(dim=-1).float()                      # [B]
            n_to_unmask = (n_masked * stride / max(t_val, 1)).ceil().long()  # [B]

            # Rank masked positions by model confidence; commit the most certain ones
            masked_conf = confidence.masked_fill(~still_masked, -float("inf"))
            _, sorted_idx = masked_conf.sort(dim=-1, descending=True)
            for b in range(B):
                k = int(n_to_unmask[b].item())
                if k > 0:
                    x[b, sorted_idx[b, :k]] = preds[b, sorted_idx[b, :k]]

        # Deterministically resolve any residual masks (integer rounding edge case)
        still_masked = x == mask_id
        if still_masked.any():
            t_one = torch.ones(B, dtype=torch.long, device=device)
            logits = self._denoise(x, self.time_embed(t_one), enc_out, enc_pad_mask)
            x[still_masked] = logits.argmax(-1)[still_masked]

        return x  # [B, tgt_len]
