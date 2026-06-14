import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
import math
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class RMSNorm(nn.Module):
    def __init__(self, dims, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x):
        # x: (B, T, C)
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

class Attention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        # Flag for special GPT-2 residual scaling
        self.proj.RESIDUAL_SCALE_FLAG = True

    def _apply_rope(self, x, seq_len, device):
        # x shape: (B, n_head, T, head_dim)
        dim = x.shape[-1]
        # Calculate frequencies: 10000^(-2i/dim)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(seq_len, device=device).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)  # (T, dim/2)
        
        # Split into real and imaginary parts (simulated via pairs)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim)
        cos = emb.cos()[None, None, :, :]  # (1, 1, T, dim)
        sin = emb.sin()[None, None, :, :]  # (1, 1, T, dim)
        
        # Rotary transformation: x_rot = x * cos + x_perp * sin
        # where x_perp = [-x1, x0, -x3, x2, ...]
        x_perp = torch.empty_like(x)
        x_perp[..., 0::2] = -x[..., 1::2]
        x_perp[..., 1::2] = x[..., 0::2]
        
        return (x * cos) + (x_perp * sin)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Split heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Queries and Keys
        q = self._apply_rope(q, T, x.device)
        k = self._apply_rope(k, T, x.device)
        
        # Memory-efficient Scaled Dot-Product Attention (includes Flash Attention)
        # is_causal=True automatically handles the triangular masking
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True,
            dropout_p=0.0 if not self.training else 0.1
        )
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # SwiGLU: GLU variant using Swish (SiLU)
        # Increases parameter count by 50% for same hidden dim, 
        # but here we'll keep the 4*n_embd expansion for the gated part
        self.c_fc = nn.Linear(n_embd, 2 * 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        # Flag for special GPT-2 residual scaling
        self.c_proj.RESIDUAL_SCALE_FLAG = True

    def forward(self, x):
        x = self.c_fc(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = Attention(n_embd, n_head)
        self.ln_2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

from transformers import PretrainedConfig

class GPT2Config(PretrainedConfig):
    model_type = "gpt2"
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        model_params=0,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.model_params = model_params
        # Standard HF names for compatibility with generate()
        self.num_hidden_layers = n_layer
        self.hidden_size = n_embd
        self.num_attention_heads = n_head
        super().__init__(**kwargs)

from transformers import GenerationMixin, GenerationConfig

class GPT2Modded(nn.Module, GenerationMixin):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList(
                [
                    TransformerBlock(config.n_embd, config.n_head)
                    for _ in range(config.n_layer)
                ]
            ),
            "ln_f": RMSNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config
        self.main_input_name = "input_ids"
        self.generation_config = GenerationConfig()
        self.gradient_checkpointing = False
        
        # Proper weight initialization
        self.apply(self._init_weights)

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # Apply GPT-2 residual scaling for stability in deep models
            if hasattr(module, 'RESIDUAL_SCALE_FLAG'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        B, T = input_ids.size()
        
        # Token embeddings only (RoPE is applied inside Attention blocks)
        x = self.transformer.wte(input_ids)
        
        for block in self.transformer.h:
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tensors
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )
