"""
fluid_lm.py  (v2) — FluidLM with pre-norm and stable residuals
===============================================================

Changes from v1
───────────────
  Pre-norm residual (critical stability fix)
    v1: u = layer(u)           → activation magnitude grows each layer
    v2: u = u + layer(norm(u)) − norm(u)
            ↑ residual on original ↑ layer sees normalised input

    Equivalently: the NS step adds Δt·F(norm(u)) to the unnormalised u.
    This keeps embedding norms bounded through all 12 layers without
    needing gradient clipping tricks.

  Separate pre-norms per layer
    nn.ModuleList of LayerNorms, one per NS layer.
    Each layer has its own normalisation state → stable at init.

  Final LayerNorm unchanged
    self.norm before lm_head remains (standard practice).

Architecture is otherwise identical to v1 (weight tying, adaptive depth,
PHYSICS_NAMES optimizer grouping, generate() method).

Parameter count increase vs v1
───────────────────────────────
  Pre-norms:      n_layers × 2 × d_model        (weight + bias)
  Channel mix:    n_layers × d_model²            (zero-init Linear)
  Total (S scale, d=768, L=12):
    Pre-norms:    12 × 2 × 768      =    18 432
    Channel mix:  12 × 768²         = 7 077 888  (+7.1M)
    Net increase: ~7.1M params on ~96M base → ~7.4%
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .ns_layer import FluidLayer


# Physics parameter names for 3-group optimizer
PHYSICS_NAMES = frozenset({"log_nu", "log_dt", "log_alpha", "p_scale_raw"})


class FluidLM(nn.Module):
    """
    FluidLM v2: pre-norm, channel mixing, Euler-default NS layers.

    Parameters
    ----------
    vocab_size : int
    d_model : int
        Embedding / velocity-field dimension.
    n_layers : int
        Number of NS time steps.
    max_seq_len : int
    nu, dt, alpha : float
        Initial physics parameters (shared across layers at init).
    integrator : str
        'euler' (default) or 'rk4'.
    use_channel_mix : bool
        Enable cross-channel linear mixing after each NS step.
    convergence_threshold : float
        ΔKE below which adaptive mode declares convergence.
    min_steps : int
        Minimum layers to run in adaptive mode.
    dropout : float
        Embedding dropout rate.
    causal : bool
        Autoregressive (True) or bidirectional (False).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 12,
        max_seq_len: int = 512,
        nu: float = 0.01,
        dt: float = 0.1,
        alpha: float = 1.0,
        integrator: str = "euler",
        use_channel_mix: bool = True,
        convergence_threshold: float = 1e-4,
        min_steps: int = 3,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()

        self.d_model              = d_model
        self.n_layers             = n_layers
        self.convergence_threshold = convergence_threshold
        self.min_steps            = min_steps

        # Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)

        # Pre-norm: one LayerNorm per NS layer (v2 addition)
        self.pre_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # NS layers
        self.layers = nn.ModuleList([
            FluidLayer(
                d_model,
                nu=nu, dt=dt, alpha=alpha,
                integrator=integrator,
                causal=causal,
                use_channel_mix=use_channel_mix,
            )
            for _ in range(n_layers)
        ])

        # Output
        self.max_seq_len = max_seq_len
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (p for p in self.parameters() if p.requires_grad) \
                 if trainable_only else self.parameters()
        return sum(p.numel() for p in params)

    # ── Optimizer parameter groups ────────────────────────────────────────────

    def make_optimizer_groups(self, lr: float, weight_decay: float = 0.1):
        """
        Three-group optimizer setup:
          physics  — log_nu, log_dt, log_alpha, p_scale_raw  (10× LR, wd=0.01)
          decay    — all other weight matrices (dim ≥ 2)      (1×  LR, wd=0.1)
          nodecay  — biases, norms, embeddings (dim < 2)      (1×  LR, wd=0.0)

        Usage:
            groups  = model.make_optimizer_groups(lr=3e-4)
            opt     = torch.optim.AdamW(groups)
            # scheduler must scale pg["lr"] = lr_now * pg.get("lr_scale", 1.0)
        """
        physics, decay, nodecay = [], [], []
        seen = set()

        for name, param in self.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            short = name.split(".")[-1]
            if short in PHYSICS_NAMES:
                physics.append(param)
            elif param.dim() >= 2:
                decay.append(param)
            else:
                nodecay.append(param)

        return [
            {"params": physics, "lr": lr, "lr_scale": 10.0, "weight_decay": 0.01},
            {"params": decay,   "lr": lr, "lr_scale":  1.0, "weight_decay": weight_decay},
            {"params": nodecay, "lr": lr, "lr_scale":  1.0, "weight_decay": 0.0},
        ]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        adaptive: bool = False,
        return_stats: bool = False,
    ):
        """
        Forward pass.

        Parameters
        ----------
        input_ids : [B, L]
        adaptive : bool
            Stop early when ΔKE < convergence_threshold (after min_steps).
        return_stats : bool
            Also return {'n_steps': int, 'delta_kes': List[float]}.

        Pre-norm residual pattern (v2)
        ──────────────────────────────
        For each layer i:
            u_normed    = pre_norms[i](u)          # normalised input
            u_ns, dke   = layers[i](u_normed)      # NS step on normed u
            Δu          = u_ns - u_normed          # what the layer added
            u           = u + Δu                   # residual on original u

        Equivalent to: u ← u + Δt·F(norm(u))
        Keeps unnormalised u bounded across all layers.
        """
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)

        u = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))  # [B, L, D]

        n_steps   = 0
        delta_kes: List[float] = []

        for i, layer in enumerate(self.layers):
            # Pre-norm residual (v2 key change)
            u_normed       = self.pre_norms[i](u)
            u_ns, delta_ke = layer(u_normed)
            u              = u + (u_ns - u_normed)   # residual on original u

            n_steps += 1
            delta_kes.append(float(delta_ke.detach()))

            if adaptive and i >= self.min_steps - 1:
                if delta_ke.item() < self.convergence_threshold:
                    break

        logits = self.lm_head(self.norm(u))   # [B, L, vocab_size]

        if return_stats:
            return logits, {"n_steps": n_steps, "delta_kes": delta_kes}
        return logits

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        adaptive: bool = True,
    ) -> torch.Tensor:
        """Top-k autoregressive generation."""
        ids = prompt_ids
        for _ in range(max_new_tokens):
            ids_ctx = ids if ids.size(1) <= self.max_seq_len \
                      else ids[:, -self.max_seq_len:]
            logits = self(ids_ctx, adaptive=adaptive)     # [B, L, V]
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # [B, V]
            if top_k > 0:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs  = logits.softmax(dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)     # [B, 1]
            ids    = torch.cat([ids, next_t], dim=1)
        return ids
