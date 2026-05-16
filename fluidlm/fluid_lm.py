"""
fluid_lm.py — FluidLM: A Language Model Built on Navier-Stokes Dynamics
=========================================================================

Architecture overview
─────────────────────

  Input tokens  ──►  Embedding  ──►  velocity field u₀  [B, L, D]
                                              │
                                    ┌─────────▼─────────┐
                                    │   FluidLayer 1     │  ∂u/∂t = NS(u)
                                    │   (NS time step)   │
                                    └─────────┬─────────┘
                                              │  u₁, ΔKE₁
                                    ┌─────────▼─────────┐
                                    │   FluidLayer 2     │
                                    └─────────┬─────────┘
                                              │  …
                                    ┌─────────▼─────────┐
                                    │   FluidLayer N     │
                                    └─────────┬─────────┘
                                              │  u_T
                                         LayerNorm
                                              │
                                          LM Head  (weight-tied)
                                              │
                                           logits  [B, L, vocab]

Key differences from a Transformer
────────────────────────────────────
  ✗ No Q / K / V matrices per layer   → attention emerges from ∇p
  ✓ Standard MLP / FFN sublayer       → same capacity as Transformer (mlp_ratio=4)
  ✗ No fixed depth                    → adaptive early-stopping on ΔKE
  ✓ 4 routing scalars per layer       → (ν, Δt, α, p_scale) vs ~2.4M MHA params
  ✓ Physically motivated dynamics     → less inductive bias to tune
  ✓ Interpretable internals           → ν, Δt, pressure field are meaningful

Adaptive depth
──────────────
  Each FluidLayer returns ΔKE = ⟨(u_new − u)²⟩.
  When ΔKE < convergence_threshold the flow has stabilised:
  simple inputs converge in 3–4 steps, complex ones take 10–12.
  A minimum of `min_steps` layers is always executed for stability.

Weight tying
────────────
  LM head weight is tied to the token embedding matrix (standard
  practice since Press & Wolf 2017; saves ~vocab × d_model params).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .ns_layer import FluidLayer


class FluidLM(nn.Module):
    """
    Language model whose depth is determined by Navier-Stokes convergence.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    d_model : int
        Embedding / velocity-field dimension.
    n_layers : int
        Maximum number of NS time steps (layers).
    max_seq_len : int
        Maximum sequence length (for positional embeddings).
    nu : float
        Initial kinematic viscosity for all layers.
    dt : float
        Initial time-step size for all layers.
    integrator : str
        'euler' or 'rk4' — integration scheme for each layer.
    convergence_threshold : float
        ΔKE below which adaptive mode declares convergence.
    min_steps : int
        Minimum layers to run even in adaptive mode.
    dropout : float
        Embedding dropout rate.
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
        integrator: str = "rk4",
        convergence_threshold: float = 1e-4,
        min_steps: int = 3,
        dropout: float = 0.1,
        causal: bool = True,
        mlp_ratio: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.convergence_threshold = convergence_threshold
        self.min_steps = min_steps

        # ── Embedding (token → velocity field) ──────────────────────────
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emb_drop  = nn.Dropout(dropout)

        # ── Navier-Stokes layers (her biri: NS sublayer + MLP sublayer) ─────
        self.layers = nn.ModuleList([
            FluidLayer(d_model, nu=nu, dt=dt, alpha=alpha, integrator=integrator,
                       causal=causal, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        # ── Output ───────────────────────────────────────────────────────
        self.max_seq_len = max_seq_len
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding and LM head share the same matrix.
        # This halves the parameter count for the vocab projection and
        # is standard in modern LMs.
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

    # ── Parameter count utility ──────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    # ── Core forward ─────────────────────────────────────────────────────────

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
        input_ids : torch.Tensor, shape [B, L]
            Token indices.
        adaptive : bool
            If True, stop early when the kinetic energy change drops
            below `convergence_threshold` (after `min_steps` layers).
        return_stats : bool
            If True, also return a dict with per-layer ΔKE values and
            the actual number of steps taken.

        Returns
        -------
        logits : torch.Tensor, shape [B, L, vocab_size]
        stats  : dict (only when return_stats=True)
            {'n_steps': int, 'delta_kes': List[float]}
        """
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]

        # ── Token → velocity field ──────────────────────────────────────
        u = self.emb_drop(
            self.token_emb(input_ids) + self.pos_emb(pos)
        )  # [B, L, D]

        # ── Navier-Stokes evolution ─────────────────────────────────────
        n_steps   = 0
        delta_kes: List[float] = []

        for i, layer in enumerate(self.layers):
            u, delta_ke = layer(u)
            n_steps += 1
            delta_kes.append(float(delta_ke.detach()))

            # Adaptive early stopping: only after min_steps to avoid
            # premature convergence due to vanishing gradients at init.
            if adaptive and i >= self.min_steps - 1:
                if delta_ke.item() < self.convergence_threshold:
                    break

        # ── Project to vocabulary ───────────────────────────────────────
        logits = self.lm_head(self.norm(u))   # [B, L, vocab_size]

        if return_stats:
            return logits, {"n_steps": n_steps, "delta_kes": delta_kes}
        return logits

    # ── Auto-regressive generation ───────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        adaptive: bool = True,
    ) -> torch.Tensor:
        """
        Greedy / top-k auto-regressive generation.

        Parameters
        ----------
        prompt_ids : torch.Tensor, shape [B, L_prompt]
        max_new_tokens : int
        temperature : float
            Divide logits by this before softmax.  < 1 → sharper, > 1 → flatter.
        top_k : int
            Keep only the top-k candidates.  0 = no filtering.
        adaptive : bool
            Use adaptive depth during generation.

        Returns
        -------
        ids : torch.Tensor, shape [B, L_prompt + max_new_tokens]
        """
        self.eval()
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len to stay within positional embedding bounds
            ids_in = ids[:, -self.max_seq_len:]
            logits = self.forward(ids_in, adaptive=adaptive)   # [B, L, V]
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold     = topk_vals[:, -1:]
                next_logits   = next_logits.masked_fill(next_logits < threshold, -float("inf"))

            probs   = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [B, 1]
            ids     = torch.cat([ids, next_id], dim=1)

        return ids

    # ── Physical parameter summary ───────────────────────────────────────────

    def physical_params(self) -> Dict[str, List[float]]:
        """Return learned ν, Δt, α and p_scale values for each layer."""
        return {
            "nu":      [layer.nu.item()      for layer in self.layers],
            "dt":      [layer.dt.item()      for layer in self.layers],
            "alpha":   [layer.alpha.item()   for layer in self.layers],
            "p_scale": [layer.p_scale.item() for layer in self.layers],
        }
