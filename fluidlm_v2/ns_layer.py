"""
ns_layer.py  (v2) — Navier-Stokes Layer
========================================

One time step of 1-D incompressible Navier-Stokes:

    ∂u/∂t = −(u·∇)u  −  ∇p  +  ν∇²u

Changes from v1
───────────────
  integrator='euler' default
    v1 default was 'rk4': 4 RHS evaluations per layer.
    v2 default is 'euler': 1 RHS evaluation per layer — 4× cheaper.
    RK4 is still available via integrator='rk4' for inference/eval.

  Channel mixing  (use_channel_mix=True)
    After the NS update, apply a residual linear projection across the
    embedding dimension D.  Initialised to zero so early training is
    identical to v1; the mixer grows as needed.

        u_new = u_ns + W @ u_ns        W ∈ ℝ^{D×D},  W₀ = 0

    Physical analogy: inter-component pressure coupling.  Without this,
    all D embedding channels evolve independently — no cross-channel
    energy transfer.  This is the v1 "channel isolation" bug.

  Uses v2 fluid_ops (O(h²) causal gradient, bfloat16-safe Poisson).

Learnable parameters per layer (unchanged)
──────────────────────────────────────────
  log_nu       — log kinematic viscosity ν
  log_dt       — log time-step Δt
  log_alpha    — log pressure range α
  p_scale_raw  — pressure gradient scaling (direct, init=1.0)
  channel_mix  — D×D linear projection (init=zeros, optional)
"""

import math

import torch
import torch.nn as nn

from .fluid_ops import gradient, laplacian, divergence, solve_poisson


class FluidLayer(nn.Module):
    """
    One Navier-Stokes time step as a learnable layer.

    Parameters
    ----------
    d_model : int
        Embedding / velocity-field dimension D.
    nu : float
        Initial kinematic viscosity ν.
    dt : float
        Initial time-step Δt.
    alpha : float
        Initial pressure range parameter α.
    integrator : str
        'euler' (default, 4× faster) or 'rk4' (4th-order accuracy).
    causal : bool
        True = autoregressive (causal operators).
        False = bidirectional (analysis tasks).
    use_channel_mix : bool
        If True, apply a zero-initialised residual linear layer after
        the NS update to enable cross-channel energy transfer.
    """

    def __init__(
        self,
        d_model: int,
        nu: float = 0.01,
        dt: float = 0.1,
        alpha: float = 1.0,
        integrator: str = "euler",       # v2: euler default
        causal: bool = True,
        use_channel_mix: bool = True,    # v2: channel mixing
    ):
        super().__init__()
        self.d_model         = d_model
        self.integrator      = integrator
        self.causal          = causal

        # Physics parameters (unchanged from v1)
        self.log_nu      = nn.Parameter(torch.tensor(math.log(nu)))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(dt)))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(alpha)))
        self.p_scale_raw = nn.Parameter(torch.tensor(1.0))

        # Channel mixing: zero-initialised residual projection
        # Starts as identity (residual + 0); grows only when it helps.
        if use_channel_mix:
            self.channel_mix = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.channel_mix.weight)
        else:
            self.channel_mix = None

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def nu(self) -> torch.Tensor:
        return self.log_nu.exp()

    @property
    def dt(self) -> torch.Tensor:
        return self.log_dt.exp()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def p_scale(self) -> torch.Tensor:
        return self.p_scale_raw

    # ── NS right-hand side  F(u) = −(u·∇)u − ∇p + ν∇²u ─────────────────────

    def _rhs(self, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate dU/dt at current state u.  [B, L, D] → [B, L, D]

        Advection  −(u·∇)u
        ───────────────────
        Speed is bounded: tanh(‖u‖₂) ∈ (0,1) prevents norm explosion.

        Pressure  −∇p
        ─────────────
        Causal mode: cumulative divergence proxy.
          cumsum normalised *before* alpha division so alpha is not
          cancelled by downstream std normalisation (v1 critical fix).

        Viscosity  ν∇²u
        ────────────────
        Local smoothing across neighbouring tokens.
        """
        # 1. Advection (uses O(h²) causal gradient from v2 fluid_ops)
        speed = torch.tanh(u.norm(dim=-1, keepdim=True))         # [B, L, 1]
        adv   = speed * gradient(u, causal=self.causal)          # [B, L, D]

        # 2. Pressure
        if self.causal:
            div_adv = divergence(adv, causal=True)               # [B, L]
            cumsum  = torch.cumsum(-div_adv, dim=1)
            cumsum  = cumsum / (cumsum.std(dim=1, keepdim=True).detach() + 1e-6)
            p       = cumsum / (self.alpha + 1e-6)
        else:
            rhs_p = -divergence(adv, causal=False)
            p = solve_poisson(rhs_p, alpha=self.alpha)
            p = p / (p.std(dim=1, keepdim=True).detach() + 1e-6)

        p_grad = gradient(p.unsqueeze(-1), causal=self.causal)   # [B, L, 1]
        p_grad = self.p_scale * p_grad.expand_as(u)              # [B, L, D]

        # 3. Viscosity
        visc = self.nu * laplacian(u, causal=self.causal)        # [B, L, D]

        return -adv - p_grad + visc

    # ── Integrators ──────────────────────────────────────────────────────────

    def _euler(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Forward Euler: u + Δt·F(u).  1 RHS evaluation."""
        return u + dt * self._rhs(u)

    def _rk4(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Classical RK4.  4 RHS evaluations — use for eval/inference only."""
        k1 = self._rhs(u)
        k2 = self._rhs(u + dt * 0.5 * k1)
        k3 = self._rhs(u + dt * 0.5 * k2)
        k4 = self._rhs(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, u: torch.Tensor):
        """
        Advance the velocity field by one time step.

        Parameters
        ----------
        u : torch.Tensor  [B, L, D]
            Velocity field (token embeddings).

        Returns
        -------
        u_new : torch.Tensor  [B, L, D]
        delta_ke : torch.Tensor  scalar
            Normalised kinetic-energy change  ⟨(u_new − u)²⟩ / (⟨u²⟩ + ε).
        """
        dt = self.dt

        if self.integrator == "euler":
            u_new = self._euler(u, dt)
        else:
            u_new = self._rk4(u, dt)

        # Channel mixing: zero-init residual projection for cross-D coupling
        if self.channel_mix is not None:
            u_new = u_new + self.channel_mix(u_new)

        delta_ke = (u_new - u).pow(2).mean() / (u.pow(2).mean() + 1e-6)
        return u_new, delta_ke
