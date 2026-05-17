"""
ns_layer.py — Navier-Stokes Layer for FluidLM
==============================================

One time step of the 1-D incompressible Navier-Stokes equation:

    ∂u/∂t = −(u·∇)u   −  ∇p   +  ν ∇²u
              ↓            ↓        ↓
          advection    pressure   viscosity
         (meaning      (attention  (smoothing /
          transport)    analog)     regularisation)

Pressure is determined by the incompressibility constraint ∇·u = 0.
Taking the divergence of the momentum equation gives the Poisson problem:

    ∇²p = −∇·( (u·∇)u )

which is solved spectrally (exact, O(L log L), autograd-compatible).

Two integrators are available:
  • 'euler'  — first-order forward Euler  (fast, 1 RHS evaluation)
  • 'rk4'    — classical 4th-order Runge-Kutta  (accurate, 4 evaluations)

Learnable parameters per layer
-------------------------------
  log_nu   — log of kinematic viscosity ν  (initialised from config)
  log_dt   — log of time-step size    Δt  (initialised from config)

Both are constrained positive via exp(·), so they never go negative.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fluid_ops import gradient, laplacian, divergence, solve_poisson


class FluidLayer(nn.Module):
    """
    One Navier-Stokes time step as a learnable neural-network layer.

    Parameters
    ----------
    d_model : int
        Embedding / velocity-field dimension D.
    nu : float
        Initial kinematic viscosity ν.  Controls how fast differences
        between neighbouring token embeddings are smoothed out.
        Analogy: dropout strength — high ν → aggressive smoothing.
    dt : float
        Initial time-step Δt.  Controls how far the field evolves per
        layer.  Analogy: learning rate / residual gate.
    integrator : str
        'euler' or 'rk4'.  RK4 is recommended for production; Euler is
        useful for fast prototyping.
    """

    def __init__(
        self,
        d_model: int,
        nu: float = 0.01,
        dt: float = 0.1,
        alpha: float = 1.0,
        integrator: str = "rk4",
        causal: bool = True,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model    = d_model
        self.integrator = integrator
        self.causal     = causal

        self.log_nu      = nn.Parameter(torch.tensor(math.log(nu)))
        self.log_dt      = nn.Parameter(torch.tensor(math.log(dt)))
        self.log_alpha   = nn.Parameter(torch.tensor(math.log(alpha)))
        self.log_p_scale = nn.Parameter(torch.tensor(0.0))  # log(1.0) → p_scale=1.0 at init

        # ── Pre-norm + MLP sublayer ─────────────────────────────────
        # norm1: NS sublayer için (routing)
        # norm2: MLP sublayer için (kapasite / knowledge storage)
        # Content-dependent advection speed (v4 fix)
        # speed_i = tanh(q_i · k_{i-1} / sqrt(d_k))  [causal, O(L)]
        # Replaces content-independent speed = tanh(||u||) which caused
        # token routing to be independent of neighbouring token content.
        self._d_k = max(d_model // 8, 16)
        self.W_q = nn.Linear(d_model, self._d_k, bias=False)
        self.W_k = nn.Linear(d_model, self._d_k, bias=False)
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_k.weight, std=0.02)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    # ── convenience properties ──────────────────────────────────────────────

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
        """Learned scaling factor for pressure gradient contribution (always positive)."""
        return self.log_p_scale.exp()

    # ── Compressible NS right-hand side  F(u) = −(u·∇)u − ∇p + ν∇²u ────────

    def _rhs(self, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the NS right-hand side at state u.

        u : [B, L, D]  (B=batch, L=token positions = spatial axis, D=embedding)
        returns du/dt : [B, L, D]

        Coordinate convention
        ─────────────────────
        x = token position (sequence index).  Differential operators ∂/∂x
        act along the L dimension.  D-dimensional embeddings are treated as
        independent field components that all share the same spatial axis.
        This is a 1-D, D-component vector field on a periodic domain.

        Compressibility
        ───────────────
        We do NOT enforce ∇·u = 0.  The flow is compressible: divergence
        is permitted, and no divergence-free projection is applied to u.
        The pressure p is used purely as a global coupling mechanism —
        it is derived from the local divergence of the advection field and
        couples distant tokens through the spectral Poisson solve.

        Backpropagation
        ───────────────
        With integrator='euler' this layer is u_new = u + dt·F(u), a plain
        residual connection.  Autograd differentiates through it exactly in
        O(n_layers · B · L · D) memory — no adjoint method is needed.
        With integrator='rk4' the same holds with 4× the intermediate
        tensors.  Adjoint (torchdiffeq) is an optional memory optimisation.

        Advection  −(u·∇)u
        ───────────────────
        Transport speed = tanh(‖u‖₂) ∈ (0,1):  bounds the Reynolds number,
        prevents norm explosion, keeps dynamics in the laminar regime.

            speed  = tanh(‖u[b,i,:]‖₂)         [B, L, 1]
            adv    = speed · ∂u/∂x              [B, L, D]

        Pressure  −∇p  (global long-range coupling)
        ────────────────────────────────────────────
        p is found by solving the Helmholtz–Poisson equation:

            (∇² − α²) p = −mean_d(∂adv_d/∂x)

        The RHS uses the mean over D (scale-invariant divergence proxy).
        ∇p is then broadcast across all D components, providing isotropic
        long-range interaction analogous to attention.

        Viscosity  ν∇²u  (local smoothing)
        ────────────────────────────────────
        Diffuses sharp embedding differences between neighbouring tokens.
        """
        # 1. Advection ─────────────────────────────────────────────────────
        # Content-dependent speed: token i's transport rate depends on
        # its query matching the previous token's key (causal, O(L)).
        q = self.W_q(u)                                               # [B, L, d_k]
        k = self.W_k(u)                                               # [B, L, d_k]
        k_prev = torch.cat([torch.zeros_like(k[:, :1]), k[:, :-1]], dim=1)  # causal shift
        speed = torch.tanh((q * k_prev).sum(-1, keepdim=True) / (self._d_k ** 0.5))  # [B, L, 1]
        adv   = speed * gradient(u, causal=self.causal)          # [B, L, D]

        # 2. Pressure (long-range coupling) ────────────────────────────────
        if self.causal:
            # Causal pressure: cumulative sum of divergence (left-to-right only).
            # Normalise the cumsum BEFORE dividing by alpha so that alpha is NOT
            # cancelled by std(cumsum/alpha) = std(cumsum)/alpha.
            div_adv = divergence(adv, causal=True)               # [B, L]
            cumsum  = torch.cumsum(-div_adv, dim=1)              # [B, L]
            cumsum  = cumsum / (cumsum.std(dim=1, keepdim=True).detach() + 1e-6)  # normalise first
            p = cumsum / (self.alpha + 1e-6)                     # alpha still in gradient path
        else:
            # Non-causal: spectral FFT Poisson solve (periodic BCs).
            rhs_p = -divergence(adv, causal=False)
            p = solve_poisson(rhs_p, alpha=self.alpha)
            p = p / (p.std(dim=1, keepdim=True).detach() + 1e-6)  # normalise
        p_grad = gradient(p.unsqueeze(-1), causal=self.causal)   # [B, L, 1]
        p_grad = self.p_scale * p_grad.expand_as(u)              # [B, L, D]

        # 3. Viscosity ──────────────────────────────────────────────────────
        visc = self.nu * laplacian(u, causal=self.causal)        # [B, L, D]

        # 4. Combine  ───────────────────────────────────────────────────────
        return -adv - p_grad + visc

    # ── Integrators ─────────────────────────────────────────────────────────

    def _euler(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return u + dt * self._rhs(u)

    def _rk4(self, u: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Classical 4th-order Runge-Kutta integration.

        k1 = F(u)
        k2 = F(u + Δt/2 · k1)
        k3 = F(u + Δt/2 · k2)
        k4 = F(u + Δt   · k3)
        u_new = u + Δt/6 · (k1 + 2k2 + 2k3 + k4)

        RK4 achieves 4th-order accuracy with only 4 function evaluations.
        For the NS equation it handles the nonlinear advection term much
        better than Euler, especially at larger Δt.
        """
        k1 = self._rhs(u)
        k2 = self._rhs(u + dt * 0.5 * k1)
        k3 = self._rhs(u + dt * 0.5 * k2)
        k4 = self._rhs(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ── Forward pass ────────────────────────────────────────────────────────

    def forward(self, u: torch.Tensor):
        """
        Advance the velocity field by one time step.

        Parameters
        ----------
        u : torch.Tensor, shape [B, L, D]
            Current velocity field (token embeddings).

        Returns
        -------
        u_new : torch.Tensor, shape [B, L, D]
            Updated velocity field after one NS time step.
        delta_ke : torch.Tensor, scalar
            Mean kinetic-energy change  ⟨(u_new − u)²⟩.
            Used as the adaptive-depth convergence criterion:
            when delta_ke < threshold the flow has stabilised and
            we can skip remaining layers.
        """
        dt = self.dt

        # ── Sublayer 1: NS routing (pre-norm + residual) ────────────────
        # u1 is stored once; used for integrator input, residual, and delta_ke.
        # Calling self.norm1(u) multiple times would be redundant and — after
        # the residual update — would evaluate on the WRONG (updated) tensor.
        u1 = self.norm1(u)                        # [B, L, D]  pre-normed, cached
        if self.integrator == "euler":
            u_ns = self._euler(u1, dt)
        else:
            u_ns = self._rk4(u1, dt)
        # Residual: add only the NS delta (u_ns = u1 + dt·F(u1)  →  delta = dt·F(u1))
        u = u + (u_ns - u1)

        # ── ΔKE — measure NS routing change BEFORE MLP (convergence criterion) ──
        # Ratio of NS force magnitude to current state magnitude.
        # Small delta_ke → physics has converged → adaptive mode can stop.
        delta_ke = (u_ns - u1).pow(2).mean() / (u1.pow(2).mean() + 1e-6)

        # ── Sublayer 2: MLP (pre-norm + residual) ─────────────────────
        u = u + self.mlp(self.norm2(u))

        return u, delta_ke
