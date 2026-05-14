"""
fluid_ops.py — Core fluid dynamics operators for FluidLM
=========================================================

Discretization: 1-D sequence of length L with periodic boundary conditions.
All operators work on tensors of shape [B, L, D] where:
  B = batch size
  L = sequence length  (the "spatial" domain — token positions)
  D = embedding dim    (the "field" dimension — each token is a D-dim vector)

Physical analogy
----------------
  Token position i  ↔  spatial grid point xᵢ
  Embedding u[b,i]  ↔  velocity vector  u(xᵢ, t)
  Sequence length L ↔  domain size
  Model depth       ↔  time axis t

Operators implemented
---------------------
  gradient(u)            ∂u/∂x        central difference, O(h²)
  laplacian(u)           ∂²u/∂x²      central difference, O(h²)
  divergence(u)          Σ_d ∂u_d/∂x  scalar field [B,L]
  solve_poisson(rhs)     ∇²p = rhs    exact spectral solve via FFT
"""

import math
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Finite-difference operators  (periodic BCs via torch.roll)
# ─────────────────────────────────────────────────────────────────────────────

def gradient(u: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    Gradient ∂u/∂x along the sequence (L) dimension.

    causal=False (default): centered difference O(h²), periodic BCs.
        ∂u/∂x [i] ≈ (u[i+1] − u[i−1]) / 2
        Used for analysis / non-autoregressive tasks.

    causal=True: backward difference O(h), zero-padded at i=0.
        ∂u/∂x [i] ≈ u[i] − u[i−1]      (u[-1] = 0 by padding)
        REQUIRED for autoregressive language modelling — prevents
        position i from seeing future tokens i+1, i+2, …

    u : [B, L, D]  →  ∂u/∂x : [B, L, D]
    """
    if causal:
        # F.pad: (last_dim_left, last_dim_right, second_last_left, …)
        # We pad along dim=1 (L): one zero frame on the left, none on right.
        import torch.nn.functional as _F
        padded = _F.pad(u, (0, 0, 1, 0))   # [B, L+1, D]
        return u - padded[:, :-1, :]        # u[i] - u[i-1]
    return (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) * 0.5


def laplacian(u: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    Second-order Laplacian ∂²u/∂x² along L.

    causal=False: central difference, periodic BCs.
        ∂²u/∂x² [i] ≈ u[i+1] − 2u[i] + u[i−1]

    causal=True: backward second difference, zero-padded.
        ∂²u/∂x² [i] ≈ u[i] − 2u[i−1] + u[i−2]

    u : [B, L, D]  →  ∇²u : [B, L, D]
    """
    if causal:
        import torch.nn.functional as _F
        padded = _F.pad(u, (0, 0, 2, 0))   # [B, L+2, D]
        return u - 2 * padded[:, 1:-1, :] + padded[:, :-2, :]
    return torch.roll(u, -1, dims=1) - 2.0 * u + torch.roll(u, 1, dims=1)


def divergence(u: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    Mean divergence field along the sequence axis.

        div(u)[b, i] = mean_d( ∂u_d/∂x [b, i, d] )

    u : [B, L, D]  →  div(u) : [B, L]
    """
    return gradient(u, causal=causal).mean(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Poisson solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_poisson(rhs: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Solve the regularised 1-D Helmholtz–Poisson equation spectrally (FFT):

        (∇² − α²) p = rhs

    Setting α = 0 recovers the standard Poisson equation.  α > 0 adds a
    "screening" term that prevents low-frequency modes from being amplified
    unboundedly (analogous to a Debye length in plasma physics).

    In Fourier space the eigenvalues become:

        μ_k = 2(cos(2πk/L) − 1) − α²

    Since μ_k ≤ −α² < 0 for all k, the denominator is always non-zero and
    bounded away from zero → no gauge ambiguity, no pressure blowup.

    Physical interpretation of α
    ─────────────────────────────
    • α = 0   : fully global attention (every token influences every other)
    • α > 0   : exponentially screened, range ∝ 1/α
    • Large α  : only nearby tokens interact through pressure

    In FluidLM, α is a learnable per-layer parameter, letting the model
    discover the right interaction range at each depth.

    rhs   : [B, L]   right-hand side
    alpha : float    screening coefficient ≥ 0
    returns p : [B, L]
    """
    B, L = rhs.shape
    device = rhs.device

    rhs_fft = torch.fft.rfft(rhs, dim=1)                          # [B, L//2+1]

    k = torch.arange(L // 2 + 1, dtype=torch.float32, device=device)
    lambda_k = 2.0 * (torch.cos(2.0 * math.pi * k / L) - 1.0) - alpha ** 2
    # lambda_k is always ≤ −α² ≤ 0, so always non-zero when α > 0
    # Guard against exact zero (α=0 case, k=0):
    lambda_k = lambda_k.masked_fill(lambda_k.abs() < 1e-8, -1e-8)

    p_fft = rhs_fft / lambda_k.unsqueeze(0)                       # [B, L//2+1]

    p = torch.fft.irfft(p_fft, n=L, dim=1)                       # [B, L]
    return p
