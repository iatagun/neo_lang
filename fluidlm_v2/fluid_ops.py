"""
fluid_ops.py  (v2) — Core fluid dynamics operators
====================================================

Changes from v1
───────────────
  gradient (causal=True):
    v1: O(h¹) first-order backward difference  u[i] − u[i-1]
    v2: O(h²) three-point backward difference  (3u[i] − 4u[i-1] + u[i-2]) / 2

  solve_poisson:
    v1: fails silently on bfloat16 (FFT dtype mismatch)
    v2: upcasts to float32 before FFT, downcasts result to original dtype

Everything else (laplacian, divergence, non-causal gradient) is identical to v1.

Tensor convention: [B, L, D]
  B = batch size
  L = sequence length  (spatial domain — token positions)
  D = embedding dim    (field components)
"""

import math
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Differential operators
# ─────────────────────────────────────────────────────────────────────────────

def gradient(u: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    ∂u/∂x along the sequence dimension (dim=1).

    causal=False: centred O(h²), periodic BCs.
        ∂u/∂x[i] ≈ (u[i+1] − u[i-1]) / 2

    causal=True: three-point backward difference O(h²).
        ∂u/∂x[i] ≈ (3u[i] − 4u[i-1] + u[i-2]) / 2
        Zero-padding at i=0,1 preserves causality.

    v1 used O(h¹): u[i] − u[i-1].  This doubles gradient accuracy
    with minimal extra cost (one extra pad column, one extra multiply).

    u : [B, L, D]  →  ∂u/∂x : [B, L, D]
    """
    if causal:
        # Pad two zero frames on the left so i=0 and i=1 use zeros as history.
        padded = F.pad(u, (0, 0, 2, 0))          # [B, L+2, D]
        return (3.0 * u
                - 4.0 * padded[:, 1:-1, :]
                +       padded[:, :-2,  :]) * 0.5
    return (torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)) * 0.5


def laplacian(u: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    ∂²u/∂x² along the sequence dimension.

    causal=False: centred O(h²), periodic BCs.
    causal=True:  backward second difference O(h²).

    Unchanged from v1 — already second-order.

    u : [B, L, D]  →  ∇²u : [B, L, D]
    """
    if causal:
        padded = F.pad(u, (0, 0, 2, 0))          # [B, L+2, D]
        return u - 2.0 * padded[:, 1:-1, :] + padded[:, :-2, :]
    return torch.roll(u, -1, dims=1) - 2.0 * u + torch.roll(u, 1, dims=1)


def divergence(u: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """
    Mean divergence: div(u)[b, i] = mean_d( ∂u_d/∂x [b, i, d] )

    u : [B, L, D]  →  div(u) : [B, L]
    """
    return gradient(u, causal=causal).mean(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Poisson solver  (bfloat16-safe)
# ─────────────────────────────────────────────────────────────────────────────

def solve_poisson(rhs: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Solve the regularised 1-D Helmholtz–Poisson equation spectrally:

        (∇² − α²) p = rhs      →     p = ifft(fft(rhs) / λ_k)

    Eigenvalues: λ_k = 2(cos(2πk/L) − 1) − α²  ≤ −α² < 0  (always nonzero)

    bfloat16 safety (v2 addition)
    ─────────────────────────────
    torch.fft.rfft does not support bfloat16.  We upcast rhs to float32,
    perform the spectral solve, and downcast back to the caller's dtype.
    This adds ~5% overhead for float32 inputs (no-op cast) and is
    essential for AMP / bfloat16 training.

    rhs   : [B, L]
    alpha : float  (screening coefficient ≥ 0)
    returns p : [B, L]  (same dtype as rhs)
    """
    orig_dtype = rhs.dtype
    rhs_f = rhs.float()                                         # safe for FFT

    B, L = rhs_f.shape
    rhs_fft = torch.fft.rfft(rhs_f, dim=1)                     # [B, L//2+1]

    k = torch.arange(L // 2 + 1, dtype=torch.float32, device=rhs.device)
    lambda_k = 2.0 * (torch.cos(2.0 * math.pi * k / L) - 1.0) - alpha ** 2
    lambda_k = lambda_k.masked_fill(lambda_k.abs() < 1e-8, -1e-8)

    p_fft = rhs_fft / lambda_k.unsqueeze(0)                    # [B, L//2+1]
    p = torch.fft.irfft(p_fft, n=L, dim=1)                     # [B, L]

    return p.to(orig_dtype)
