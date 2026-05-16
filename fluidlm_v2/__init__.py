"""
fluidlm_v2 — Optimised Navier-Stokes Language Model
=====================================================

Improvements over fluidlm/:
  1. Pre-norm residual in FluidLM       → training stability
  2. Channel mixing per NS layer        → representation capacity, faster convergence
  3. Euler default (RK4 opt-in)         → 4× fewer RHS evaluations per layer
  4. O(h²) causal gradient              → better numerical accuracy
  5. bfloat16-safe spectral Poisson     → mixed precision training support
  6. torch.compile-ready (no Python branching in hot path)

Original architecture (fluidlm/) is untouched.
"""

from .fluid_ops import gradient, laplacian, divergence, solve_poisson
from .ns_layer  import FluidLayer
from .fluid_lm  import FluidLM

__all__ = ["FluidLayer", "FluidLM", "gradient", "laplacian", "divergence", "solve_poisson"]
