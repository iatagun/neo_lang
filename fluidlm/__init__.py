"""
fluidlm/__init__.py — Navier-Stokes tabanlı dil modeli çekirdek kütüphanesi
"""
from .fluid_ops import gradient, laplacian, divergence, solve_poisson
from .ns_layer  import FluidLayer
from .fluid_lm  import FluidLM

__all__ = [
    "gradient", "laplacian", "divergence", "solve_poisson",
    "FluidLayer",
    "FluidLM",
]
