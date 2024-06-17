"""
Utilities
"""

import deepquantum.photonic as dqp


def set_hbar(hbar: float) -> None:
    """Set the global reduced Planck constant."""
    dqp.hbar = hbar

def set_kappa(kappa: float) -> None:
    """Set the global kappa."""
    dqp.kappa = kappa
