"""Bargmann (stellar) representation of photonic states, gates and channels.

The legacy heralding helpers (``conditional_fock`` and friends) remain importable from here.
"""

from .ansatz import PolyExpAnsatz, fock_to_ansatz, vacuum_ansatz
from .heralded import _single_mode_density, conditional_fock
from .integrals import gaussian_contract
from .moments import gaussian_to_bargmann
from .recurrence import _MAX_WORK_BYTES, _check_work, _device_layers, _fock_coefficients, _recurrence_plan
from .state import BargmannState
from .triples import (
    displacement_kernel,
    loss_kernel,
    op_to_kernel,
    passive_kernel,
    squeezing_kernel,
    two_mode_squeezing_kernel,
)
