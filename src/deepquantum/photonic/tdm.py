"""Time domain multiplexing"""

from collections.abc import Sequence
from typing import Any

import torch

from .circuit import QumodeCircuit
from .state import GaussianState


class QumodeCircuitTDM(QumodeCircuit):
    r"""Time-domain-multiplexed photonic quantum circuit.

    Note:
        Use double precision for large squeezing to reduce cancellation in conditional covariances.
        Homodyne uses the exact single-quadrature update without a finite-squeezing approximation.

    Args:
        nmode: The number of spatial modes in the circuit.
        init_state: The initial state of the circuit. It can be a vacuum state with ``'vac'``.
            For Gaussian backend, it can be arbitrary Gaussian states with ``[cov, mean]``.
            Use ``xxpp`` convention and :math:`\hbar=2` by default.
        cutoff: The Fock space truncation. Default: ``None``
        backend: Use ``'gaussian'`` for Gaussian backend or ``'bosonic'`` for Bosonic backend. Default: ``'gaussian'``
        name: The name of the circuit. Default: ``None``
    """

    def __init__(
        self, nmode: int, init_state: Any, cutoff: int | None = None, backend: str = 'gaussian', name: str | None = None
    ) -> None:
        assert backend in ('gaussian', 'bosonic')
        super().__init__(
            nmode=nmode,
            init_state=init_state,
            cutoff=cutoff,
            backend=backend,
            basis=False,
            detector='pnrd',
            name=name,
            mps=False,
            chi=None,
        )
        self.samples = None

    def forward(
        self,
        data: torch.Tensor | None = None,
        state: Any = None,
        nstep: int | None = None,
        *,
        memory: GaussianState | Sequence[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        r"""Perform a forward pass of the TDM photonic quantum circuit and return the final state.

        Args:
            data: The input data for the ``encoders`` with the shape of
                :math:`(\text{batch}, \text{ntimes}, \text{nfeat})`. Default: ``None``
            state: Initial spatial or full concurrent state. Mutually exclusive with
                ``memory`` and registered sources. Default: ``None``
            nstep: The number of the evolved time steps. Default: ``None``
            memory: Gaussian loop modes in next-step order. With explicit memory, prepare fresh
                spatial inputs from ``init_state`` on every step of this call. Otherwise an initial
                spatial state enters on the first step only. Registered sources prepare each step.
                Default: ``None``

        Returns:
            The full Homodyne-conditioned state after the last step. For Gaussian delay circuits
            ``tdm_memory(state)`` extracts its loop modes for continuation.
        """
        assert self._with_delay, 'No delay loop.'
        for i in range(self.nmode):
            assert i in self.wires_homodyne
        if data is None:
            if nstep is None:
                nstep = 1
        else:
            size = data.size()
            assert data.ndim == 3
            if nstep is None:
                nstep = size[1]
        if memory is not None and (self.backend != 'gaussian' or self._uses_submodes):
            raise NotImplementedError('TDM Homodyne memory requires a Gaussian circuit without submodes.')
        fresh_each_step = memory is not None or bool(self.sources)
        samples = []
        for i in range(nstep):
            data_i = None if data is None else data[:, i % size[1], :]
            super().forward(data_i, state=state, memory=memory)
            samples.append(self.measure_homodyne(shots=1))
            self.state = self.state_measured
            if i + 1 < nstep:
                state = None if fresh_each_step else self.state
                memory = self.tdm_memory(self.state) if fresh_each_step else None
        self.samples = torch.stack(samples, dim=-1)  # (batch, nwire, nstep)
        return self.state

    def get_samples(self, wires: int | list[int] | None = None) -> torch.Tensor:
        """Get the measured samples according to the given ``wires``."""
        if wires is None:
            wires = self.wires
        wires = sorted(self._convert_indices(wires))
        return self.samples[..., wires, :]
