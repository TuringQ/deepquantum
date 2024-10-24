"""
Time domain multiplexing
"""

from typing import Any, List, Optional, Union

import torch

from .circuit import QumodeCircuit


class QumodeCircuitTDM(QumodeCircuit):
    r"""Time-domain-multiplexed photonic quantum circuit.

    Note: When using large squeezing parameters, we recommend using a double data type and
        a smaller ``eps`` for Homodyne to avoid issues with non-positive definiteness of the covariance matrix.

    Args:
        nmode (int): The number of spatial modes in the circuit.
        init_state (Any): The initial state of the circuit. It can be a vacuum state with ``'vac'``.
            For Gaussian backend, it can be arbitrary Gaussian states with ``[cov, mean]``.
            Use ``xxpp`` convention and :math:`\hbar=2` by default.
        cutoff (int or None, optional): The Fock space truncation. Default: ``None``
        name (str or None, optional): The name of the circuit. Default: ``None``
        noise (bool, optional): Whether to introduce Gaussian noise. Default: ``False``
        mu (float, optional): The mean of Gaussian noise. Default: 0
        sigma (float, optional): The standard deviation of Gaussian noise. Default: 0.1
    """
    def __init__(
        self,
        nmode: int,
        init_state: Any,
        cutoff: Optional[int] = None,
        name: Optional[str] = None,
        noise: bool = False,
        mu: float = 0,
        sigma: float = 0.1
    ) -> None:
        super().__init__(nmode=nmode, init_state=init_state, cutoff=cutoff, backend='gaussian', basis=False,
                         detector='pnrd', name=name, mps=False, chi=None, noise=noise, mu=mu, sigma=sigma)
        self.samples = None
        self._cir_global = None

    def forward(
        self,
        data: Optional[torch.Tensor] = None,
        state: Any = None,
        nstep: Optional[int] = None
    ) -> List[torch.Tensor]:
        r"""Perform a forward pass of the TDM photonic quantum circuit and return the final state.

        Args:
            data (torch.Tensor or None, optional): The input data for the ``encoders`` with the shape of
                :math:`(\text{batch}, \text{ntimes}, \text{nfeat})`. Default: ``None``
            state (Any, optional): The initial state for the photonic quantum circuit. Default: ``None``
            nstep (int or None, optional): The number of the evolved time steps. Default: ``None``

        Returns:
            List[torch.Tensor]: The covariance matrix and displacement vector of the measured final state.
        """
        assert self._if_delayloop, 'No delay loop.'
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
        self.state = state
        samples = []
        for i in range(nstep):
            if data is None:
                self.state = super().forward(state=self.state)
            else:
                data_i = data[:, i % size[1], :]
                self.state = super().forward(data_i, self.state)
            samples.append(self.measure_homodyne(shots=1))
            self.state = self.state_measured
        self.samples = torch.stack(samples, dim=-1) # (batch, nwire, nstep)
        return self.state

    def get_samples(self, wires: Union[int, List[int], None] = None) -> torch.Tensor:
        """Get the measured samples according to the given ``wires``."""
        if wires is None:
            wires = self.wires
        wires = sorted(self._convert_indices(wires))
        return self.samples[..., wires, :]
