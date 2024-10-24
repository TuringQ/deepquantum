"""
Time domain multiplexing
"""

from typing import Any, List, Optional, Union

import torch

from .circuit import QumodeCircuit
from .draw import DrawCircuit_TDM_global

from .qmath import shift_func

from collections import defaultdict
from copy import copy, deepcopy



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

    def construct_global(self, shots):
        """Construct the global circuit for TDM given shots."""
        assert shots >=1, 'shots must be larger than 1'
        self._prepare_unroll_dict()
        self._unroll_circuit()
        nmode1 = self.nmode
        nmode2 = self._nmode_tdm
        nmode3 = nmode2 + (shots-1) * nmode1
        circ_global = QumodeCircuit(nmode=nmode3, init_state='vac', cutoff=self.cutoff, backend=self.backend, basis=self.basis)
        circ_global.operators =  deepcopy(self._operators_tdm)
        circ_global.measurements =  deepcopy(self._measurements_tdm)
        sublists_ = deepcopy(list(self._unroll_dict.values()))
        no_delay_list = [i[-1] for i in sublists_]
        for i in range(len(circ_global.operators)): # shots=1 case
            op  = circ_global.operators[i]
            circ_global.npara += op.npara
            op.nmode = nmode3
            if self._operators_tdm[i] in self._encoders_tdm:
                circ_global.encoders.append(op)
                circ_global.ndata += op.npara
        for i in range(len(circ_global.measurements)):
            mea = circ_global.measurements[i]
            circ_global.npara += mea.npara
            mea.nmode = nmode3
            if self._measurements_tdm[i] in self._encoders_tdm:
                circ_global.encoders.append(mea)
                circ_global.ndata += mea.npara

        map_dict = defaultdict(list)
        for i in range(nmode2, nmode3, nmode1):
            for k in range(len(no_delay_list)):
                map_dict[no_delay_list[k]].append(i+k)

        for i in range(shots-1): # shots>1 case
            for mea in self._measurements_tdm:
                mea_copy = deepcopy(mea)
                wires = mea_copy.wires
                mea_copy.wires = [map_dict[wires[0]][i]] # single wire
                mea_copy.nmode = nmode3
                circ_global.measurements.append(mea_copy)
                circ_global.npara += mea_copy.npara
                if mea in self._encoders_tdm:
                    circ_global.encoders.append(mea_copy)
                    circ_global.ndata += mea_copy.npara
        for i in range(shots-1):
            for op in self._operators_tdm:
                op_copy = deepcopy(op)
                wires = op.wires
                not_delay_wire = all([item in no_delay_list for item in wires]) # check if is the delay wire
                if not_delay_wire:
                    pos = [ ]
                    for k in wires:
                        pos.append(map_dict[k][i])
                    op_copy.wires = pos
                    op_copy.nmode = nmode3
                    circ_global.operators.append(op_copy)
                else:
                    if len(wires)==2:
                        idx = no_delay_list.index(wires[-1])
                        temp = sublists_[idx]
                        for k in range(len(temp)-1):
                            if wires[0] in temp[k]:
                                temp[k] = shift_func(temp[k], 1)
                                new_wires = [temp[k][0], map_dict[wires[-1]][i]]
                        op_copy.wires = new_wires
                        op_copy.nmode = nmode3
                        circ_global.operators.append(op_copy)
                    if len(wires)==1:
                        op_copy.wires = [temp[k][0]]
                        op_copy.nmode = nmode3
                        circ_global.operators.append(op_copy)
                if op in self._encoders_tdm:
                     circ_global.encoders.append(op_copy)
                     circ_global.ndata += op_copy.npara
                circ_global.npara += op_copy.npara
        circ_global._shots = shots
        return circ_global
