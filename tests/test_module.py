import torch
from torch import nn

import deepquantum as dq


def test_module_to_func():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cir_qubit = dq.QubitCircuit(1)
            self.cir_qubit.h(0)
            self.cir_qubit.hamiltonian([[1, 'x0'], [1, 'y0'], [1, 'z0']], encode=True)
            self.cir_qubit.observable(0)
            self.cir_mps = dq.QubitCircuit(1, mps=True)
            self.cir_mps.h(0)
            self.cir_mps.hamiltonian([[1, 'x0'], [1, 'y0'], [1, 'z0']], encode=True)
            self.cir_fock = dq.QumodeCircuit(2, 'vac', cutoff=2, backend='fock', basis=False)
            self.cir_fock.ps(0, encode=True)
            self.cir_fock.bs([0, 1], encode=True)
            self.cir_bosonic = dq.QumodeCircuit(1, 'vac', backend='bosonic')
            self.cir_bosonic.cat(0)
            self.pattern = dq.Pattern()
            self.pattern.n([0, 1])
            self.pattern.e(0, 1)
            self.pattern.m(0, encode=True)
            self.pattern.x(1)

    model = Model()
    model.double()
    for buffer in model.buffers():
        assert buffer.dtype in (torch.double, torch.cdouble)
    if torch.cuda.is_available():
        model.to('cuda')
        for buffer in model.buffers():
            assert buffer.device.type == 'cuda'
    if torch.mps.is_available():
        model.float()
        model.to('mps')
        for buffer in model.buffers():
            assert buffer.device.type == 'mps'
