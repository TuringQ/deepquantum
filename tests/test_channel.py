import deepquantum as dq
import pytest
import torch


def test_qubit_channel():
    cir = dq.QubitCircuit(2, den_mat=True)
    cir.hlayer()
    cir.bit_flip(0)
    cir.phase_flip(1)
    cir.depolarizing(0)
    cir.pauli(1)
    cir.amp_damp(0)
    cir.phase_damp(1)
    cir.gen_amp_damp(0)
    assert torch.allclose(torch.trace(cir()), torch.tensor(1.) + 0j)
