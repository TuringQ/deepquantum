import deepquantum as dq
import numpy as np
import pytest
import torch


def test_mapper():
    cnot = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,0,1],
                     [0,0,1,0]])
    nqubit = 2
    nmode = 6
    ugate = cnot
    aux = [0, 0]
    aux_pos = [4, 5]
    success = 1 / 3
    umap = dq.UnitaryMapper(nqubit=nqubit, nmode=nmode, ugate=ugate,
                            success=success, aux=aux, aux_pos=aux_pos)
    basis = umap.basis
    Re3 = umap.solve_eqs_real(total_trials=1, trials=10, precision=1e-5) # for real solution
    # check the result
    cnot_test = Re3[0][0][0]
    init_state = [1,0,1,0,0,0]
    test_circuit = dq.QumodeCircuit(nmode=6, init_state=init_state, basis=True)
    test_circuit.any(cnot_test, list(range(6)))
    temp_cnot = torch.zeros((4, 4), dtype=torch.float64)
    for i in range(4):
        temp_re = test_circuit(state=basis[i])
        for j in range(4):
            out_state = dq.FockState(basis[j])
            temp_cnot[i][j] = (temp_re[out_state]).real
    assert torch.allclose(temp_cnot, torch.tensor(cnot * success))
