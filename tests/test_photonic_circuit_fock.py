import numpy as np
import tensorflow as tf
import torch

import deepquantum.photonic as dqp
import strawberryfields.backends.tfbackend as sf




def test_circuit_displace(pure=True, scalar=True, mode=0):
    if scalar:
        r=torch.rand(())
        phi=torch.randn(())
        r_sf = r.numpy()
        phi_sf = phi.numpy()
      
    else:
        r=torch.rand(3)                # (batch_size, )
        phi=torch.randn(3)
        r_sf = r.numpy()
        phi_sf = phi.numpy()

    cir_sf = sf.circuit.Circuit(num_modes=2, cutoff_dim=3, pure=pure, batch_size=3, dtype=tf.complex128)
    cir_sf.displacement(r=r_sf, phi=phi_sf, mode=mode)
    #print('debug cir_sf._state', cir_sf._state)

    cir_fock = dqp.QumodeCircuit(n_modes=2, cutoff=3, pure=pure, batch_size=3, dtype=torch.complex128, backend='fock')
    cir_fock.displace(r=r, phi=phi, mode=mode)
    cir_fock()
    #print('debug cir_fock._state', cir_fock._state)

    error = np.mean(np.abs(cir_sf._state.numpy() - cir_fock.state.tensor.numpy()))

   
    return error



if __name__ == '__main__':

    for i in range(10):
        error = test_circuit_displace(pure=True, scalar=True)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')



    for i in range(10):
        error = test_circuit_displace(pure=True, scalar=False)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')

   

 
    for i in range(10):
        error = test_circuit_displace(pure=False, scalar=True)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')



    for i in range(10):
        error = test_circuit_displace(pure=False, scalar=False)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')

    print('\n\n\nTest for `dqp.QumodeCircuit.displace` passed!')