import torch
import numpy as np
import sys
sys.path.append("..")
from qumode import QumodeCircuit, GaussianState
from gaussian.ops import Displacement




####################
####################


n_modes = 3
batch_size = 2
# initialize a gaussian state
gs = GaussianState(batch_size, n_modes)
print('initial gaussian sate:', gs.displacement(), gs.covariance())

# circuit
cir = QumodeCircuit(n_modes=n_modes, backend='gaussian')
cir.displace(r=torch.tensor(1.0), phi=torch.tensor(0.0), mode=0)
cir.displace(r=torch.tensor([2.0]), phi=torch.tensor([0.0]), mode=1)
cir.squeeze(r=torch.tensor(0.2), phi=torch.tensor(1.1), mode=0)
cir.squeeze(r=torch.tensor(0.2), phi=torch.tensor(1.1), mode=1)
cir.squeeze(r=torch.tensor(0.2), phi=torch.tensor(1.1), mode=2)
cir.phase_shift(phi=torch.tensor(torch.pi/2), mode=0)
cir.beam_split(r=torch.tensor(0.2), phi=torch.tensor(1.1), mode=[0,1])

# new gaussian state
new_gs = cir(gs)
print('new gaussian sate:', new_gs.displacement(), new_gs.covariance())






######################
######################


x = torch.tensor([[1.7, 2.1, -4.0], 
                  [-1.1, 1.3, 2.0],
                  [-1.1, 1.3, 2.0]])
n_modes = 2
batch_size = 3



encoding_cir = QumodeCircuit(n_modes=n_modes, backend='gaussian')
encoding_cir.displace(mode=0) 
encoding_cir.displace(mode=1)

var_cir = QumodeCircuit(n_modes=n_modes, backend='gaussian')
var_cir.add(Displacement(mode=0))
var_cir.add(Displacement(mode=1))


# forward iterations
encoding_cir.operators[0].set_params(r=x[:,0], phi=torch.tensor(0.0))
encoding_cir.operators[1].set_params(r=x[:,1], phi=torch.tensor(2.0))

state = GaussianState(batch_size=batch_size, n_modes=n_modes)
#state.reset(n_modes, batch_size)
print('initial gaussian sate:', state.displacement(), state.covariance())

cir = encoding_cir + var_cir + encoding_cir + var_cir # data re-uploading

state = cir(state)
print('final gaussian sate:', state.displacement(), state.covariance())

