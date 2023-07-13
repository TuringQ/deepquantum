import torch
import numpy as np
import sys
sys.path.append("..")
from qumode import QumodeCircuit, GaussianState, homodyne_measure, heterodyne_measure, \
    prob_gbs, mean_photon_number, diff_photon_number
from gaussian.ops import Displacement
import qumode




####################
####################


# n_modes = 4
# batch_size = 1

# # initialize a gaussian state
# gs = GaussianState(batch_size, n_modes)
# print('initial gaussian sate:', gs.displacement(), gs.covariance())

# # 
# r = torch.randn(n_modes, requires_grad=True)
# theta = 2*torch.pi*torch.randn(n_modes, requires_grad=True)

# # circuit
# cir = QumodeCircuit(n_modes=n_modes, backend='gaussian')
# cir.displace(r=torch.tensor(1.0), phi=torch.tensor(0.0), mode=0)
# cir.displace(r=torch.tensor([2.0]), phi=torch.tensor([0.0]), mode=1)
# cir.squeeze(r=r[0], phi=theta[0], mode=0)
# cir.squeeze(r=r[1], phi=theta[1], mode=1)
# cir.squeeze(r=r[2], phi=theta[2], mode=2)
# cir.squeeze(r=r[3], phi=theta[3], mode=3)
# cir.phase_shift(phi=torch.tensor(torch.pi/2), mode=0)
# cir.beam_split(r=torch.tensor(0.2), phi=torch.tensor(1.1), mode=[0,1])
# #cir.random_unitary(seed=134)

# # new gaussian state
# new_gs = cir(gs)
# print('new gaussian sate:', new_gs.displacement(), new_gs.covariance())
# #print('squeezing parameters: ', cir.squeezing_paras)
# #print('random unitary matrix: ', cir.random_u)


# # prob = prob_gbs(new_gs, cir, torch.tensor([0,0,1,0]))
# # print(f'The probability of photon pattern {[0,0,1,0]} is {prob}.')

# # prob = prob_gbs(new_gs, cir, torch.tensor([0,1,1,0]))
# # print(f'The probability of photon pattern {[0,1,1,0]} is {prob}.')


# # measurement
# res = homodyne_measure(new_gs, 1)
# print(f'The result of homodyne measuring the mode {1} is {res}.')
# res = heterodyne_measure(new_gs, 1)
# print(f'The result of heterodyne measuring the mode {1} is {res}.')


# mean = mean_photon_number(new_gs, 0)
# print(f'The mean photon number of mode {0} is {mean}.')
# mean = mean_photon_number(new_gs, 1)
# print(f'The mean photon number of mode {1} is {mean}.')

# diff = diff_photon_number(new_gs, 0, 1)
# print(f'The difference of two mode photon number is {diff}.')



######################
######################

x = torch.tensor([[1.7, 2.1, -4.0], 
                [-1.1, 1.3, 2.0],
                [-1.1, 1.3, 2.0]], requires_grad=True)
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
#print('initial gaussian sate:', state.displacement(), state.covariance())

cir = encoding_cir + var_cir + encoding_cir + var_cir # data re-uploading

state = cir(state)
#print('final gaussian sate:', state.displacement(), state.covariance())

diff = diff_photon_number(state, 0, 1)
#print(f'The difference of two mode photon number is {diff}.')



