import sys
sys.path.append("..")

import numpy as np
import torch
from torch import nn
from torch import tensor
import torch.nn.functional as F
from circuit import QumodeCircuit, FockState
from fock.ops import Displacement, BeamSplitter
from sklearn import datasets
import matplotlib.pyplot as plt

# #------QC Example 1
# state = FockState(batch_size=3, n_modes=2, cutoff=15, pure=False, dtype=torch.complex128) 

# cir = QumodeCircuit(n_modes=2, backend='fock')
# cir.displace(r=tensor(1.0), phi=tensor(2.0), mode=0)
# cir.displace(r=tensor(2.0), phi=tensor(3.0), mode=1)


# state = cir(state) 

# # samples = state.homodyne_measure(phi=0., mode=0, shots=1) 

# samples, state = homodyne_measure(state, phi=0, mode=0, shots=1)   

# print('measurements:', samples)
# print('collapsed state:', state.tensor)
# print('collapsed state:', state.tensor.shape)



# #------QML Example 1 

# x = torch.tensor([[1.7, 2.1, -4.0], 
#                   [-1.1, 1.3, 2.0]])

# state = FockState(batch_size=2, n_modes=2, cutoff=2)


# encoding_cir = QumodeCircuit(n_modes=2, backend='fock')
# encoding_cir.displace(mode=0) 
# encoding_cir.displace(mode=1)


# var_cir = QumodeCircuit(n_modes=2, backend='fock')
# var_cir.add(Displacement(mode=0))
# var_cir.add(Displacement(mode=1))



# # forward iterations 

# # loop over encoding_cir to encoding classocal data into quantum state
# for i, op in enumerate(encoding_cir.operators):
#     op.set_params(r=x[:,i], phi=torch.tensor(0.0)) 




# # state.reset()
# # state = encoding_cir(state)
# # state = var_cir(state)


# state.reset()
# cir = encoding_cir + var_cir + encoding_cir + var_cir # data re-uploading
# state = cir(state)

# print('state tensor', state.tensor)

# # yhat = quad_expectation(state, phi=0, mode=0)


# for pn, p in cir.named_parameters():
#     print(pn, p.grad)
    
# state.tensor.sum().backward()

# for pn, p in cir.named_parameters():
#     print(pn, p.grad)




# #------QC Example 2
# state = FockState(batch_size=3, n_modes=2, cutoff=15, pure=True, dtype=torch.complex128) 

# cir = QumodeCircuit(n_modes=2, backend='fock')
# cir.add(Displacement(mode=0))
# cir.operators[0].set_params(r=tensor(1.0), phi=tensor(2.0))
# bs = BeamSplitter(mode1=0, mode2=1)
# cir.add(bs)
# bs.set_params(theta=tensor(0.2), phi=tensor(0.0))
# print('theta', 'phi', bs.theta, bs.phi)
# bs.set_params(theta=tensor(0.5), phi=tensor(0.5))
# print('theta', 'phi', bs.theta, bs.phi)
# state = cir(state) 

# samples = state.homodyne_measure(phi=0., mode=0, shots=1) 



# print('measurements:', samples)
# print('collapsed state:', state.tensor.shape)


#------QML Example 2

cutoff = 15


class CVQNN(nn.Module):
    """
    https://strawberryfields.ai/photonics/demos/run_quantum_neural_network.html
    """
    def __init__(self, batch_size):
        super().__init__()
        self.var_cir = QumodeCircuit(batch_size=batch_size, n_modes=1, backend='fock', cutoff=cutoff, dtype=torch.complex64)
        self._build_cir()
        
    def _build_cir(self):
        for i in range(5):
            self.var_cir.phase_shift(mode=0)
            self.var_cir.squeeze(mode=0)
            self.var_cir.phase_shift(mode=0)
            self.var_cir.displace(mode=0)
            self.var_cir.kerr(mode=0)

    def forward(self):
        # Applies layers of gates to the initial state
        state = self.var_cir()
        return  state.tensor.squeeze()


model = CVQNN(batch_size=1)
model.to(torch.device("cuda"))
print(model)


target_state = np.zeros(cutoff)
target_state[1] = 1
target_state = torch.tensor(target_state, dtype=torch.complex64, device="cuda")

yhat = model() # Note：这时候模型还没有参数, 一定要先跑一次forward，才能有参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  

history = []
for i in range(1000):
    yhat = model()
    fidelity = torch.abs(torch.sum(torch.conj(yhat) * target_state)) ** 2
    modulus = torch.abs(torch.sum(torch.conj(yhat) * yhat)) ** 2
    loss = 1 - fidelity
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step: {i} Loss: {loss:.4f} Fidelity: {fidelity:.4f} Modulus: {modulus:.4f}")
    history.append(loss.item())


plt.plot(history)
plt.show()
    







