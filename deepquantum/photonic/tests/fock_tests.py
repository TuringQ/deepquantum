import sys
sys.path.append("..")

import torch
from torch import nn
from torch import tensor
from circuit import QumodeCircuit, FockState
from fock.ops import Displacement, BeamSplitter



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



# #------QML, , most of angles are changing for every forward iterqaion

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
#     op.set_params(r=x[:,i], phi=torch.tensor(0.0)) # can i call set_params of the same gate multiple times?




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

# :todo:
class CVQNN(nn.Module):
    """number of parameters per layer 2N*N+3N, N = n_qumodes"""
    def __init__(self, batch_size=10, n_qumodes=2):
        super().__init__()
        assert n_qumodes == 2, "only support 2 qumodes"

        self.encoding_cir = QumodeCircuit(batch_size, n_qumodes, backend='fock')
        self.var_cir = QumodeCircuit(batch_size, n_qumodes, backend='fock')
        self._build_cir()
        

    def _build_cir(self):
        self.encoding_cir.displace(mode=0)
        self.encoding_cir.displace(mode=1)

        self.var_cir.beam_split(mode1=0, mode2=1) 

    def forward(self, x):
        # Load classical data into quantum states
        self.encoding_cir.operators[0].set_params(x[:, 0], tensor(0.0))
        self.encoding_cir.operators[1].set_params(x[:, 1], tensor(0.0))
        # Applies layers of gates to the initial state
        cir = self.encoding_cir + self.var_cir
        state = cir()
        output, _ = state.quad_expectation(phi=tensor(0.0), mode=0)
        return output


model = CVQNN(batch_size=1, n_qumodes=2)
print(model)
print(list(model.named_buffers()))
print(list(model.named_parameters()))

x = torch.tensor([[1.0, -0.8]], dtype=torch.float64)
print('model(x):', model(x))
print(list(model.named_buffers()))
print(list(model.named_parameters()))

print('===============')
for pn, p in model.named_parameters():
    print(pn, p.grad)
    
model(x).sum().backward()

for pn, p in model.named_parameters():
    print(pn, p.grad)


