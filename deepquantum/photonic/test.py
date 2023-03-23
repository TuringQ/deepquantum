import torch
from torch import tensor
from qumode import QumodeCircuit, FockState
from fock.ops import Displacement



#------QC
state = FockState(n_modes=2, cutoff=2) 

cir = QumodeCircuit(n_modes=2, backend='fock')
cir.displace(r=tensor(1.0), phi=tensor(0.0), mode=0)
cir.displace(r=tensor(2.0), phi=tensor(0.0), mode=1)
#cir.displace(r=tensor(3.0), phi=tensor(3.0), mode=0)

state = cir(state) 

# samples = homodyne_measure(state, phi=0, mode=0)   # (shots, modes)

# breakpoint()

#------QML

x = torch.tensor([[1.7, 2.1, -4.0], 
                  [-1.1, 1.3, 2.0]])

state = FockState(batch_size=2, n_modes=2, cutoff=2)


encoding_cir = QumodeCircuit(n_modes=2, backend='fock')
encoding_cir.displace(mode=0) 
encoding_cir.displace(mode=1)


var_cir = QumodeCircuit(n_modes=2, backend='fock')
var_cir.add(Displacement(mode=0))
var_cir.add(Displacement(mode=1))



# forward iterations
encoding_cir.operators[0].set_params(r=x[:,0], phi=torch.tensor(0.0))
encoding_cir.operators[1].set_params(r=x[:,1], phi=torch.tensor(2.0))




# state.reset()
# state = encoding_cir(state)
# state = var_cir(state)


state.reset()
cir = encoding_cir + var_cir + encoding_cir + var_cir # data re-uploading
state = cir(state)

print('state tensor', state.tensor)

# yhat = quad_expectation(state, phi=0, mode=0)


for pn, p in cir.named_parameters():
    print(pn, p.grad)
    
state.tensor.sum().backward()

for pn, p in cir.named_parameters():
    print(pn, p.grad)

