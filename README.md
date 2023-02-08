# DeepQuantum

DeepQuantum aims at establishing a bridge between artificial intelligence (AI) and quantum computing (QC). It provides strong support for the scientific research community and developers in the field to easily develop QML applications.

## Install DeepQuantum

Install DeepQuantum with the following commands.

> git clone http://gitlab.turingq.com/deepquantum/deepquantum.git
>
> cd deepquantum
>
> pip install -e .

# Example

```python
# use common gates
x = torch.tensor((0,0,0,0,0,0,1,0)) + 0j
cnot = CNOT(nqubit=3, wires=[1,0])
print(cnot(x))
ry = Ry(requires_grad=True)
print(ry.matrix)
theta = torch.randn(1, requires_grad=True)
phi = torch.randn(1, requires_grad=True)
lambd = torch.randn(1, requires_grad=True)
print(Rz(inputs=2.).matrix)
print(Rz().get_matrix(2.))
print(Rz().get_matrix(theta))
theta = torch.tensor(1., requires_grad=True)
phi = torch.tensor(1., requires_grad=True)
lambd = torch.tensor(1., requires_grad=True)
u3 = U3Gate(inputs=[theta, phi, lambd])
print(u3.matrix)
print(U3Gate().get_matrix(theta, phi, lambd))

# PQC
N = 4
batch = 2
# amplitude encoding
data = torch.randn(batch, 2 ** N)
cir = Circuit(N)
cir.amplitude_encoding(data)
cir.rxlayer()
cir.rylayer(wires=0)
cir.cnot_ring()
cir.measure(wires=0, observables='x')
state = cir()
exp = cir.expectation()
print(state, state.norm(dim=-2))
print(cir.state.shape)
print(exp)
# angle encoding
data = torch.randn(batch, N)
data = torch.arctan(data)
cir = Circuit(N)
cir.hlayer()
cir.rxlayer(encode=True)
cir.rylayer(wires=[0, 2])
cir.cnot_ring()
cir.measure(wires=[0, 2])
state = cir(data)
exp = cir.expectation()
print(state, state.norm(dim=-2))
print(cir.state.shape)
print(exp)
```

