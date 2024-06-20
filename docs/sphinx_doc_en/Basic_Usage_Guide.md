# 2. Basic Usage Guide

Import DeepQuantum and related libraries.

```python
import deepquantum as dq
import torch
import torch.nn as nn
```

## 2.1 Quantum Circuit Object

The fundamental object in DeepQuantum is QubitCircuit.
Initialize a quantum circuit with the number of qubits (n), i.e., cir=dq.QubitCircuit(n).
DeepQuantum can help users to easily implement parameterized quantum circuits for quantum machine learning.

## 2.2 Quantum State

QubitState is a class representing quantum states. For example, we can use QubitState to prepare a single-qubit quantum state, with its data being a torch tensor, stored in the attribute state.

Conveniently encode classical data into a quantum state

```python
qstate = dq.QubitState(nqubit=1, state=[0,1])
print(qstate.state)
```
Additionally, some built-in quantum states can be invoked.

Initialize to all-zero state by default

```python
qstate = dq.QubitState(nqubit=2, state='zeros')
print(qstate.state)
```

Initialize to equal-weighted superposition state

```python
qstate = dq.QubitState(nqubit=2, state='equal')
print(qstate.state)
```

Initialize to GHZ state

```python
qstate = dq.QubitState(nqubit=3, state='ghz')
print(qstate.state)
```

## 2.3 Basic Quantum Gates

We can act various quantum gates on the QubitCircuit. For example, we can act the Hadamard gate on qubit 1: cir.h(1) or act the Rx gate on qubit 2: cir.rx(2, inputs=0.2), and it is similar for multi-qubit gates: cir.cnot(0, 1).
Also, we can place a layer of quantum gates on the quantum circuit using one line, i.e., cir.rxlayer(). If the quantum gate does not have specified input parameters, variational parameters will be initialized automatically.

```python
cir = dq.QubitCircuit(4)
```

The first parameter 'wires' specifies where to place the gates.
The internal variational parameters are automatically initialized.

```python
cir.rxlayer(wires=[0,2])
```

We can also manually initialize the parameters, as shown below.

Use 'inputs' to manually initialize fixed parameters

```python
cir.rxlayer(wires=[0, 1, 2, ...], inputs=[theta_0, theta_1, ...])
```

If the parameters are required to be trainable, see the following example

```python
class MyCircuit(nn.Module):
    def __init__(self, nqubit):
        super().__init__()
        # Manually initialize the variational parameter to 1
        self.params = nn.Parameter(torch.ones(nqubit))
        self.cir = self.circuit(nqubit)

    def circuit(self, nqubit):
        cir = dq.QubitCircuit(nqubit)         
        cir.hlayer()
        # Using 'encode', specify where the variational parameters  
        # are encoded into the quantum circuit
        cir.rylayer(encode=True)
        cir.cnot_ring()
        for i in range(nqubit):
            cir.observable(i)
        return cir

    def forward(self):
        # During the forward process, variational parameters are 
        # added to the quantum circuit as 'data'
        self.cir(data=self.params)
        return self.cir.expectation().mean()
```
DeepQuantum supports the flexible use of multi-controlled quantum gates (i.e., the controlled quantum gate activates only when all control bits are 1), as illustrated below.

```python
cir = dq.QubitCircuit(4)
cir.toffoli(0, 1, 2) # Specify control bits and target bits in sequence
cir.fredkin(0, 1, 2)
# General quantum gates can specify any control bits using the 'controls' parameter
cir.x(3, controls=[0,1,2])
cir.crx(0, 1)
cir.crxx(0, 1, 2)
cir.u3(2, controls=[0,1,3])
cir.draw()
```

## 2.4 Measurement and Expectation

### 2.4.1 Measurement

Measurement is one of the core operations in quantum computing. We take the measurement of a GHZ state as an example.

```python
cir = dq.QubitCircuit(3)
cir.h(0)
cir.cnot(0, 1)
cir.cnot(0, 2)
cir.barrier()
cir()
# Measure the final state in QubitCircuit, and the returned result
# is a dictionary or a list of dictionaries.
# The dictionary uses bit strings as keys and their measured counts as values.
# The default value of 'shots' is 1024.
# The bit string from left to right corresponds to the order of wires, which means
# the first qubit is at the top, and the last qubit is at the bottom.
print(cir.measure())
# We can also set the sampling number, perform partial measurements, 
# and display ideal probabilities.
print(cir.measure(shots=100, wires=[1,2], with_prob=True))
```

### 2.4.2 Expectation

When employing parameterized quantum circuits in variational quantum algorithms, it often involves calculating the expectation value of an observable for the final state. Here, we'll demonstrate this with the simplest quantum circuit.

```python
cir = dq.QubitCircuit(4)
cir.xlayer([0,2])
# Multiple observables can be added, and the results of 
# each expectation value will be automatically concatenated
# Flexibly specify measurement wires and bases using a list-based combination
# e.g., wires=[0,1,2]、basis='xyz' representing the observable whose
# wires 0, 1, and 2 corresponds to Pauli-X, Pauli-Y, and Pauli-Z, respectively
for i in range(4):
    cir.observable(i)      
cir() # Expectation value can be obtained after running the circuit
print(cir.expectation())
```

### 2.4.3 Conditional Measurement

The condition parameter allows conditional measurement, and the related wires are determined by control bits controls.

```python
cir = dq.QubitCircuit(3)
cir.h(0)
cir.x(1, controls=0, condition=True)
cir.x(2, controls=1, condition=True)
print(cir())
# Conduct random delay measurements
state, measure_rst, prob = cir.defer_measure(with_prob=True)
print(state)
# Choose specific measurement results
print(cir.post_select(measure_rst))
cir.draw()
```

Note: defer_measure and post_select do not alter the final state state stored by QubitCircuit, making them incompatible with measure and expectation for conditional measurement.