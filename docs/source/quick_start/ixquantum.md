# Single-GPU simulation

`QubitCircuitGPU` runs quantum circuits on one NVIDIA GPU with cuQuantum or one Iluvatar GPU with ixquantum.
It uses the same gate construction methods as `QubitCircuit` and supports statevector simulation, measurement sampling, and adjoint gradients.

## Installation

### NVIDIA cuQuantum

Install NVIDIA CUDA-enabled PyTorch, a matching CUDA toolkit with `nvcc`, and a C++17 compiler.
The CUDA runtime alone is not sufficient to compile the extension.

```bash
python -m pip install 'deepquantum[cuquantum]'
```

Set `CUDA_HOME` to the CUDA toolkit directory.
The loader finds the cuStateVec SDK installed by `custatevec-cu12` in the active Python environment.
For a separate SDK installation, set `CUQUANTUM_ROOT` to its root directory containing `include/` and `lib/` or `lib64/`.

### Iluvatar ixquantum

Obtain the GPU driver, CoreX toolkit, CoreX-compatible PyTorch, and ixquantum SDK from Iluvatar (天数智芯).
The [Iluvatar developer downloads page](https://developer.iluvatar.com/downloads) provides contact information at the bottom of the page.
This backend has been tested on Linux x86_64 with a BI-V150 GPU, CoreX 5.0.0, CoreX PyTorch 2.10.0, and ixquantum 1.8.0.
A C++17 compiler and Python development headers are also required.

For CoreX installed under `/usr/local/corex`, configure the environment as follows:

```bash
export PATH="/usr/local/corex/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:/usr/local/corex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export IXQUANTUM_ROOT=/path/to/ixquantum
python -m pip install 'deepquantum[ixquantum]'
```

Set `IXQUANTUM_ROOT` to the directory containing `custatevec.h` and `libcustatevec.so.1.8.0`, or an SDK directory with `include/` and `lib/` or `lib64/`.
If unset, it defaults to `/usr/local/corex`.

For a source checkout, use `python -m pip install -e '.[cuquantum]'` or `python -m pip install -e '.[ixquantum]'`.
The extension compiles on first use and is cached for subsequent runs.

## Build and run a circuit

Choose the GPU with `device` when constructing the circuit.
By default, the backend is selected from the installed PyTorch build.
You can also set `backend='cuquantum'` or `backend='ixquantum'` explicitly.

The following example prepares a three-qubit GHZ state:

```python
import deepquantum as dq

circuit = dq.QubitCircuitGPU(3, device='cuda:0')
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)

state = circuit()
print(state.shape)                 # torch.Size([8, 1])
print(circuit.measure(shots=1024))  # approximately equal '000' and '111' counts
circuit.close()
```

The state is a `complex64` tensor, with qubit 0 as the most significant bit, matching `QubitCircuit`.
Its storage is reused by later executions; use `state.clone()` to keep an independent copy.
`close()` releases the circuit's GPU resources.
To use a different GPU, construct a new circuit on that device.

## Train a circuit

Gates added without fixed inputs have trainable parameters.
For Pauli observables, when gradients are enabled and parameters require gradients, `expectation()` computes the forward result and uses adjoint differentiation during `backward()`.

```python
import torch
import deepquantum as dq

circuit = dq.QubitCircuitGPU(2, device='cuda:0')
circuit.ry(0)
circuit.cnot(0, 1)
circuit.observable(1, 'z')

optimizer = torch.optim.Adam(circuit.parameters(), lr=0.05)
for _ in range(10):
    optimizer.zero_grad()
    loss = circuit.expectation().sum()
    loss.backward()
    optimizer.step()

# Read the trained circuit without computing gradients.
with torch.no_grad():
    circuit()
    value = circuit.expectation()

circuit.close()
```

Complete `backward()` before running or modifying the same circuit again.
For inference, call `circuit()` before `expectation()`, as shown above.

## Supported operations

Supported gates include fixed unitary matrices, controlled gates, Pauli rotations, and phase gates.
Gradients are available for supported scalar rotations and phase gates with Pauli observables, to first order only.
The returned statevector is not differentiable.

`measure()` returns sampled bit-string counts without changing the final state.
Use `observable()` for Pauli observables or `observable_matrix(matrix, wires)` for a general fixed Hermitian observable.
Matrix observables currently support expectation values without gradients and must be used separately from Pauli observables.

Runtime data encoding, batches, density matrices, noise, mid-circuit measurements, and conditional operations are currently not supported.
Individual inverse gates are supported; whole-circuit `inverse()` and circuit addition with `+` are not.

Updating trainable parameter values does not require recompilation.
After directly changing a fixed gate parameter or matrix, call `circuit.invalidate()` before the next execution.
