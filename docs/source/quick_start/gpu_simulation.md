# Single-GPU simulation

`QubitCircuitGPU` runs quantum circuits on one NVIDIA GPU with cuQuantum or one Iluvatar GPU with ixquantum.
It uses the same gate construction methods as `QubitCircuit` and supports statevector simulation, measurement sampling, and adjoint gradients.

## Installation

Run the commands from a DeepQuantum source checkout. Choose the section for your GPU, then run the GHZ example below in that Conda environment.
Both examples use Python 3.10. For CoreX, match the Python version to the PyTorch wheel from Iluvatar (`cp310` for Python 3.10 or `cp311` for Python 3.11).

### NVIDIA cuQuantum

This example requires a compatible NVIDIA driver, the [CUDA 12.6 Toolkit](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html) with `nvcc`, and a C++17 compiler.

```bash
conda create -n dq-cuquantum python=3.10 pip -y
conda activate dq-cuquantum
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[cuquantum]' 'custatevec-cu12==1.14.0'
export CUDA_HOME=/usr/local/cuda-12.6  # Adjust to your CUDA Toolkit path.
```

### Iluvatar ixquantum

Install the GPU driver and CoreX toolkit from Iluvatar (天数智芯). Obtain the CoreX PyTorch wheel and the ixquantum SDK files from Iluvatar as well.

- `custatevec.h` is the C API header used to compile the DeepQuantum GPU extension.
- `libcustatevec.so.1.8.0` is the ixquantum shared library that DeepQuantum links to and calls at runtime.

Please contact Iluvatar for the ixquantum SDK files using the contact details at the bottom of its [developer downloads page](https://developer.iluvatar.com/downloads).
This backend has been tested on Linux x86_64 with a BI-V150 GPU, CoreX 5.0.0, CoreX PyTorch 2.10.0, and ixquantum 1.8.0.
A C++17 compiler and Python development headers are also required.

Set `IXQUANTUM_ROOT` to the directory containing the ixquantum SDK files.
Replace `/path/to/corex-torch.whl` with the path to the CoreX PyTorch wheel.
The commands assume CoreX is installed at `/usr/local/corex`.

```bash
conda create -n dq-ixquantum python=3.10 pip -y
conda activate dq-ixquantum
export PATH="/usr/local/corex/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:/usr/local/corex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export IXQUANTUM_ROOT=/path/to/ixquantum
python -m pip install /path/to/corex-torch.whl
python -m pip install -e '.[ixquantum]'
python -c "import torch; assert getattr(torch, 'corex', False) and torch.cuda.is_available()"
```

The extension compiles on first use and is cached for subsequent runs.

## Build and run a circuit

Choose the GPU with `device` when constructing the circuit.
By default, the backend is selected from the installed PyTorch build.
You can also set `backend='cuquantum'` or `backend='ixquantum'` explicitly.

Run the following Python example in the environment you just configured. It prepares a three-qubit GHZ state and checks the measurement counts.

```python
import deepquantum as dq

circuit = dq.QubitCircuitGPU(3, device='cuda:0')
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)

state = circuit()
counts = circuit.measure(shots=1024)
assert state.shape == (8, 1)
assert set(counts) <= {'000', '111'} and sum(counts.values()) == 1024
print(counts)  # Approximately equal '000' and '111' counts.
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
