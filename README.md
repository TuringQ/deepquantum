# DeepQuantum

<p align="center">
  <a href="https://github.com/TuringQ/deepquantum">
    <img width=70% src="docs/assets/deepquantum_logo_light_v1.png" alt="DeepQuantum logo"/>
  </a>
</p>

[![docs](https://img.shields.io/badge/docs-link-blue.svg)](https://dqapi.turingq.com/)
[![PyPI](https://img.shields.io/pypi/v/deepquantum.svg?logo=pypi)](https://pypi.org/project/deepquantum/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deepquantum)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?logo=apache)](./LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/deepquantum.svg)](https://pypi.org/project/deepquantum/)
[![Downloads](https://static.pepy.tech/badge/deepquantum)](https://pepy.tech/project/deepquantum)

DeepQuantum is a platform that integrates artificial intelligence (AI) and quantum computing (QC). It is an efficient programming framework designed for quantum machine learning and photonic quantum computing. By leveraging the PyTorch deep learning platform for QC, DeepQuantum provides a powerful and easy-to-use tool for creating and simulating quantum circuits and photonic quantum circuits. This makes it ideal for developers to quickly get started and explore the field in depth. It also serves as a valuable learning platform for quantum computing enthusiasts.

# Key Features

<p align="center">
  <a href="https://github.com/TuringQ/deepquantum">
    <img width=90% src="docs/assets/deepquantum_architecture.png" alt="DeepQuantum architecture"/>
  </a>
</p>

- **AI-Enhanced Quantum Computing Framework**: Seamlessly integrated with PyTorch, it utilizes technologies such as automatic differentiation, vectorized parallelism, and GPU acceleration for efficiency. It facilitates the easy construction of hybrid quantum-classical models, enabling end-to-end training and inference.
- **User-Friendly API Design**: The API is designed to be simple and intuitive, making it easy to initialize quantum neural networks and providing flexibility in data encoding.
- **Photonic Quantum Computing Simulation**: The Photonic module includes Fock, Gaussian and Bosonic backends, catering to different simulation needs in photonic quantum computing. It comes with built-in optimizers to support on-chip training of photonic quantum circuits.
- **Large-Scale Quantum Circuit Simulation**: Based on tensor networks, it enables the simulation of 100+ qubits on a laptop, leading the industry in both simulation efficiency and scale.
- **Advanced Architecture for Cutting-Edge Algorithm Exploration**: The first framework to support algorithm design and mapping for time-domain-multiplexed photonic quantum circuits, and the first to realize closed-loop integration of quantum circuits, photonic quantum circuits, and MBQC, enabling robust support for both specialized and universal photonic quantum algorithm design.

# Installation

Before installing DeepQuantum, we recommend first manually installing PyTorch 2.
If the latest version of PyTorch is not compatible with your CUDA version, manually install a compatible PyTorch 2 version.

The PyTorch installation instructions currently recommend:
1. Install [Miniconda](https://docs.anaconda.com/miniconda/) or [Anaconda](https://www.anaconda.com/download).
2. Setup conda environment. For example, run `conda create -n <ENV_NAME> python=3.12` and `conda activate <ENV_NAME>`.
3. Install PyTorch following the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Currently, this suggests running `pip install torch`.

If you want to customize your installation, please follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) to build from source.

To install DeepQuantum with `pip`, run

```bash
pip install deepquantum
# or for developers
pip install deepquantum[dev]
# or use tsinghua source
pip install deepquantum -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Alternatively, to install DeepQuantum from source, run

```bash
git clone https://github.com/TuringQ/deepquantum.git
cd deepquantum
pip install -e .
# or use tsinghua source
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# Getting Started

To begin, please start with the tutorials (CN) on [basics](./docs/basics.ipynb) and [photonic basics](./docs/photonic_basics.ipynb).

Below are some minimal demos to help you get started.

- Quantum circuit

```python
import deepquantum as dq
cir = dq.QubitCircuit(2)
cir.h(0)
cir.cnot(0, 1)
cir.rx(1, 0.2)
cir.observable(0)
print(cir())
print(cir.expectation())
```

- Quantum circuit based on matrix product state

You can simply set `mps=True` in `QubitCircuit` and adjust the bond dimension `chi` to control the complexity.

```python
cir = dq.QubitCircuit(2, mps=True, chi=4)
cir.h(0)
cir.cnot(0, 1)
cir.rx(1, 0.2)
cir.observable(0)
print(cir())
print(cir.expectation())
```

- Photonic quantum circuit with the Fock backend, based on Fock basis state

```python
cir = dq.QumodeCircuit(2, [1,1])
cir.dc([0,1])
cir.ps(0, 0.1)
cir.bs([0,1], [0.2,0.3])
print(cir())
print(cir.measure())
```

- Photonic quantum circuit with the Fock backend, based on Fock state tensor

```python
cir = dq.QumodeCircuit(2, [(1, [1,1])], basis=False)
cir.dc([0,1])
cir.ps(0, 0.1)
cir.bs([0,1], [0.2,0.3])
print(cir())
print(cir.measure())
```

- Photonic quantum circuit with the Gaussian backend

```python
cir = dq.QumodeCircuit(2, 'vac', cutoff=10, backend='gaussian')
cir.s(0, 0.1)
cir.d(1, 0.1)
cir.bs([0,1], [0.2,0.3])
print(cir())
print(cir.measure())
print(cir.photon_number_mean_var(wires=0))
print(cir.measure_homodyne(wires=1))
```

- Photonic quantum circuit with the Bosonic backend

```python
cir = dq.QumodeCircuit(2, 'vac', backend='bosonic')
cir.cat(0, 0.5, 0.0)
cir.gkp(1, 0.5, 0.5)
cir.bs([0,1], [0.2,0.3])
print(cir())
print(cir.photon_number_mean_var(wires=0))
print(cir.measure_homodyne(wires=1))
```

- Pattern of measurement-based quantum computation

```python
pattern = dq.Pattern(2)
# Hadamard gate on qubit 1
pattern.n(2)
pattern.e(1, 2)
pattern.m(1)
pattern.x(2, domain=1)
# CNOT
pattern.n([3,4])
pattern.e(2, 3)
pattern.e(0, 3)
pattern.e(3, 4)
pattern.m(2)
pattern.m(3)
pattern.x(4, domain=3)
pattern.z(4, domain=2)
pattern.z(0, domain=2)
print(abs(pattern().full_state))
```

- Transpile quantum circuit to MBQC pattern

```python
cir = dq.QubitCircuit(2)
cir.h(0)
cir.cnot(0, 1)
cir.rx(1, 0.2)
pattern = cir.pattern()
print(cir())
print(pattern().full_state)
print(cir() / pattern().full_state)
```

- Distributed simulation of quantum circuit

```python
import torch
# OMP_NUM_THREADS=2 torchrun --nproc_per_node=4 main.py
backend = 'gloo' # for CPU
# torchrun --nproc_per_node=4 main.py
backend = 'nccl' # for GPU
rank, world_size, local_rank = dq.setup_distributed(backend)
if backend == 'nccl':
    device = f'cuda:{local_rank}'
elif backend == 'gloo':
    device = 'cpu'
data = torch.arange(4, dtype=torch.float, device=device, requires_grad=True)
cir = dq.DistributedQubitCircuit(4)
cir.rylayer(encode=True)
cir.cnot_ring()
cir.observable(0)
cir.observable(1, 'x')
if backend == 'nccl':
    cir.to(f'cuda:{local_rank}')
state = cir(data).amps
result = cir.measure(with_prob=True)
exp = cir.expectation().sum()
exp.backward()
if rank == 0:
    print(state)
    print(result)
    print(exp)
    print(data.grad)
dq.cleanup_distributed()
```

- Distributed simulation of photonic quantum circuit

```python
# OMP_NUM_THREADS=2 torchrun --nproc_per_node=4 main.py
backend = 'gloo' # for CPU
# torchrun --nproc_per_node=4 main.py
backend = 'nccl' # for GPU
rank, world_size, local_rank = dq.setup_distributed(backend)
nmode = 4
cutoff = 4
data = torch.arange(14, dtype=torch.float) / 10
cir = dq.DistributedQumodeCircuit(nmode, [0] * nmode, cutoff)
for i in range(nmode):
    cir.s(i, encode=True)
for i in range(nmode - 1):
    cir.bs([i, i + 1], encode=True)
if backend == 'nccl':
    data = data.to(f'cuda:{local_rank}')
    cir.to(f'cuda:{local_rank}')
state = cir(data).amps
result = cir.measure(with_prob=True)
if rank == 0:
    print(state)
    print(result)
dq.cleanup_distributed()
```

# License

DeepQuantum is open source, released under the Apache License, Version 2.0.