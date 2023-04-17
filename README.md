# DeepQuantum

DeepQuantum is a platform that integrates artificial intelligence (AI) and quantum computing (QC). It enables the development of various quantum machine learning applications. Leveraging the PyTorch deep learning platform for QC, DeepQuantum offers a powerful and easy-to-use tool for researchers and developers in the field to create QML applications. It also serves as a learning platform for quantum computing enthusiasts.

## Install DeepQuantum

Install DeepQuantum with the following commands.

> git clone http://gitlab.turingq.com/deepquantum/deepquantum.git
>
> cd deepquantum
>
> pip install -e .

## Getting Started

Please begin with the tutorial on [basics](./docs/basics.ipynb).

The following is a minimal demo.

```python
import deepquantum as dq
cir = dq.Circuit(2)
cir.h(0)
cir.cnot([0,1])
cir.rx(1, 0.2)
cir.observable(0)
print(cir())
print(cir.expectation())
```