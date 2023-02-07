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
N = 4
batch = 2
# amplitude encoding
data = torch.randn(batch, 2 ** N)
cir = Circuit(N)
cir.amplitude_encoding(data)
cir.rxlayer()
state = cir()
print(state, state.norm(dim=-2))
# angle encoding
data = torch.randn(batch, N)
cir = Circuit(N)
cir.rxlayer(encode=True)
state = cir(data)
print(state, state.norm(dim=-2))
```

