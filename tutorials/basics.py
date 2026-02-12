# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: dq
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial

# %% [markdown]
# import DeepQuantum以及相关的库。

# %%
import deepquantum as dq
import numpy as np
import torch
import torch.nn as nn

# %% [markdown]
# ## 基本的量子门
#
# 所有的Gate都是Operation的子类，都有布尔变量`den_mat`和`tsr_mode`。
#
# 参数`den_mat`表示这个操作是处理密度矩阵，还是态矢。
#
# 参数`tsr_mode`表示这个操作的输入输出是张量态（形状为（batch, 2, ..., 2）的tensor），还是态矢或密度矩阵。

# %% [markdown]
# 我们利用`QubitState`准备一个单比特量子态（默认为'zeros'表示全0态，此外，有'equal'表示等权叠加态，'ghz'表示GHZ态），其数据是torch的tensor，存在属性`state`中。

# %%
qstate = dq.QubitState(nqubit=1, state=[0, 1])
state1 = qstate.state
print(state1)

# %% [markdown]
# 实例化一个单比特量子门，可以从`matrix`属性获得量子门的基础表示（相对于指定任意控制位的量子门而言）。

# %%
x = dq.PauliX()
print(x.matrix)

# %% [markdown]
# 用量子门对量子态进行操作，既可以通过手动的矩阵乘法，也可以通过把量子门直接作用在量子态上。

# %%
print(x.matrix @ state1)
print(x(state1))

# %% [markdown]
# 我们再看带参数的量子门的例子。

# %%
rx = dq.Rx(torch.pi / 2)
print(rx.matrix @ state1)
print(rx(state1))

# %% [markdown]
# 可以利用`get_matrix()`获取带参数量子门的计算过程。

# %%
rx.get_matrix(torch.pi) @ state1

# %% [markdown]
# 注意`matrix`只是纯粹的矩阵表示，因此当需要记录计算图来求参数的梯度时，必须使用`update_matrix()`或`get_matrix()`，区别在于前者使用该量子门本身的参数，而后者需要输入参数，适用于外部的指定参数。

# %%
rx = dq.Rx(torch.pi / 2, requires_grad=True)
theta = nn.Parameter(torch.tensor(torch.pi / 2))
print(rx.matrix)
print(rx.update_matrix())
print(rx.get_matrix(theta))

# %%
print(rx.matrix @ state1)
print(rx.update_matrix() @ state1)
print(rx.get_matrix(theta) @ state1)
print(rx(state1))

# %% [markdown]
# 处理多比特量子态时，量子门的`nqubit`需要与量子态的一致，用`wires`来指定量子门作用于哪些线路上。同样，既可以通过`get_unitary()`得到完整的酉矩阵后进行手动的矩阵乘法，也可以通过把量子门直接作用在量子态上。需要注意，前者的计算效率远低于后者。

# %%
state2 = dq.QubitState(nqubit=2, state=[0, 0, 0, 1]).state
print(state2)
rx = dq.Rx(torch.pi / 2, nqubit=2, wires=[1], requires_grad=True)
print(rx.get_unitary() @ state2)
print(rx(state2))

# %% [markdown]
# DeepQuantum中几乎所有的量子门都支持额外指定任意多的控制位（例外有`CNOT`、`Toffoli`、`Fredkin`等）。因此，同一个量子门可能会有不止一种调用方法。比如cnot和Toffoli可以用`PauliX`实现，Fredkin可以用`Swap`实现。

# %%
cx1 = dq.CNOT(wires=[0, 1])
cx2 = dq.PauliX(nqubit=2, wires=[1], controls=[0])

ccx1 = dq.Toffoli(wires=[0, 1, 2])
ccx2 = dq.PauliX(nqubit=3, wires=[2], controls=[0, 1])

cswap1 = dq.Fredkin(wires=[0, 1, 2])
cswap2 = dq.Swap(nqubit=3, wires=[1, 2], controls=[0])

# %% [markdown]
# 注意，它们的`matrix`是不同的。并且，前者的计算效率略高于后者。

# %%
print(cx1.matrix)
print(cx2.matrix)

print(ccx1.matrix)
print(ccx2.matrix)

print(cswap1.matrix)
print(cswap2.matrix)

# %% [markdown]
# 后者可以用`get_unitary()`来检查。

# %%
print(cx2.get_unitary())
print(ccx2.get_unitary())
print(cswap2.get_unitary())

# %% [markdown]
# 用户还可以通过`UAnyGate`来封装任意酉矩阵。比如，把$4\times4$的酉矩阵作用在三比特量子态的后两个量子比特上，其中用`minmax`指定作用范围，需要和酉矩阵的大小匹配。

# %%
unitary = [[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
u = dq.UAnyGate(unitary=unitary, nqubit=3, minmax=[1, 2])
print(u.get_unitary())

# %% [markdown]
# ## 制备GHZ态
#
# 我们来实现一下制备GHZ态（$|\psi\rangle = \left(|000\rangle+|111\rangle\right)/\sqrt{2}$）这个经典的例子。

# %%
state3 = dq.QubitState(3).state
h = dq.Hadamard(nqubit=3, wires=0)
cx1 = dq.CNOT(nqubit=3, wires=[0, 1])
cx2 = dq.CNOT(nqubit=3, wires=[0, 2])
print(cx2(cx1(h(state3))))

# %% [markdown]
# ## 量子线路：QubitCircuit
#
# 量子线路是DeepQuantum的核心对象。通过`QubitCircuit`进行初始化，然后可以在实例对象上添加各种量子门，最后进行演化和测量。我们把上面这个例子用量子线路来实现。并且可以对线路进行可视化。

# %%
cir = dq.QubitCircuit(3)
cir.h(0)
cir.cnot(0, 1)
cir.cnot(0, 2)
print(cir())
cir.draw()

# %% [markdown]
# 我们可以对线路进行测量，返回的结果是字典或者字典的列表，字典的key是比特串，value是对应测量到的次数，shots默认为1024。

# %%
cir.barrier()
print(cir.measure())
cir.draw()

# %% [markdown]
# 也可以设定采样次数、进行部分测量以及显示理想的概率。

# %%
print(cir.measure(shots=100, wires=[1, 2], with_prob=True))
cir.draw()

# %% [markdown]
# 再来看一个对CNOT门实现分解的例子：$\text{CNOT}=e^{-i{\frac {\pi }{4}}}R_{y_{1}}(-\pi /2)R_{x_{1}}(-\pi /2)R_{x_{2}}(-\pi /2)R_{xx}(\pi /2)R_{y_{1}}(\pi /2)$

# %%
cir = dq.QubitCircuit(2)
cir.ry(0, torch.pi / 2)
cir.rxx([0, 1], torch.pi / 2)
cir.rx(1, -torch.pi / 2)
cir.rx(0, -torch.pi / 2)
cir.ry(0, -torch.pi / 2)
print(np.exp(-1j * np.pi / 4) * cir.get_unitary())
cir.draw()

# %% [markdown]
# CNOT、Toffoli、Fredkin也有不同的API去添加。

# %%
cir = dq.QubitCircuit(3)
cir.cnot(0, 1)
cir.cx(0, 1)
cir.x(1, 0)

cir.toffoli(0, 1, 2)
cir.ccx(0, 1, 2)
cir.x(2, [0, 1])

cir.fredkin(0, 1, 2)
cir.cswap(0, 1, 2)
cir.swap([1, 2], 0)

cir.draw()

# %% [markdown]
# ## 参数化量子线路
#
# DeepQuantum可以帮助用户很方便地实现参数化量子线路，从而进行量子机器学习。
#
# `QubitCircuit`的实例中添加的带参数的量子门，如果没有指定输入参数，会自动初始化变分参数。
#
# 补充说明：如果指定了输入参数，那么输入参数会在量子门中被记录为buffer，从而保留其原来的性质。比如，参数不需要求梯度时，就会保持不变。又比如，参数是上一层神经网络的输出，那么在backward过程中就会记录梯度，但它的更新不是通过`QubitCircuit`而是上一层神经网络本身。

# %%
cir = dq.QubitCircuit(4)
cir.rx(0)
cir.rxx([1, 2])
cir.u3(3)
cir.p(0)
cir.cu(3, 0)
cir.cp(1, 2)
cir.draw()

# %% [markdown]
# 也可以直接添加一层量子门，并通过`wires`指定放置于哪几条线路。

# %%
cir = dq.QubitCircuit(4)
cir.hlayer()
cir.rxlayer([0, 2])
cir.rylayer([1, 3])
cir.u3layer()
cir.cxlayer()
cir.draw()

# %% [markdown]
# `cnot_ring()`可以用`minmax`参数指定线路范围，`step`设定每一对control和target相隔的距离，`reverse`指定是否从大到小。

# %%
cir = dq.QubitCircuit(5)
cir.cnot_ring()
cir.barrier()
cir.cnot_ring(minmax=[1, 4], step=3, reverse=True)
cir.draw()

# %% [markdown]
# ### 振幅编码
#
# 下面我们展示一个振幅编码的例子，先准备一些数据。

# %%
nqubit = 4
batch = 2
data = torch.randn(batch, 2**nqubit)

# %% [markdown]
# 然后构建量子线路，并通过`observable`指定测量线路和测量基底。测量线路和测量基底也可以使用列表形式的组合，如`wires=[0,1,2]`、`basis='xyz'`。

# %%
cir = dq.QubitCircuit(nqubit)
cir.rxlayer()
cir.cnot_ring()
cir.observable(wires=0, basis='z')

# %% [markdown]
# 量子门和观测量分别被记录在`operators`和`observables`中。

# %%
print(cir)

# %% [markdown]
# 振幅编码会自动补0或者舍弃多余的数据，以及进行归一化。
#
# 通过线路的`forward()`得到末态，`forward()`有`data`和`state`两个参数，分别对应放入量子门的数据，以及线路作用的初态，即分别对应角度编码和振幅编码。
#
# 测量期望，输出的形状为(batch, 观测量的数量)。

# %%
state = cir.amplitude_encoding(data)
state = cir(state=state)
exp = cir.expectation()
print(state.shape)
print(state.norm(dim=-2))
print(exp)

# %% [markdown]
# ### 角度编码
#
# 角度编码只需要对相应的Gate或Layer指定`encode=True`，会自动将数据的特征依次加入编码层，多余的会被舍弃。

# %%
nqubit = 4
batch = 2
data = torch.sin(torch.tensor(list(range(batch * nqubit)))).reshape(batch, nqubit)
print(data)

# %% [markdown]
# 这次我们对每条线路都进行一次测量。

# %%
cir = dq.QubitCircuit(nqubit)
cir.hlayer()
cir.rxlayer(encode=True)
cir.cnot_ring()
for i in range(nqubit):
    cir.observable(wires=i)
state = cir(data)
exp = cir.expectation()
print(state.shape)
print(state.norm(dim=-2))
print(exp)

# %% [markdown]
# `QubitCircuit`支持data re-uploading，只需要初始化时指定`reupload=True`，数据就会被循环地放入线路中。
#
# 补充说明：`encode`只能针对一条一维的数据，线路对batch的支持是通过`torch.vmap`，并且计算完一次前向过程会自动初始化`encoders`，因为量子门无法保存多组参数。

# %%
cir = dq.QubitCircuit(nqubit, reupload=True)
cir.rxlayer(encode=True)
cir.cnot_ring()
cir.rxlayer(encode=True)
cir.cnot_ring()
cir.encode(data[0])
print(cir)


# %% [markdown]
# ### 混合量子-经典模型
#
# DeepQuantum基于PyTorch，能够方便自然地实现量子模型和经典模型的混合计算。


# %%
class Net(nn.Module):
    def __init__(self, dim_in, nqubit) -> None:
        super().__init__()
        self.fc = nn.Linear(dim_in, nqubit)
        self.cir = self.circuit(nqubit)

    def circuit(self, nqubit):
        cir = dq.QubitCircuit(nqubit)
        cir.hlayer()
        cir.rxlayer(encode=True)
        cir.cnot_ring()
        for i in range(nqubit):
            cir.observable(wires=i)
        return cir

    def forward(self, x):
        x = torch.arctan(self.fc(x))
        self.cir(x)
        exp = self.cir.expectation()
        return exp


nqubit = 4
batch = 2
nfeat = 8
x = torch.sin(torch.tensor(list(range(batch * nfeat)))).reshape(batch, nfeat)
net = Net(nfeat, nqubit)
y = net(x)
print(net.state_dict())
print('y', y)

# %% [markdown]
# ### 线路拼接以及更灵活地使用数据

# %%
nqubit = 2
batch = 2
data1 = torch.sin(torch.tensor(list(range(batch * nqubit)))).reshape(batch, nqubit)
data2 = torch.cos(torch.tensor(list(range(batch * nqubit)))).reshape(batch, nqubit)
cir1 = dq.QubitCircuit(nqubit)
cir1.rxlayer(encode=True)
cir2 = dq.QubitCircuit(nqubit)
cir2.rylayer(encode=True)
cir3 = dq.QubitCircuit(nqubit)
cir3.rzlayer()

# %% [markdown]
# 通过线路加法来共享变分参数。
#
# 注意，不建议对encoder部分进行复杂的线路加法来共享数据，因为需要保证数据的顺序与encoders完全一致。一旦出现错位，由于共享了encoders，会造成全局的影响。

# %%
data = torch.cat([data1, data2], dim=-1)
cir = cir1 + cir3 + cir2 + cir3
cir.observable(0)
cir.encode(data[0])
print(cir)
cir(data)
print(cir.expectation())

# %% [markdown]
# 上面的结果也可以由多个线路的分段演化得到。

# %%
state = cir1(data1)
state = cir3(state=state)
state = cir2(data2, state=state)
state = cir3(state=state)
cir3.reset_observable()
cir3.observable(0)
print(cir3.expectation())

# %% [markdown]
# 这种方式当然就可以更灵活地使用数据。

# %%
state = cir1(data1)
state = cir2(data2, state=state)
state = cir3(state=state)
state = cir1(data2, state=state)
state = cir2(data1, state=state)
cir2.reset_observable()
cir2.observable(0)
print(cir2.expectation())
