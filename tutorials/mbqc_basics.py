# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dq_cuda_251
#     language: python
#     name: python3
# ---

# %%
import deepquantum as dq
import numpy as np
import torch

print('version', dq.__version__)

# %% [markdown]
# # 基于测量的量子计算（MBQC）模块

# %% [markdown]
# ## 构建Pattern

# %% [markdown]
# ### 通过转译QubitCircuit线路构建Pattern

# %% [markdown]
# 构建``QubitCircuit``

# %%
cir = dq.QubitCircuit(2)
cir.h(0)
cir.h(1)
cir.cnot(0, 1)
cir.draw()

# %% [markdown]
# 转译为MBQC的``Pattern``类

# %%
# Transpile circuit to measurement pattern
pattern = cir.pattern()
print(pattern)

# %% [markdown]
# 如果``QubitCircuit``初态是batch形式，转译后的``Pattern``初态依然是batch形式。

# %%
n_qubits = 2
batch_size = 5
init_state = torch.rand(batch_size, 2**n_qubits)  # 输入QubitCiurcuit后会自动归一化

cir = dq.QubitCircuit(n_qubits, init_state=init_state)
cir.h(0)
cir.h(1)
cir.cnot(0, 1)

pattern = cir.pattern()
print(pattern.init_state.full_state.shape)
print(pattern.init_state.full_state)

# %% [markdown]
# 可视化构建的图态。其中，方形node表示输入，edge表示node间存在CZ纠缠，蓝色node表示待测，剩余的灰色node表示输出。
#
# 绿色/红色虚线表示测量角度对于t domain/s domain的依赖。因为转译完得到的是未经优化的wild pattern，中间的Z/X修正尚未转移到测量角度上，所以图中并没有显示。

# %%
pattern.draw()

# %% [markdown]
# ### 手动构建Pattern

# %% [markdown]
# 除了通过``QubitCircuit``的转译，用户可以初始化``Pattern``后，通过手动添加``NEMC``commands，构建自定义的``Pattern``。

# %% [markdown]
# 输入的节点可以用初始化参数中``nodes_state``设置，类型可以是``int``，代表初始化节点的数量，也可以用``List``指定节点的编号。
#
# 而初态用参数``state``设置，默认为全 $ \left |+\right \rangle$ 态。除了输入自定义的态矢，可以用``str``类型的输入，支持``'plus'``， ``'minus'``， ``'zero'``和``'one'``。

# %%
pattern = dq.Pattern(nodes_state=[0, 1])
## 等效为 pattern = dq.Pattern(nodes_state=2)

print(pattern.init_state.full_state)

pattern.draw()

# %%
# 自定义初态
pattern = dq.Pattern(nodes_state=[0, 1], state=[1, 0, 0, 0])
print(pattern.init_state.full_state)

# 初态str表示
pattern = dq.Pattern(nodes_state=[0, 1], state='minus')
print(pattern.init_state.full_state)

pattern = dq.Pattern(nodes_state=[0, 1], state='zero')
print(pattern.init_state.full_state)

pattern = dq.Pattern(nodes_state=[0, 1], state='one')
print(pattern.init_state.full_state)

# %% [markdown]
# 可以在`nodes_state`的基础上，额外加入`edges`和`nodes`作为输入的初始图态。

# %%
pattern = dq.Pattern(nodes_state=[0, 1], nodes=[2, 3], edges=[[2, 3]])

pattern.draw()

# %% [markdown]
# 根据NEMC Commands序列，生成特定的``Pattern``

# %% [markdown]
# | Command | 定义 | Pattern函数 |
# |------|------|------|
# | $N_i$ | Node (qubit) preparation command with node index $i$ | $n(i)$ |
# | $E_{ij}$ | Entanglement command which apply $CZ$ gate to nodes $(i, j)$ | $e(i, j)$ |
# | $^t[M_i^{\lambda, \alpha}]^s$ | Measurement command which perform measurement of node $i$ ,with <br>measurement plane $\lambda = XY, YZ$ or $XZ$, <br>measurement angle $\alpha$ defined on the plane $\lambda$, <br>$s$ and $t$ feedforward domains that adaptively changes the measurement angles to $\alpha' = (-1)^{q_s}\alpha + \pi q_t$, <br>where $q_s, q_t$ are the sum of all measurement outcomes in the $s$ and $t$ domains. | $m(i, \alpha, \lambda, t, s)$ |
# | $X_i^s$ | Correction X command applied to qubit $i$ with signal domain $s$ | $x(i, s)$ |
# | $Z_i^s$ | Correction Z command applied to qubit $i$ with signal domain $s$ | $z(i, s)$ |

# %% [markdown]
# 例如，生成 $ X_3^{s_0+s_1}\ ^{t_1} [M_2^{\pi}]^{s_0}E_{23}N_3[M_1^{\pi}]^{s_0} M_0^{\pi}E_{12}E_{02}N_2E_{01}N_1N_0$

# %%
pattern = dq.Pattern(nodes_state=[0, 1])
pattern.n(2)
pattern.e(0, 2)
pattern.e(1, 2)
pattern.m(node=0, angle=np.pi)
pattern.m(node=1, angle=np.pi, s_domain=[0])
pattern.n(3)
pattern.e(2, 3)
pattern.m(node=2, angle=np.pi, s_domain=[0], t_domain=[1])
pattern.x(node=3, domain=[0, 1])
pattern.draw()

# %% [markdown]
# ## 优化Pattern

# %% [markdown]
# 进行``standardize``操作，将``Pattern``从右向左按NEMC的指令类型进行排列，形成标准形式。
#
# 注意，相比wild pattern，standard form会占用更多内存。如果标准化后出现内存溢出的报错，可以尝试对wild pattern直接进行前向演化。

# %%
pattern.standardize()
print(pattern)

# %% [markdown]
# 通过signal shifting 进一步优化，目的是消除测量角度对于``t_domain``的依赖(Z-dependency)，从而降低量子深度。

# %%
pattern.shift_signals()
print(pattern)

# %% [markdown]
# 通过图态的可视化，可以观察到signal shifting后的测量不再依赖于绿色的Z修正。

# %%
pattern.draw()

# %% [markdown]
# ## 执行MBQC模拟

# %% [markdown]
# ### 前向演化
#
# 和DeepQuantum其它模块一样，仅需一个前向函数即可执行对``Pattern``的模拟。

# %%
pattern()

# %% [markdown]
# 返回的结果是末态输出的``GraphState``，使用属性``full_state``可以得到末态的态矢。

# %%
state = pattern().full_state
print(state)

# %% [markdown]
# 对应的测量结果由``GraphState``中``measure_dict``保存：

# %%
print(pattern.state.measure_dict)

# %% [markdown]
# 支持通过 ``data`` 输入``Pattern``中 ``encode=True`` 的command参数(即测量角度)，通过 ``state`` 指定初态，对``Pattern``进行演化：

# %%
pattern = dq.Pattern(nodes_state=[0, 1])
pattern.n(2)
pattern.e(0, 2)
pattern.e(1, 2)
pattern.m(node=0, encode=True)
pattern.m(node=1, encode=True, s_domain=[0])
pattern.x(node=2, domain=[0, 1])

angle = torch.randn(2)
print(pattern(data=angle).full_state)

# %% [markdown]
# 初态的类型需要是``GraphState``：

# %%
init_graph_state = dq.GraphState([0, 1], state=[1, 0, 0, 0])

print(pattern(data=angle, state=init_graph_state).full_state)

# %% [markdown]
# ### 支持batch输入

# %% [markdown]
# 测量角度的batch输入:

# %%
angle = torch.randn(6, 2)
print(pattern(data=angle).full_state)

# %% [markdown]
# 初态的batch输入：

# %%
init_graph_state = dq.GraphState([0, 1], state=[[1, 0, 0, 0], [0, 1, 0, 0]])

print(pattern(state=init_graph_state).full_state)

# %% [markdown]
# 也支持转译前``QubitCircuit``中参数化门的encode：

# %%
cir = dq.QubitCircuit(2)
cir.h(0)
cir.rz(0, encode=True)
cir.ry(1, encode=True)
cir.cnot(0, 1)

pattern = cir.pattern()

data = torch.randn(2)
print(pattern(data=data).full_state)

data = torch.randn(6, 2)
print(pattern(data=data).full_state)

# %% [markdown]
# ### 支持自动微分

# %% [markdown]
# MBQC模块支持基于``PyTorch``的自动微分，用户可以利用这一特性设计和模拟含梯度优化的变分算法(VMBQC)。

# %%
data = torch.randn(2, requires_grad=True)
print(pattern(data=data).full_state)
