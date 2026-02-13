# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: qdl
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Basic gates in measurement-based quantum computation (MBQC)

# %%
import deepquantum as dq
import numpy as np

# %% [markdown]
# ## Single-qubit operation $HR_z(-\alpha)|+\rangle$

# %% [markdown]
# Matrix representation for $R_z(\alpha)$ and $|+>$ state
# $$ R_z(\alpha) =
# \begin{pmatrix}
# e^{-i\alpha/2} & 0 \\
# 0              & e^{i\alpha/2}
# \end{pmatrix}\\
# |+> = \frac{\sqrt{2}}{2} \begin{pmatrix}
# 1 \\
# 1 \\
# \end{pmatrix}
# $$
#
# $$ R_z(\alpha)|+> =
# \frac{\sqrt{2}}{2} \begin{pmatrix}
# e^{-i\alpha/2} \\
# e^{i\alpha/2} \\
# \end{pmatrix}
# $$
#
# The following circuits are equivalent
#
# ![MBQC example](./figure/basic_gate_MBQC/RZ.jpg)
# <div style="text-align:center">图 1: MBQC实现单比特门 $HR_z(-\alpha)|+\rangle$ 操作</div>

# %%
# define the angle of Rz gate
alpha = np.pi / 3
cir = dq.QubitCircuit(2)

# prepare cluster state
cir.h(0)
cir.h(1)
cir.cz(0, 1)

# |alpha> basis: apply phase shift and Hadamard gate
cir.p(0, -alpha)
cir.h(0)
cir.barrier()

cir.x(1, controls=0, condition=True)

cir()

# check the output state
state, measure_rst, prob = cir.defer_measure(with_prob=True)
# MBQC has an extra global phase: np.exp(-1j * alpha / 2)
print(state * np.exp(1j * alpha / 2))
print(cir.post_select(measure_rst) * np.exp(1j * alpha / 2))  # choose measurement result

cir.draw()

# %% [markdown]
# Verify single qubit operation $HR_z(-\alpha)|+\rangle$ in the circuit-based quantum computation (CBCQ)

# %%
cir = dq.QubitCircuit(1)
# prepare state |+>
cir.h(0)

# apply Rz and Hadamard gate
cir.rz(0, -alpha)
cir.h(0)

print(cir())
cir.draw()

# %% [markdown]
# ## Random single-qubit-rotation gate

# %% [markdown]
# 对于任意单比特门的实现，采用三个H-Rz gate级联的形式，最后根据测量结果加上对应的Pauli修正
#
# ![MBQC example](./figure/basic_gate_MBQC/Single.jpg)
# <div style="text-align:center">图 2: MBQC single gate 的实现</div>

# %%
alpha = np.pi / 2
beta = np.pi / 3
gamma = np.pi / 4
cir = dq.QubitCircuit(4)

# prepare cluster state
cir.hlayer()
cir.cz(0, 1)
cir.cz(1, 2)
cir.cz(2, 3)
cir.barrier()

# measurement
cir.p(0, -alpha)
cir.h(0)
cir.p(1, -beta)
cir.h(1)
cir.p(2, -gamma)
cir.h(2)

cir.x(3, controls=2, condition=True)
cir.z(3, controls=1, condition=True)
cir.x(3, controls=0, condition=True)
cir()

state, measure_rst, prob = cir.defer_measure(with_prob=True)
print(state * np.exp(1j * (alpha + beta + gamma) / 2))
print(cir.post_select(measure_rst) * np.exp(1j * (alpha + beta + gamma) / 2))  # choose measurement result

cir.draw()

# %% [markdown]
# Verify random single-qubit-rotation gate  in circuit based quantum computation(CBQC)

# %%
# verify
cir = dq.QubitCircuit(1)
cir.h(0)

# alpha, beta, gamma系数的正负是根据测量结果决定的
# q0的结果对应rx门的beta系数的正负
# q1的结果对应第二个rz门的gamma系数的正负
# alpha始终保持不变

cir.rz(0, -alpha)
cir.rx(0, beta * (-1) ** (int(measure_rst[0]) + 1))
cir.rz(0, gamma * (-1) ** (int(measure_rst[1]) + 1))
cir.h(0)

print(cir())
cir.draw()

# %% [markdown]
# ## CNOT gate

# %% [markdown]
# Matrix representation for CNOT gate
# $$ CNOT=
# \begin{pmatrix}
# 1 & 0 & 0 & 0 \\
# 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & 1 \\
# 0 & 0 & 1 & 0
# \end{pmatrix}
# = |0><0| \otimes I + |1><1| \otimes X
# $$
#
# $$
# CNOT|++>=|++> \\
# CNOT|+->=|--> \\
# CNOT|-+>=|-+> \\
# CNOT|-->=|+->
# $$
#
# ![MBQC example](./figure/basic_gate_MBQC/CNOT.jpg)
# <div style="text-align:center">图 3: MBQC CNOT 的实现</div>

# %%
cir = dq.QubitCircuit(4)

# construct cluster state
# cir.x(1)  # input |->|+> state
# cir.x(2)  # input |+>|-> state

cir.hlayer()
cir.swap([1, 2])
cir.cz(0, 1)
cir.cz(0, 2)
cir.cz(0, 3)
cir.barrier()

# measurement
cir.h(0)
cir.h(1)
cir.barrier()

cir.x(3, controls=0, condition=True)
cir.z(2, controls=1, condition=True)
cir.z(3, controls=1, condition=True)

cir()

state, measure_rst, prob = cir.defer_measure(with_prob=True)
print(state)
print(cir.post_select(measure_rst))  # choose measurement result

cir.draw()

# %% [markdown]
# Verify CNOT gate in circuit-based quantum computation (CBQC)

# %%
cir = dq.QubitCircuit(2)
# cir.x(0)
# cir.x(1)

cir.hlayer()
cir.cx(0, 1)
print(cir())
cir.draw()
