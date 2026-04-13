# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dq_draw
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 高斯玻色采样模拟分子振动激发

# %% [markdown]
# 分子中振动模式的激发会影响化学反应的结果。
# 振动激发可以为分子提供额外的动能，以克服特定反应的能量壁垒，并有助于控制分子的稳定性。
# 然而，对涉及分子振动和电子态同时变化的过程进行所有振动能级激发概率模拟是具有挑战性的。
# 本例中，我们演示如何使用高斯玻色取样（GBS）来模拟分子振动激发。

# %% [markdown]
# ### 甲酸分子的振动激发

# %%
import deepquantum as dq
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# 我们还需要描述分子振动模式和分子在振动跃迁过程中几何结构变化的分子参数。
# 这些参数由杜辛斯基矩阵和位移向量表示，这些参数是从原子坐标、原子质量、振动频率以及分子的正常模式获得的。
# 这些分子参数可以通过电子结构计算获得。

# %% [markdown]
# $1^1A'$ 态及$1^2A'$ 态的平衡结构坐标如下

# %%
ri = np.genfromtxt('./data/formic_ri.csv', delimiter=',', skip_header=0)[:, np.newaxis]
rf = np.genfromtxt('./data/formic_rf.csv', delimiter=',', skip_header=0)[:, np.newaxis]

# %% [markdown]
# $1^1A'$ 态及 $1^2A'$ 态的质量加权简正坐标如下

# %%
li = np.genfromtxt('./data/formic_li.csv', delimiter=',', skip_header=0)
lf = np.genfromtxt('./data/formic_lf.csv', delimiter=',', skip_header=0)

# %% [markdown]
# $1^1A'$ 态及 $1^2A'$ 态简正模式频率如下

# %%
omega = np.genfromtxt('./data/formic_omega.csv', delimiter=',', skip_header=0)
omegap = np.genfromtxt('./data/formic_omegap.csv', delimiter=',', skip_header=0)

# %% [markdown]
# 计算中用到的物理学常量

# %%
c = 299792458.0  # 光速
mu = 1.6605390666 * 10**-27  # 原子质量单位
h = 6.62607015 * 10**-34  # 普朗克常数

m_c = 12  # 碳原子相对原子质量
m_h = 1.007825037  # 氢原子相对原子质量
m_o = 15.994914640  # 氧原子相对原子质量

# %% [markdown]
# Duschinsky 矩阵 $U$ 及位移矢量 $\delta$ 计算如下

# %%
u = []
for li_ele in li:
    for lf_elf in lf:
        u.append(np.sum(li_ele * lf_elf))
u = np.array(u[-1::-1]).reshape(7, 7).T
print(u)

# %%
delta = []
m = np.diag([m_c, m_c, m_c, m_o, m_o, m_o, m_o, m_o, m_o, m_h, m_h, m_h, m_h, m_h, m_h])
for i in range(len(omegap)):
    d = lf[i].T @ np.sqrt(m) @ (ri - rf)
    denominator = np.sqrt(h / (4 * np.pi**2 * 100 * omegap[i] * c * mu)) / (10**-10)
    delta.append(d / denominator)
delta = np.array(delta[-1::-1])
print(delta)

# %% [markdown]
# Duschinsky矩阵非对角元素描述了不同电子态之间简正模式的混合情况

# %%
plt.imshow(abs(u), cmap='Greens')
plt.colorbar()
plt.xlabel('Mode index')
plt.ylabel('Mode index')
plt.tight_layout()
plt.show()

# %% [markdown]
# 非对角元素的存在显示了不同电子态之间简正模式的混合。
# 由 Duschinsky 矩阵和位移向量计算 GBS 参数如下：

# %%
pre_transition_squeezing = np.sqrt(omega[-1::-1])
post_transition_squeezing = np.sqrt(omegap[-1::-1])

j_mat = np.diag(post_transition_squeezing) @ u @ np.linalg.inv(np.diag(pre_transition_squeezing))

cl, lambda_1, cr = np.linalg.svd(j_mat)

delta_2 = np.linalg.inv(j_mat) @ delta / np.sqrt(2)
delta_2 = delta_2.flatten()
lambda_2 = np.log(lambda_1)

# %% [markdown]
# 我们现在可以计算每个振动模式中的平均振动量子数

# %%
modes = 7  # 简正模式数量
cutoff = 3
shots = 500000

# %%
cir = dq.photonic.QumodeCircuit(
    nmode=modes,
    init_state='vac',
    # init_state=init_state,
    cutoff=cutoff,
    backend='gaussian',
)

for i in range(modes):
    cir.d(wires=[i], r=delta_2[i])

cir.any(cr, wires=list(range(modes)))

for i in range(modes):
    cir.s(wires=[i], r=-lambda_2[i])

cir.any(cl, wires=list(range(modes)))

state = cir()

# 线路可视化
cir.draw()

# %%
sample = cir.measure(shots=shots, mcmc=True)

# %%
sample2 = []
for ele in sample.items():
    # print(ele[0].state)
    sample2.append(ele[0].state * ele[1])
counts2 = np.sum(sample2, axis=0)

plt.figure(figsize=(8, 4))
plt.ylabel('Photon number')
plt.xlabel(r'Frequency (cm$^{-1}$)')
plt.xticks(range(len(omegap)), np.round(omegap, 1), rotation=90)
plt.bar(range(len(omegap)), counts2, color='green')
plt.tight_layout()
plt.show()

# %% [markdown]
# 现在模拟一个光激发过程，其中涉及从电子基态的预激发振动态进行振动跃迁。
# 预激发振动态可以通过应用位移门来模拟。
# 我们向第6个振动模式插入平均一个振动量子，并计算激发电子态中每个振动模式的平均光子数。

# %%
cir = dq.photonic.QumodeCircuit(
    nmode=modes,
    init_state='vac',
    # init_state=init_state,
    cutoff=cutoff,
    backend='gaussian',
)

cir.d(wires=[5], r=1.0)

for i in range(modes):
    cir.d(wires=[i], r=delta_2[i])

cir.any(cr, wires=list(range(modes)))

for i in range(modes):
    cir.s(wires=[i], r=-lambda_2[i])

cir.any(cl, wires=list(range(modes)))

state = cir()

# 线路可视化
cir.draw()

# %%
sample3 = cir.measure(shots=shots, mcmc=True)

# %%
sample4 = []
for ele in sample3.items():
    # print(ele[0].state)
    sample4.append(ele[0].state * ele[1])
counts4 = np.sum(sample4, axis=0)

plt.figure(figsize=(8, 4))
plt.ylabel('Photon number')
plt.xlabel(r'Frequency (cm$^{-1}$)')
plt.xticks(range(len(omegap)), np.round(omegap, 1), rotation=90)
plt.bar(range(len(omegap)), counts4, color='green')
plt.tight_layout()
plt.show()
