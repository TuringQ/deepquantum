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
# # 高斯玻色采样(Gaussian Boson Sampling)

# %% [markdown]
# ## 数学背景

# %% [markdown]
# <div style="margin-right: 15px; border-radius: 10px; background-color: rgb(255， 255， 255); text-align: center;">
#     <img src="./fig/gbs.png" width="40%"/>
#     <p style="padding: 10px; font-size: small; text-align: center; line-height: 0%;">
#         <b>
#     </p>
# </div>
#
# 高斯玻色采样(GBS)可以作为玻色采样的一种变体，不同之处在于输入的量子态是高斯压缩态而不是离散的Fock态。
# 压缩态是高斯态的一种，高斯态指的是这个量子态对应的Wigner函数是高斯分布，比如相干态。
# 单模压缩态的Wigner函数对应的高斯分布在 $X$，$P$ 两个正交分量上会压缩或者拉伸，单模压缩态可以将压缩门作用到真空态上得到，也可以用下面的Fock态基矢展开[2]，需要注意的是这里的Fock态光子数从0到无穷大取偶数，因此输出的量子态的空间是无限大且只包含偶数光子数的Fock态空间。
#
# <div style="margin-right: 15px; border-radius: 10px; background-color: rgb(255， 255， 255); text-align: center;">
#     <img src="./fig/s.png" width="40%"/>
#     <p style="padding: 10px; font-size: small; text-align: center; line-height: 0%;">
#         <b>
#     </p>
# </div>
#
# GBS采样概率的理论计算和玻色采样类似，不同之处在于对粒子数分辨探测器和阈值探测器两种情况，分别需要用hafnian函数和torotonian函数来计算。

# %% [markdown]
# 1. 粒子数分辨探测器
#
# 在探测端口使用粒子数分辨探测器时对应数学上需要计算hafnian函数，
# 对于 $2m\times 2m$ 对称矩阵 $A$ 的hafnian定义如下[3]，
#
# <div style="margin-right: 15px; border-radius: 10px; background-color: rgb(255， 255， 255); text-align: center;">
#     <img src="./fig/haf.png" width="30%"/>
#     <p style="padding: 10px; font-size: small; text-align: center; line-height: 0%;">
#         <b>
#     </p>
# </div>
#
# 这里PMP表示所有完美匹配排列的集合，当 $n=4$ 时，$PMP(4) = {(0,1)(2,3),(0,2)(1,3),(0,3)(1,2)}$，对应的矩阵 $B$ 对应的hafnian如下
#
# $$
# haf(B) = B_{0,1}B_{2,3}+B_{0,2}B_{1,3} + B_{0,3}B_{1,2}
# $$
#
# 在图论中，hafnian计算了图 $G$ 对应的邻接矩阵A描述的图的完美匹配数(这里图 $G$ 是无权重，无环的无向图)，比如邻接矩阵
# $A =\begin{pmatrix}
# 0&1&1&1\\
# 1&0&1&1\\
# 1&1&0&1\\
# 1&1&1&0
# \end{pmatrix}$，haf(A)=3，对应的完美匹配图如下。
#
# <div style="margin-right: 15px; border-radius: 10px; background-color: rgb(255， 255， 255); text-align: center;">
#     <img src="./fig/f9.png" width="30%"/>
#     <p style="padding: 10px; font-size: small; text-align: center; line-height: 0%;">
#         <b>
#     </p>
# </div>
#
# 当计算的图是二分图时，得到的hafnian计算结果就是permanent。
#
# <div style="margin-right: 15px; border-radius: 10px; background-color: rgb(255， 255， 255); text-align: center;">
#     <img src="./fig/per.png" width="40%"/>
#     <p style="padding: 10px; font-size: small; text-align: center; line-height: 0%;">
#         <b>
#     </p>
# </div>
#
# 因此任何计算hafnian的算法也可以用来计算permanent，同样的计算hafnian也是 $\#P$ 难问题。
#
# 对于粒子数探测器，输出的Fock态 $S = (s_1, s_2,..,s_m)$ 时，对应的 $s_i=0,1,2...$，
# 输出态的概率理论计算如下
#
# <div style="margin-right: 15px; border-radius: 10px; background-color: rgb(255， 255， 255); text-align: center;">
#     <img src="./fig/f5.png" width="30%"/>
#     <p style="padding: 10px; font-size: small; text-align: center; line-height: 0%;">
#         <b>
#     </p>
# </div>
#
# 这里 $Q,A,X$ 的定义如下，
#
# <div style="margin-right: 15px; border-radius: 10px; background-color: rgb(255， 255， 255); text-align: center;">
#     <img src="./fig/f6.png" width="20%"/>
#     <p style="padding: 10px; font-size: small; text-align: center; line-height: 0%;">
#         <b>
#     </p>
# </div>
#
# $Q,A$ 由输出量子态的协方差矩阵 $\Sigma$ 决定 ( $\Sigma$ 描述的是 $a,a^+$ 的协方差矩阵)，子矩阵 $A_s$
# 由输出的Fock态决定，具体来说取矩阵 $A$ 的 $i, i+m$ 行和列并且重复 $s_i$ 次来构造 $A_s$ 。
# 如果 $s_i=0$，那么就不取对应的行和列，如果所有的 $s_i=1$, 那么对应的子矩阵 $A_s = A$。
#
# 考虑高斯态是纯态的时候， 矩阵$A$可以写成直和的形式，$A = B \oplus B^*$, $B$ 是 $m\times m$ 的对称矩阵。这种情况下输出Fock态的概率如下
#
# <div style="margin-right: 15px; border-radius: 10px; background-color: rgb(255， 255， 255); text-align: center;">
#     <img src="./fig/f4.png" width="30%"/>
#     <p style="padding: 10px; font-size: small; text-align: center; line-height: 0%;">
#         <b>
#     </p>
# </div>
#
# 这里的子矩阵 $B_s$ 通过取 $i$ 行和 $i$ 列并且重复 $s_i$ 次来构造，同时这里hafnian函数计算的矩阵维度减半，可以实现概率计算的加速。
#
# 当所有模式输出的光子数 $s_i = 0,1$ 时，对应的 $A_s$ 是A的子矩阵，也对应到邻接矩阵A对应的图 $G$ 的子图，利用这个性质可以解决很多子图相关的问题，比如稠密子图，最大团问题等。

# %% [markdown]
# 2. 阈值探测器
#
# 使用阈值探测器时对应的输出Fock态概率 $S = (s_1, s_2,..,s_m)，s_i\in \{0,1\}$，此时理论概率的计算需要用到Torontonian函数[4]
#
# <div style="margin-right: 15px; border-radius: 10px; background-color: rgb(255， 255， 255); text-align: center;">
#     <img src="./fig/f7.png" width="20%"/>
#     <p style="padding: 10px; font-size: small; text-align: center; line-height: 0%;">
#         <b>
#     </p>
# </div>
# <div style="margin-right: 15px; border-radius: 10px; background-color: rgb(255， 255， 255); text-align: center;">
#     <img src="./fig/f8.png" width="40%"/>
#     <p style="padding: 10px; font-size: small; text-align: center; line-height: 0%;">
#         <b>
#     </p>
# </div>
#
# 这里 $O_s = I-(\Sigma^{-1})_s$，直观上来看对于阈值探测器对应的特定的Fock态输出只需要将粒子数分辨探测器对应的多个Fock态概率求和即可。
#

# %% [markdown]
# ## 代码演示

# %% [markdown]
# 下面简单演示6个模式的高斯玻色采样任务

# %%
import deepquantum as dq
import numpy as np

squeezing = [1] * 6
unitary = np.eye(6)
gbs = dq.photonic.GaussianBosonSampling(nmode=6, squeezing=squeezing, unitary=unitary)
gbs()
gbs.draw()  # 画出采样线路

# %% [markdown]
# 设置粒子数分辨探测器开始采样并输出Fock态结果

# %%
gbs.detector = 'pnrd'
result = gbs.measure(shots=1024, mcmc=True)
print(result)

# %% [markdown]
# 设置阈值探测器开始采样并输出Fock态结果

# %%
gbs.detector = 'threshold'
result = gbs.measure(shots=1024, mcmc=True)
print(result)

# %% [markdown]
# ## 附录

# %% [markdown]
# [1] Lvovsky, Alexander I. "Squeezed light." Photonics: Scientific Foundations, Technology and Applications 1 (2015): 121-163.
#
# [2]Bromley, Thomas R., et al. "Applications of near-term photonic quantum computers: software and algorithms." Quantum Science and Technology 5.3 (2020): 034010.
#
# [3]Quesada, Nicolás, Juan Miguel Arrazola, and Nathan Killoran. "Gaussian boson sampling using threshold detectors." Physical Review A 98.6 (2018): 062322.
#
# [4]J. M. Arrazola and T. R. Bromley, Physical Review Letters 121, 030503 (2018)
