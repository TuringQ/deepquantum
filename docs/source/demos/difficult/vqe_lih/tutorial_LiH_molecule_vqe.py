# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dq_202603
#     language: python
#     name: python3
# ---

# %% [markdown]
# # LiH 基态能量：八模单光子线路与三项纠缠锻造
#
# ## 背景与动机
#
# 分子的基态是它能量最低的量子态。求出不同原子间距下的基态能量，可以帮助我们理解
# 分子的成键与结构变化。这里以 LiH 为例：给定一个描述分子能量的哈密顿量，
# 我们希望找到使能量最低的量子态，并观察能量如何随核间距变化。
#
# 困难在于，描述量子态所需的空间会随系统规模迅速增大。一个 $N$ 比特纯态需要
# $2^N$ 个复振幅来表示；随着分子模型变大，直接保存和求解整个矩阵的经典计算成本也会增加。
# 在量子设备上，制备足够准确的态则需要更多比特和更复杂的线路。
# 本例的六比特模型仍然很小，能够直接对角化，因此适合用来验证一种减少单次线路宽度的方法。
#
# 变分量子本征求解器（VQE）的思路是：先用带参数的线路制备一个候选态，计算它的能量，
# 再由经典优化器调整线路参数，使能量逐步降低。它不要求一开始就知道基态是什么，
# 而是通过反复“制备—评价—调整”寻找低能量态。
#
# 那么，能否只用较小的线路，估计较大系统的能量？
# 文献 [*Doubling the Size of Quantum Simulators by Entanglement Forging*](https://doi.org/10.1103/PRXQuantum.3.010309)
# 提出了纠缠锻造（entanglement forging）：利用 Schmidt 分解，把一个大系统的态表示为
# 两个较小子系统态的乘积之和，分别计算局部测量结果，再在经典计算机上重构整体能量。
# 论文用五比特线路研究了十比特表示的水分子模型。它减少的是单次量子线路所需的比特数，
# 代价是额外的局部态制备、测量和经典组合计算；并不是把原问题的所有计算成本都减半。
#
# 本教程沿用这一思路，研究随附数据定义的**六比特 LiH 有效模型**：
# 把它划分为两个三比特子系统，选取三项乘积态构造变分表示，
# 再使用**一个光子、八个光学模式**实现每个三比特子系统的计算。
# 最后通过一个 $3\times3$ 矩阵求解展开系数，并优化光学线路参数。
# 整个过程在 DeepQuantum 中进行数值模拟，六比特矩阵的直接对角化结果用于检查计算是否正确。
#
# 下面先说明分子哈密顿量如何写成 Pauli 算符的组合，再介绍编码和能量重构方法，
# 最后完成优化并比较参考能量。重点是理解：较小的光学线路提供哪些信息，
# 以及这些信息如何组合成六比特模型的变分能量。

# %% [markdown]
# ## 1. 环境、数据路径与实验配置
#
# 需要 Python ≥ 3.10、DeepQuantum 及其依赖、Matplotlib。建议在已经安装 DeepQuantum 的环境运行。
# 下方优先使用当前仓库的 `src/`，并打印实际导入路径，便于区分源码与其他已安装版本。
# 可以从本 notebook 所在目录或仓库根目录启动；其他位置请显式填写 `DATA_DIR`。
#
# 数据文件来自随教程分发的 `data_lih/LiH_data_*.pkl`。只加载可信的 pickle 文件。
# 文件仅提供 `pauli_list`、`constant` 和 `fci_energy`，没有说明距离单位、基组、活性空间、
# 冻结轨道或费米子到量子比特映射。**这些信息需要向数据生成者确认，不能从文件名推断。**
# 因此图中的横轴暂记为“数据标签 R（原始单位未注明）”，不标注 Å、nm 或 bohr。
# 本教程的结论限定为随附六比特有效模型。

# %%
import json
import pickle
import sys
from datetime import datetime
from functools import reduce
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

DATA_DIR = None  # 可设为包含 LiH_data_0.1.pkl 等文件的绝对路径
cwd = Path.cwd().resolve()
repo_root = next((p for p in (cwd, *cwd.parents) if (p / 'src/deepquantum').is_dir()), None)
if repo_root is not None:
    sys.path.insert(0, str(repo_root / 'src'))
import deepquantum as dq  # noqa: E402

if DATA_DIR is None:
    candidates = [cwd / 'data_lih']
    if repo_root is not None:
        candidates.append(repo_root / 'docs/source/demos/difficult/vqe_lih/data_lih')
    DATA_DIR = next((p for p in candidates if (p / 'LiH_data_0.1.pkl').is_file()), None)
    if DATA_DIR is None:
        raise FileNotFoundError('请将 DATA_DIR 设置为本教程 data_lih 目录的绝对路径')
DATA_DIR = Path(DATA_DIR).resolve()

SEED = 12
DTYPE = torch.float64
CDTYPE = torch.complex128
DEVICE = torch.device('cpu')  # 本教程以 CPU 双精度为验证基线
EPOCHS = 150
LEARNING_RATE = 0.02
R_VALUES = [round(0.1 * i, 1) for i in range(1, 10)]
BASIS_INDICES = (6, 5, 3)  # 二进制 110、101、011，最左侧为逻辑 qubit 0
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUTPUT_DIR = DATA_DIR.parent / 'outputs_lih' / RUN_ID

torch.manual_seed(SEED)
np.random.seed(SEED)
print('DeepQuantum:', dq.__file__)
print('PyTorch:', torch.__version__)
print('数据目录:', DATA_DIR)
print('输出目录:', OUTPUT_DIR)

# %% [markdown]
# ### 哈密顿量的 Pauli 分解与参考能量
#
# **哈密顿量告诉我们如何计算能量。** 对归一化量子态 $|\psi\rangle$，能量期望值为
#
# $$E(\psi)=\langle\psi|H_{\mathrm{eff}}|\psi\rangle.$$
#
# 寻找基态，就是在允许的量子态中寻找这个期望值的最小值。
# 本例采用六比特模型，因此 $|\psi\rangle$ 有 64 个振幅，$H_{\mathrm{eff}}$ 是一个
# $64\times64$ 的厄米矩阵。直接对角化这个小矩阵可以得到最低特征值，供后面的变分结果比较。
#
# #### 1）为什么要在 Pauli 基底下分解？
#
# 一个大的哈密顿量通常不方便作为一个整体去测量，但可以拆成许多容易处理的算符项。
# 单比特的 $I,X,Y,Z$ 构成一组算符基底；对六比特，把六个这样的算符做张量积，
# 就得到六比特的 **Pauli 字符串**。这里说的是“算符空间的基底”，
# 与 $|000000\rangle$ 这样的“量子态基矢”要区分开。
#
# 例如
#
# $$P_k=I\otimes I\otimes I\otimes Z\otimes Z\otimes Z,$$
#
# 可以简写为 `IIIZZZ`。从左到右分别对应六个逻辑比特：前三个取恒等算符，
# 后三个测量 Z 算符的乘积。若在计算基态 $|b_0b_1b_2b_3b_4b_5\rangle$ 上测量，
# 这一项的值就是 $(-1)^{b_3+b_4+b_5}$。
#
# 六比特算符可以写成这些 Pauli 字符串的线性组合：
#
# $$H_{\mathrm{P}}=\sum_{k=1}^{K}w_kP_k,\qquad
# P_k\in\{I,X,Y,Z\}^{\otimes6}.$$
#
# $P_k$ 指定测什么，$w_k$ 指定这一项对能量的权重。厄米哈密顿量在这个基底下的系数为实数。
# 完整的六比特 Pauli 算符基底包含 $4^6$ 项，但具体哈密顿量中很多系数为零；
# 随附数据每份列出 62 项。**这 62 项是能量算符的分解项，不是 62 个量子态。**
#
# 本教程直接读取已经完成的 Pauli 分解，不在这里重新进行电子结构计算或费米子映射。
# `pauli_list` 的每一行包含六个 Pauli 字母以及最后的权重，例如
# `['I', 'I', 'I', 'Z', 'Z', 'Z', w]` 就代表 $w\,IIIZZZ$。
#
# #### 2）怎样由各项测量结果得到能量？
#
# 期望值对算符是线性的，因此只要分别得到各 Pauli 字符串的期望值，就可以加权相加：
#
# $$\langle H_{\mathrm{P}}\rangle=\sum_{k=1}^{K}w_k\langle P_k\rangle.$$
#
# 数据还单独提供一个与量子态无关的能量偏移 `constant`，记为 $E_{\mathrm{offset}}$。
# 本教程严格按数据约定使用
#
# $$H_{\mathrm{eff}}=H_{\mathrm{P}}+E_{\mathrm{offset}}I_{64},\qquad
# E(\psi)=\sum_k w_k\langle P_k\rangle+E_{\mathrm{offset}}.$$
#
# Pauli 列表中本身也包含 `IIIIII`，它的期望值恒为 1，其系数已经计入上面的求和。
# 外部 `constant` 是另一个单独保存的偏移，只再加一次；不要把 `IIIIII` 的系数重复加进去。
# 常数偏移改变能量的零点，但不改变哪个量子态的能量最低。
# 它具体包含哪些物理贡献，需要数据生成过程说明，本教程不把它直接认定为核排斥能。
#
# 接下来划分子系统时，可以把每项写成
#
# $$P_k=A_k\otimes B_k,$$
#
# 其中 $A_k$ 取字符串的前三个字母，$B_k$ 取后三个字母。
# 例如 `IIIZZZ` 分为 $A_k=III$ 和 $B_k=ZZZ$。
# 这正是后面用三比特子系统信息重构六比特能量的入口；对于纠缠态，还需要计算不同展开项之间的非对角贡献。
#
# #### 3）参考能量用来比较什么？
#
# 文件字段 `fci_energy` 保存一个参考值。本教程把它记为 $E_{\mathrm{ref}}$，
# 并通过直接对角化检查：
#
# $$E_{\mathrm{ref}}=\lambda_{\min}(H_{\mathrm{P}})+E_{\mathrm{offset}}.$$
#
# 下方代码会对九份数据逐一检查这个等式。因此，后文比较的是
# **同一份六比特有效哈密顿量的变分能量与精确最低能量**。
# FCI 通常指在选定轨道空间中的全组态相互作用计算，但当前文件没有提供其轨道空间等元数据，
# 不能仅凭字段名认定它是未截断分子的完整基态能量。
#
# 在无采样误差、重构准确的情况下，归一化变分态的能量满足
#
# $$E_{\mathrm{VQE}}\ge E_{\mathrm{ref}}.$$
#
# 我们用 $\Delta E=E_{\mathrm{VQE}}-E_{\mathrm{ref}}$ 衡量结果离参考值还有多远。
# 误差可能来自所选变分态表达能力有限，也可能来自优化尚未找到足够好的参数。

# %%
PAULI = {
    'I': torch.eye(2, dtype=CDTYPE, device=DEVICE),
    'X': torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE, device=DEVICE),
    'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE, device=DEVICE),
    'Z': torch.diag(torch.tensor([1, -1], dtype=CDTYPE, device=DEVICE)),
}


def load_problem(r):
    with (DATA_DIR / f'LiH_data_{r:.1f}.pkl').open('rb') as stream:
        raw = pickle.load(stream)
    strings, weights = [], []
    for row in raw['pauli_list']:
        if len(row) != 7 or any(p not in PAULI for p in row[:6]):
            raise ValueError('每项应为六个 Pauli 字母和一个系数')
        weight = complex(row[-1])
        if not np.isfinite(weight) or abs(weight.imag) > 1e-12:
            raise ValueError('厄米 Pauli 展开的系数应为有限实数')
        strings.append(''.join(row[:6]))
        weights.append(weight.real)
    if not strings:
        raise ValueError('Pauli 列表不能为空')
    constant, reference = float(raw['constant']), float(raw['fci_energy'])
    if not np.isfinite([constant, reference]).all():
        raise ValueError('能量数据必须有限')
    weights = torch.tensor(weights, dtype=DTYPE, device=DEVICE)
    hamiltonian = sum(w * reduce(torch.kron, [PAULI[p] for p in s]) for w, s in zip(weights, strings, strict=True))
    exact = torch.linalg.eigvalsh(hamiltonian)[0].item() + constant
    if not np.isclose(exact, reference, atol=1e-9, rtol=0):
        raise ValueError('数据参考能量与六比特矩阵的最低能量不一致')
    return {
        'r': r,
        'strings': strings,
        'weights': weights,
        'constant': constant,
        'reference': reference,
        'hamiltonian': hamiltonian,
    }


problems = {r: load_problem(r) for r in R_VALUES}
print('九份数据的直接对角化校验通过')
print('R 标签     Pauli 项数     参考能量 / Hartree')
for r, problem in problems.items():
    print(f'{r:4.1f}        {len(problem["strings"]):3d}          {problem["reference"]: .9f}')

# %% [markdown]
# ## 2. 编码与变分表示：方法概述
#
# 整个方法可以分为三步：**把六比特态写成两个三比特态的组合；用局部测量构造能量矩阵；
# 求解小矩阵并调整线路参数。** 先看这种表示为什么可行，再说明光学线路如何实现它。
#
# ### 1）从 Schmidt 分解到三项变分表示
#
# 将六个逻辑比特划分为 A、B 两部分，每部分三个比特。
# Schmidt 分解告诉我们，一个六比特纯态可以写成
#
# $$|\psi_6\rangle=\sum_{i=0}^{r-1}\lambda_i
# \bigl(U|b_i\rangle\bigr)\otimes\bigl(V|b_i\rangle\bigr),
# \qquad \sum_i\lambda_i^2=1.$$
#
# 这里，$|b_i\rangle$ 是三比特计算基态，$U,V$ 分别把它们变成两个子系统的一组正交态，
# $\lambda_i$ 表示每一对态在整体态中的权重，可以取非负实数；$r$ 最多为 8。
# 直观地说，整体态由若干对三比特态组合而成，两个子系统之间的关联保存在这些配对和权重中。
# 我们并没有预先知道基态的 Schmidt 分解，而是借用这种形式构造可优化的候选态。
#
# 为了降低本例的参数和测量数量，只选择三个输入基态：
#
# $$|b_0\rangle=|110\rangle,\qquad |b_1\rangle=|101\rangle,\qquad |b_2\rangle=|011\rangle.$$
#
# 并让两个子系统使用同一个参数化线路，即 $U=V=U(\theta)$。
# 与代码一致，展开系数允许为复数，记为 $c_i$：
#
# $$|\Phi(\theta,c)\rangle=\sum_{i=0}^{2}c_i
# \bigl(U(\theta)|b_i\rangle\bigr)\otimes\bigl(U(\theta)|b_i\rangle\bigr),
# \qquad \sum_i|c_i|^2=1.$$
#
# 这就是本教程实际优化的态。$\theta$ 控制线路，$c_i$ 控制三个乘积态的组合。
# 为避免与通常取非负实数的 Schmidt 系数混淆，下面称 $c_i$ 为“展开系数”。
# 只取三项且共享 $U$ 都是本例的限制，因此这种形式不保证能够精确表示目标基态。
#
# ### 2）用八个光学模式表示三个逻辑比特
#
# 三个逻辑比特有 $2^3=8$ 个计算基态。一个光子分布在八个模式中，也有八个基态位置可选，
# 因此可以建立一一对应：
#
# $$|b_0b_1b_2\rangle\ \longleftrightarrow\ |0,\ldots,1_j,\ldots,0\rangle,
# \qquad j=4b_0+2b_1+b_2.$$
#
# 右侧是光学 Fock 态，表示只有第 $j$ 个模式中有一个光子；模式编号从 0 开始。
# 例如 $|110\rangle$ 对应模式 6 中的单光子，即 $|0,0,0,0,0,0,1,0\rangle$。
# 左侧两个数字 1 是逻辑比特取值，不表示有两个光子。
#
# 这种一元／模式编码使我们可以用八模 Clements 干涉线路实现三比特空间中的酉变换 $U(\theta)$。
# 分别制备所需的输入态、施加线路和测量旋转，就能计算后续所需的局部算符信息。
# 本例没有同时制备一个物理六比特纠缠态，而是在经典计算机上组合两部分的测量结果。
#
# 需要区分逻辑比特与光学模式：编码 $q$ 个逻辑比特需要 $2^q$ 个模式。
# 因此，“使用三比特子系统”在这里对应八模单光子线路，并不意味着只需要三个光学模式。
#
# ### 3）把整体能量写成一个小矩阵
#
# 前面已经将哈密顿量的每一项分为 $P_k=A_k\otimes B_k$。
# 将三项变分态代入期望值，可得到
#
# $$\langle P_k\rangle=\sum_{i,j=0}^{2}c_i^*c_j
# \langle b_i|U^\dagger A_kU|b_j\rangle
# \langle b_i|U^\dagger B_kU|b_j\rangle.$$
#
# 把后面的两项乘积记为一个 $3\times3$ 矩阵的元素：
#
# $$(M_k)_{ij}=\langle b_i|U^\dagger A_kU|b_j\rangle
# \langle b_i|U^\dagger B_kU|b_j\rangle.$$
#
# 于是 $\langle P_k\rangle=c^\dagger M_kc$。
# 对所有 Pauli 项按权重求和，定义
#
# $$M=\sum_k w_kM_k,\qquad E(\theta,c)=c^\dagger M(\theta)c+E_{\mathrm{offset}}.$$
#
# 按 $|b_0\rangle=|110\rangle$、$|b_1\rangle=|101\rangle$、$|b_2\rangle=|011\rangle$
# 的顺序，矩阵和系数向量具体写成
#
# $$
# M=\begin{pmatrix}
# M_{00}&M_{01}&M_{02}\\
# M_{10}&M_{11}&M_{12}\\
# M_{20}&M_{21}&M_{22}
# \end{pmatrix}
# =\begin{pmatrix}
# M_{00}&M_{01}&M_{02}\\
# M_{01}^{*}&M_{11}&M_{12}\\
# M_{02}^{*}&M_{12}^{*}&M_{22}
# \end{pmatrix},\qquad
# c=\begin{pmatrix}c_0\\c_1\\c_2\end{pmatrix}.
# $$
#
# 这里每一行、每一列对应的是一个**六比特乘积态**
# $|\phi_i\rangle=U|b_i\rangle\otimes U|b_i\rangle$，而不是一个光学模式。
# 例如
#
# $$M_{00}=\sum_k w_k
# \langle110|U^\dagger A_kU|110\rangle
# \langle110|U^\dagger B_kU|110\rangle,$$
#
# $$M_{01}=\sum_k w_k
# \langle110|U^\dagger A_kU|101\rangle
# \langle110|U^\dagger B_kU|101\rangle.$$
#
# 其余元素同理。对角元 $M_{00},M_{11},M_{22}$ 为实数；
# 只需重构上三角的 $M_{01},M_{02},M_{12}$，下三角由复共轭确定。
# 单个 Pauli 项的 $M_k$ 也有同样的矩阵结构，再按 $w_k$ 加权得到 $M$。
#
# 将矩阵乘法展开，可以更清楚地看到对角项与非对角项的作用：
#
# $$
# E(\theta,c)=\sum_{i=0}^{2}|c_i|^2M_{ii}
# +2\operatorname{Re}\!\left(c_0^*c_1M_{01}+c_0^*c_2M_{02}+c_1^*c_2M_{12}\right)
# +E_{\mathrm{offset}}.
# $$
#
# 只测三个对角元就会漏掉相干贡献，因此还需要下面介绍的叠加态制备和测量。
# $M$ 本身不包含外部常数偏移；若希望把偏移也写入矩阵，应使用
# $M+E_{\mathrm{offset}}I_3$，此时就不能在结果中再加一次偏移。
#
# $M$ 可以理解为：在当前选定的三个乘积态之间，哈密顿量如何作用。
# 它的对角元对应各乘积态的能量贡献，非对角元记录不同项之间的相干贡献，不能省略。
# 由于哈密顿量厄米，$M$ 也是厄米矩阵，即 $M_{ji}=M_{ij}^*$。
# $\dagger$ 表示共轭转置；因为代码允许复系数，这里不能用普通转置代替。
#
# ### 4）先求最优系数，再优化线路
#
# 固定 $\theta$ 后，三个乘积态彼此正交，归一化条件就是 $c^\dagger c=1$。
# 在这个条件下，使能量最低的 $c$ 是 $M$ 的最低特征值对应的归一化特征向量，因此
#
# $$E(\theta)=\lambda_{\min}\bigl(M(\theta)\bigr)+E_{\mathrm{offset}}.$$
#
# 也就是说，**固定线路时，不必用梯度逐个调整展开系数，只需求解一个 $3\times3$ 矩阵。**
# 然后再调整线路参数 $\theta$，重新计算 $M$，让这个最低能量继续下降。
# 代码中 `torch.linalg.eigh` 完成系数求解，Adam 完成线路参数更新。
#
# 每一步得到的是当前三维子空间中的最低能量，而不是自动得到完整六比特模型的基态能量。
# 后面会将它与 $E_{\mathrm{ref}}$ 比较。下一节具体说明如何通过三比特测量得到 $M$ 的元素，
# 尤其是不能直接当作普通期望值测量的非对角元。

# %% [markdown]
# ## 3. 从局部测量重构投影矩阵
#
# 上一节把能量计算转化为求 $M$。接下来要解决的是：怎样用三比特子系统的测量结果，
# 填出这个矩阵的三个对角元和三个独立非对角元？
#
# 对每个 Pauli 项 $P_k=A_k\otimes B_k$，分别计算两部分的局部矩阵元，再相乘、加权：
#
# $$M_{ij}=\sum_k w_k\,
# \langle b_i|U^\dagger A_kU|b_j\rangle
# \langle b_i|U^\dagger B_kU|b_j\rangle.$$
#
# 下面先用 $O$ 代表任意一个局部三比特 Pauli 算符，它可以是 $A_k$，也可以是 $B_k$。
# 记 $a_{ij}=\langle b_i|U^\dagger O U|b_j\rangle$。
#
# ### 1）对角元：分别制备三个基态
#
# 制备 $|b_i\rangle$，经过线路 $U$ 后测量 $O$，得到的就是
#
# $$a_{ii}=\langle b_i|U^\dagger O U|b_i\rangle.$$
#
# 例如制备 $|110\rangle$ 后，对子系统 A 测量 $A_k$、对子系统 B 测量 $B_k$，
# 两次期望值的乘积给出 $(M_k)_{00}$。对全部 Pauli 项加权求和，就得到 $M_{00}$。
# $M_{11}$ 和 $M_{22}$ 分别使用 $|101\rangle$ 和 $|011\rangle$，方法相同。
#
# ### 2）非对角元：用两个叠加态提取实部和虚部
#
# $a_{ij}$（$i\ne j$）的左右态不同，不是某一个态上的普通期望值。
# 为了将它转为可测量的量，对每一对 $i<j$ 制备
#
# $$|R_{ij}\rangle=\frac{|b_i\rangle+|b_j\rangle}{\sqrt2},\qquad
# |I_{ij}\rangle=\frac{|b_i\rangle+i|b_j\rangle}{\sqrt2}.$$
#
# 这两个态只差一个相对相位。将它们分别经过同一个 $U$，测量同一个算符 $O$，记结果为
#
# $$O_{RR}=\langle R_{ij}|U^\dagger O U|R_{ij}\rangle,\qquad
# O_{II}=\langle I_{ij}|U^\dagger O U|I_{ij}\rangle.$$
#
# 这里的 $RR$、$II$ 是叠加态的标签；$O_{II}$ 不是测量恒等算符。
# 利用 $a_{ji}=a_{ij}^*$ 展开，可以得到
#
# $$O_{RR}=\frac{a_{ii}+a_{jj}}2+\operatorname{Re}(a_{ij}),\qquad
# O_{II}=\frac{a_{ii}+a_{jj}}2-\operatorname{Im}(a_{ij}).$$
#
# 因此
#
# $$\boxed{
# a_{ij}=\left(O_{RR}-\frac{a_{ii}+a_{jj}}2\right)
# +i\left(\frac{a_{ii}+a_{jj}}2-O_{II}\right).}$$
#
# 也就是先减去两个对角元的平均贡献，分别得到实部和虚部。
# 例如，若四个测量结果为 $a_{ii}=-0.2$、$a_{jj}=0.4$、$O_{RR}=0.35$、$O_{II}=-0.15$，
# 则对角平均值为 0.1，重构结果为 $a_{ij}=0.25+0.25i$。
# 这只是说明计算步骤的数值例子，不是本次 LiH 运行的实测数据。
#
# 下面通过三组例子，具体说明如何制备这些叠加态。
#
# ### 3）制备叠加态所用的门
#
# 逻辑比特按从左到右编号为 0、1、2。$H(j)$、$S(j)$ 表示作用在第 $j$ 个比特上；
# $\mathrm{SWAP}(j,l)$ 交换两个比特；$\mathrm{CNOT}(j\to l)$ 的第一个比特是控制位，第二个是目标位。
# 下方箭头按实际作用的先后顺序排列；若写成矩阵乘积，最先作用的门在最右侧。
#
# $$H=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\qquad
# S=\begin{pmatrix}1&0\\0&i\end{pmatrix},$$
#
# $$\mathrm{SWAP}=\begin{pmatrix}
# 1&0&0&0\\0&0&1&0\\0&1&0&0\\0&0&0&1
# \end{pmatrix},\qquad
# \mathrm{CNOT}=\begin{pmatrix}
# 1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0
# \end{pmatrix}.$$
#
# 两比特矩阵按 $|00\rangle,|01\rangle,|10\rangle,|11\rangle$ 排序。
# 下面的门是三逻辑比特空间中描述制备过程的一种方式；映射到八模单光子编码后，
# 对应八维空间中的模式变换，不要求在光学芯片上另外实现物理双比特 CNOT 门。
#
# ### 例一：$|110\rangle$ 与 $|101\rangle$，用于重构 $M_{01}$
#
# 以 $|110\rangle$ 为初态，目标是
#
# $$|R_{01}\rangle=\frac{|110\rangle+|101\rangle}{\sqrt2},\qquad
# |I_{01}\rangle=\frac{|110\rangle+i|101\rangle}{\sqrt2}.$$
#
# 先制备实叠加态：
#
# $$
# \begin{aligned}
# |110\rangle
# &\xrightarrow{H(2)}\frac{|110\rangle+|111\rangle}{\sqrt2}\\
# &\xrightarrow{\mathrm{SWAP}(1,2)}\frac{|101\rangle+|111\rangle}{\sqrt2}\\
# &\xrightarrow{\mathrm{CNOT}(1\to2)}\frac{|101\rangle+|110\rangle}{\sqrt2}
# =|R_{01}\rangle.
# \end{aligned}
# $$
#
# 如果要制备虚叠加态，再作用 $S(2)$。$|110\rangle$ 的第 2 位是 0，保持不变；
# $|101\rangle$ 的第 2 位是 1，乘上 $i$：
#
# $$|R_{01}\rangle\xrightarrow{S(2)}
# \frac{|110\rangle+i|101\rangle}{\sqrt2}=|I_{01}\rangle.$$
#
# 在八模编码中，这两个态分别对应光子在模式 6 和 5 之间的等权叠加，相对相位分别为 0 和 $\pi/2$。
# 将这两个输入分别经过 $U$ 并测量 $O$，结合已测的 $a_{00},a_{11}$，即可重构 $a_{01}$。
#
# ### 例二：$|110\rangle$ 与 $|011\rangle$，用于重构 $M_{02}$
#
# 仍从 $|110\rangle$ 出发，目标是
#
# $$|R_{02}\rangle=\frac{|110\rangle+|011\rangle}{\sqrt2},\qquad
# |I_{02}\rangle=\frac{|110\rangle+i|011\rangle}{\sqrt2}.$$
#
# 前三步与例一相同，再交换第 0、1 位：
#
# $$
# \begin{aligned}
# |110\rangle
# &\xrightarrow{H(2)}\frac{|110\rangle+|111\rangle}{\sqrt2}\\
# &\xrightarrow{\mathrm{SWAP}(1,2)}\frac{|101\rangle+|111\rangle}{\sqrt2}\\
# &\xrightarrow{\mathrm{CNOT}(1\to2)}\frac{|101\rangle+|110\rangle}{\sqrt2}\\
# &\xrightarrow{\mathrm{SWAP}(0,1)}\frac{|011\rangle+|110\rangle}{\sqrt2}
# =|R_{02}\rangle.
# \end{aligned}
# $$
#
# 随后使用 $S(2)$：$|110\rangle$ 的末位为 0，$|011\rangle$ 的末位为 1，所以
#
# $$|R_{02}\rangle\xrightarrow{S(2)}
# \frac{|110\rangle+i|011\rangle}{\sqrt2}=|I_{02}\rangle.$$
#
# 在八模编码中，它们对应模式 6 和 3 之间的相干叠加。
# 结合这两种输入的测量结果与 $a_{00},a_{22}$，可以重构 $a_{02}$。
#
# ### 例三：$|101\rangle$ 与 $|011\rangle$，用于重构 $M_{12}$
#
# 改用 $|101\rangle$ 为初态，目标是
#
# $$|R_{12}\rangle=\frac{|101\rangle+|011\rangle}{\sqrt2},\qquad
# |I_{12}\rangle=\frac{|101\rangle+i|011\rangle}{\sqrt2}.$$
#
# 制备步骤为
#
# $$
# \begin{aligned}
# |101\rangle
# &\xrightarrow{H(1)}\frac{|101\rangle+|111\rangle}{\sqrt2}\\
# &\xrightarrow{\mathrm{SWAP}(0,1)}\frac{|011\rangle+|111\rangle}{\sqrt2}\\
# &\xrightarrow{\mathrm{CNOT}(0\to1)}\frac{|011\rangle+|101\rangle}{\sqrt2}
# =|R_{12}\rangle.
# \end{aligned}
# $$
#
# 这次使用 $S(1)$，因为两个目标基态的第 1 位分别为 0 和 1：
#
# $$|R_{12}\rangle\xrightarrow{S(1)}
# \frac{|101\rangle+i|011\rangle}{\sqrt2}=|I_{12}\rangle.$$
#
# 它们对应模式 5 和 3 之间的相干叠加。
# 结合两种输入的测量结果与 $a_{11},a_{22}$，即可重构 $a_{12}$。
#
# ### 4）把局部结果填回 $M$，并与代码对应
#
# 对每个 $k$，分别令 $O=A_k$ 和 $O=B_k$，按照以上步骤得到局部矩阵元
# $a_{ij}^{(A,k)}$ 和 $a_{ij}^{(B,k)}$。随后计算
#
# $$(M_k)_{ij}=a_{ij}^{(A,k)}a_{ij}^{(B,k)},\qquad
# M_{ij}=\sum_k w_k(M_k)_{ij}.$$
#
# 这里两部分采用同样的 $i,j$ 顺序，直接相乘，不对其中一项额外取共轭。
# 重构出 $M_{01},M_{02},M_{12}$ 后，再用 $M_{ji}=M_{ij}^*$ 填充下三角。
#
# 对于任意一个局部算符，三个基态给出对角项，三对实、虚叠加态给出非对角项，
# 因此总共需要 **9 种输入制备**。下方代码中的 `PREPARATIONS` 按以下顺序存放它们：
#
# | 列编号 | 输入态 | 在重构中的用途 |
# |---|---|---|
# | 0、1、2 | $\lvert 110\rangle,\lvert 101\rangle,\lvert 011\rangle$ | 对角项 |
# | 3、4、5 | $\lvert R_{01}\rangle,\lvert R_{02}\rangle,\lvert R_{12}\rangle$ | 非对角项的实部 |
# | 6、7、8 | $\lvert I_{01}\rangle,\lvert I_{02}\rangle,\lvert I_{12}\rangle$ | 非对角项的虚部 |
#
# 代码直接构造这些门序列最终得到的归一化向量，避免在主计算中重复搭建制备门。
# 它与上面三组例子制备的是同样的输入态。实际光学实现仍需要相应的相干制备，
# 其制备变换也可与后续 $U$ 合成为一个八模酉变换。
#
# 输入准备好之后，还需要旋转测量基：X 使用 $H$，Y 使用 $HS^\dagger$
# （先 $S^\dagger$ 后 $H$），Z 不旋转，I 则不计入奇偶权重。
# 例如测量三比特算符 `XYZ`，使用 $H\otimes(HS^\dagger)\otimes I$ 旋转后，
# 按三个输出逻辑位的奇偶性赋予 $+1$ 或 $-1$，加权概率即可得到期望值。
# 这里测量 Y 用的是 $S^\dagger$；前面制备带 $+i$ 相位的输入态用的是 $S$，两者用途不同。

# %%
PAIRS = ((0, 1), (0, 2), (1, 2))
BASIS = torch.eye(8, dtype=CDTYPE, device=DEVICE)[:, list(BASIS_INDICES)]
PREPARATIONS = torch.stack(
    [BASIS[:, i] for i in range(3)]
    + [(BASIS[:, i] + BASIS[:, j]) / np.sqrt(2) for i, j in PAIRS]
    + [(BASIS[:, i] + 1j * BASIS[:, j]) / np.sqrt(2) for i, j in PAIRS],
    dim=1,
)  # (8 个模式, 9 种制备)
assert torch.allclose((PREPARATIONS.abs() ** 2).sum(0), torch.ones(9, dtype=DTYPE, device=DEVICE))

H_GATE = torch.tensor([[1, 1], [1, -1]], dtype=CDTYPE, device=DEVICE) / np.sqrt(2)
S_DAGGER = torch.diag(torch.tensor([1, -1j], dtype=CDTYPE, device=DEVICE))
ROTATIONS = {'I': PAULI['I'], 'Z': PAULI['I'], 'X': H_GATE, 'Y': H_GATE @ S_DAGGER}
LOGICAL_BITS = torch.tensor(list(product((0, 1), repeat=3)), device=DEVICE)


def measurement_setting(pauli_string):
    if len(pauli_string) != 3 or any(p not in PAULI for p in pauli_string):
        raise ValueError('局部算符必须是三个 I/X/Y/Z 字母')
    rotation = reduce(torch.kron, [ROTATIONS[p] for p in pauli_string])
    mask = torch.tensor([p != 'I' for p in pauli_string], device=DEVICE)
    signs = (1 - 2 * ((LOGICAL_BITS * mask).sum(-1) % 2)).to(DTYPE)
    return rotation, signs


def measurement_tables(strings):
    settings = [[measurement_setting(s[:3]), measurement_setting(s[3:])] for s in strings]
    rotations = torch.stack([torch.stack([item[0] for item in pair]) for pair in settings])
    signs = torch.stack([torch.stack([item[1] for item in pair]) for pair in settings])
    return rotations, signs


print('输入制备:', tuple(PREPARATIONS.shape))
rotations, signs = measurement_tables(problems[0.1]['strings'])
print('测量旋转:', tuple(rotations.shape), '奇偶权重:', tuple(signs.shape))


# %% [markdown]
# ### 精确概率、有限采样与光损耗
#
# 默认 `shots=None`，直接使用数值模拟得到的概率，梯度由 PyTorch 自动微分。
# 有限采样时，`sampling='multinomial'` 生成离散样本用于评估，不能通过这些离散样本直接反向传播。
# `sampling='straight_through'` 则使用 Gumbel-softmax 直通估计，前向为离散样本，
# 反向是有偏的连续松弛梯度；它不是硬件测量的真实可微接口，主训练不使用它。
#
# `loss_db` 在各输出模式上施加相同的光损耗，概率包含环境模式。
# 只保留系统模式并归一化后，得到的是**光子存活条件下**的期望值。
# 均匀损耗在精确条件概率中抵消，但会减少有限入射 shots 的存活样本，增大统计误差。
# 没有存活样本的测量设置会明确报错，不用零值伪装成有效期望值。

# %%
def measured_probabilities(probs, shots=None, sampling='multinomial', temperature=0.1):
    if shots is None:
        return probs
    if type(shots) is not int or shots < 1:
        raise ValueError('shots 必须是正整数或 None')
    flat = probs.reshape(-1, probs.shape[-1])
    if sampling == 'multinomial':
        draws = torch.multinomial(flat.detach(), shots, replacement=True)
        counts = torch.zeros_like(flat).scatter_add_(1, draws, torch.ones_like(draws, dtype=flat.dtype))
        result = counts / shots
    elif sampling == 'straight_through':
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature 必须为正有限数')
        logits = flat.clamp_min(torch.finfo(flat.dtype).tiny).log()
        # 分块控制内存；没有改变 shots 定义。
        result = torch.zeros_like(flat)
        for start in range(0, shots, 128):
            size = min(128, shots - start)
            result = (
                result
                + functional.gumbel_softmax(
                    logits[:, None, :].expand(-1, size, -1),
                    tau=temperature,
                    hard=True,
                    dim=-1,
                ).sum(1)
                / shots
            )
    else:
        raise ValueError('sampling 必须为 multinomial 或 straight_through')
    return result.reshape(probs.shape)


class PhotonicProjectedEnergy(nn.Module):
    """固定的八模、三项共享 U Ansatz；每次只构造一次线路酉矩阵。"""

    def __init__(self, problem, loss_db=None):
        super().__init__()
        if loss_db is not None and (not np.isfinite(loss_db) or loss_db < 0):
            raise ValueError('loss_db 必须为非负有限数或 None')
        self.loss_db = loss_db
        self.constant = problem['constant']
        self.circuit = dq.Clements(nmode=8, init_state=[1] + [0] * 7, cutoff=2)
        if loss_db is not None:
            for wire in range(8):
                self.circuit.loss_db(wire, torch.tensor(loss_db, dtype=DTYPE, device=DEVICE))
        self.circuit.to(dtype=DTYPE, device=DEVICE)
        rotations, signs = measurement_tables(problem['strings'])
        self.register_buffer('rotations', rotations)
        self.register_buffer('signs', signs)
        self.register_buffer('preparations', PREPARATIONS.clone())
        self.register_buffer('weights', problem['weights'].clone())
        self.n_parameters = self.circuit.ndata

    def forward(self, theta, shots=None, sampling='multinomial', temperature=0.1):
        if theta.ndim != 1 or theta.numel() != self.n_parameters:
            raise ValueError(f'theta 应为长度 {self.n_parameters} 的一维张量')
        if theta.device != self.preparations.device or theta.dtype != self.weights.dtype:
            raise ValueError('theta 必须与模型的 device/dtype 一致')
        unitary = self.circuit(data=theta)
        n_total = unitary.shape[-1]
        inputs = functional.pad(self.preparations, (0, 0, 0, n_total - 8))
        states = unitary @ inputs  # (系统+环境模式, 9 种制备)
        system = torch.einsum('taoi,ip->tapo', self.rotations, states[:8])
        probabilities = system.abs().square()  # (Pauli 项, 子系统, 制备, 系统模式)
        if n_total > 8:
            environment = states[8:].abs().square().T
            environment = environment[None, None].expand(*probabilities.shape[:2], -1, -1)
            probabilities = torch.cat((probabilities, environment), dim=-1)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
        observed = measured_probabilities(probabilities, shots, sampling, temperature)
        surviving = observed[..., :8]
        survival = surviving.sum(-1)
        if torch.any(survival <= 0):
            raise RuntimeError('部分测量设置没有存活光子；增加 shots 或降低损耗')
        expectations = ((surviving / survival[..., None]) * self.signs[:, :, None, :]).sum(-1)
        diagonal_local = expectations[..., :3]
        diagonal = (diagonal_local.prod(dim=1) * self.weights[:, None]).sum(0)
        average_diagonal = torch.stack(
            [(diagonal_local[..., i] + diagonal_local[..., j]) / 2 for i, j in PAIRS],
            dim=-1,
        )
        off_local = expectations[..., 3:6] - average_diagonal + 1j * (average_diagonal - expectations[..., 6:9])
        off_diagonal = (off_local.prod(dim=1) * self.weights[:, None]).sum(0)
        matrix = torch.diag(diagonal.to(off_diagonal.dtype))
        rows, columns = (0, 0, 1), (1, 2, 2)
        matrix[rows, columns] = off_diagonal
        matrix[columns, rows] = off_diagonal.conj()
        values, vectors = torch.linalg.eigh(matrix)
        return {
            'energy': values[0] + self.constant,
            'matrix': matrix,
            'coefficients': vectors[:, 0],
            'eigenvalues': values + self.constant,
            'survival': survival,
            'expectations': expectations,
        }


# %% [markdown]
# ## 4. 数值自检：先检查重构，再做优化
#
# 本节用独立的六比特直接矩阵计算检查测量重构，验证所有三比特 Pauli 测量旋转、
# 投影基态的正交性、矩阵厄米性和变分下界，并用中心差分核对一个参数的自动微分。
# 直接六比特计算仅作为小模型的验证工具，不参与后续训练的能量计算。
#
# 对随机共享线路，复系数与实系数的最优值一般不同，下方同时输出两者。
# 为避免最小特征值简并导致梯度不唯一，先检查最低两个特征值的间隔。

# %%
for letters in product('IXYZ', repeat=3):
    string = ''.join(letters)
    rotation, parity = measurement_setting(string)
    observable = reduce(torch.kron, [PAULI[p] for p in string])
    torch.testing.assert_close(rotation.mH @ torch.diag(parity.to(CDTYPE)) @ rotation, observable)

model = PhotonicProjectedEnergy(problems[0.1])
generator = torch.Generator(device=DEVICE).manual_seed(SEED)
theta_test = (
    2 * torch.pi * torch.rand(model.n_parameters, generator=generator, dtype=DTYPE, device=DEVICE)
).requires_grad_()
result = model(theta_test)
unitary = model.circuit(data=theta_test)
projected_basis = torch.stack([torch.kron(unitary[:, j], unitary[:, j]) for j in BASIS_INDICES], dim=1)
direct_matrix = projected_basis.mH @ problems[0.1]['hamiltonian'] @ projected_basis
torch.testing.assert_close(projected_basis.mH @ projected_basis, torch.eye(3, dtype=CDTYPE, device=DEVICE))
torch.testing.assert_close(result['matrix'], direct_matrix, atol=1e-10, rtol=1e-10)
torch.testing.assert_close(result['matrix'], result['matrix'].mH)
assert result['energy'].item() >= problems[0.1]['reference'] - 1e-10
state6 = projected_basis @ result['coefficients']
direct_energy = torch.vdot(state6, problems[0.1]['hamiltonian'] @ state6).real + problems[0.1]['constant']
torch.testing.assert_close(result['energy'], direct_energy)
assert (result['eigenvalues'][1] - result['eigenvalues'][0]).item() > 1e-7
gradient = torch.autograd.grad(result['energy'], theta_test)[0]
assert torch.isfinite(gradient).all()
h = 1e-5
with torch.no_grad():
    plus, minus = theta_test.detach().clone(), theta_test.detach().clone()
    plus[0] += h
    minus[0] -= h
    finite_difference = (model(plus)['energy'] - model(minus)['energy']) / (2 * h)
torch.testing.assert_close(gradient[0], finite_difference, atol=1e-7, rtol=1e-5)
print('测量旋转、投影重构、正交性、变分下界和梯度检查通过')
print('线路参数数:', model.n_parameters)
print('重构矩阵最大误差:', (result['matrix'] - direct_matrix).abs().max().item())
print('复系数能量:', result['energy'].item())
print('实系数能量:', torch.linalg.eigvalsh(result['matrix'].real)[0].item() + model.constant)


# %% [markdown]
# ## 5. 单个距离的精确概率优化
#
# 主实验设置无光损耗、`shots=None`。每个 R 标签使用独立且可复现的随机初始化，
# 不把前一个距离的最优参数作为下一个距离的初值。下方每次参数更新后重新评价能量，
# 保存历史最佳参数，并在结束时再次计算其能量和系数，避免“参数已更新但能量属于更新前”的错位。
#
# Adam 的当前能量不保证单调下降；历史最佳曲线仅是记录。150 次更新与一次随机初始化
# 也不保证找到该 Ansatz 的全局最优值。最低特征值发生简并时，梯度优化需要额外谨慎。

# %%
def optimize_problem(problem, seed, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    if type(epochs) is not int or epochs < 1:
        raise ValueError('epochs 必须为正整数')
    model = PhotonicProjectedEnergy(problem)
    rng = torch.Generator(device=DEVICE).manual_seed(seed)
    theta = nn.Parameter(2 * torch.pi * torch.rand(model.n_parameters, generator=rng, dtype=DTYPE, device=DEVICE))
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    with torch.no_grad():
        initial = model(theta)['energy'].item()
    history = [initial]
    best_energy, best_theta = initial, theta.detach().clone()
    for _ in range(epochs):
        optimizer.zero_grad()
        energy = model(theta)['energy']
        energy.backward()
        if not torch.isfinite(energy) or not torch.isfinite(theta.grad).all():
            raise RuntimeError('能量或梯度出现非有限数')
        optimizer.step()
        with torch.no_grad():
            updated = model(theta)['energy'].item()
        history.append(updated)
        if updated < best_energy:
            best_energy, best_theta = updated, theta.detach().clone()
    with torch.no_grad():
        final = model(best_theta)
    if min(history) < problem['reference'] - 1e-9:
        raise AssertionError('精确无损耗能量低于参考下界，请检查重构')
    return {
        'r': problem['r'],
        'seed': seed,
        'history': history,
        'theta': best_theta.cpu(),
        'coefficients': final['coefficients'].cpu(),
        'energy': final['energy'].item(),
        'reference': problem['reference'],
        'last_energy': history[-1],
        'error': final['energy'].item() - problem['reference'],
    }


example_r = R_VALUES[-1]
example_run = optimize_problem(problems[example_r], SEED + len(R_VALUES) - 1)
print(
    f'R={example_r}: 最佳复算能量={example_run["energy"]:.9f} Hartree，'
    f'参考={example_run["reference"]:.9f}，误差={example_run["error"] * 1000:.4f} mHartree'
)
fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
ax.plot(example_run['history'], label='Current energy')
ax.plot(np.minimum.accumulate(example_run['history']), label='Best observed energy')
ax.axhline(example_run['reference'], color='black', linestyle='--', label='Data reference')
ax.set(xlabel='Completed Adam updates (0 = initial)', ylabel='Energy (Hartree)', title=f'R label = {example_r}')
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. 九个数据标签的扫描与结果保存
#
# 复用上节已完成的 R=0.9 结果，其余距离各训练一次。报告的是每次训练历史最佳参数的精确复算能量。
# 图中误差可能包含表示限制和优化未充分收敛的影响。
#
# 输出写入独立时间戳目录，不覆盖输入数据或教程文件：`summary.json` 保存配置和误差，
# `parameters.pt` 保存参数、复系数和完整能量轨迹，PNG 保存本次曲线。
# 距离单位与模型元数据仍明确记为“未提供”。

# %%
runs = []
for index, r in enumerate(R_VALUES):
    run = example_run if r == example_r else optimize_problem(problems[r], SEED + index)
    runs.append(run)
    print(f'R={r:.1f}  E={run["energy"]:.9f}  E_ref={run["reference"]:.9f}  error={run["error"] * 1000:.4f} mHartree')

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
config = {
    'epochs': EPOCHS,
    'learning_rate': LEARNING_RATE,
    'base_seed': SEED,
    'dtype': str(DTYPE),
    'device': str(DEVICE),
    'shots': None,
    'loss_db': None,
    'shared_unitary': True,
    'coefficient_domain': 'complex',
    'basis_indices': list(BASIS_INDICES),
    'distance_unit': 'not provided',
    'basis_set': 'not provided',
    'active_space': 'not provided',
    'mapping': 'not provided',
    'constant_physical_components': 'not provided',
    'torch_version': str(torch.__version__),
    'deepquantum_version': str(getattr(dq, '__version__', 'unknown')),
}
summary = {
    'config': config,
    'results': [{k: r[k] for k in ('r', 'seed', 'energy', 'reference', 'error', 'last_energy')} for r in runs],
}
(OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
torch.save({'config': config, 'runs': runs}, OUTPUT_DIR / 'parameters.pt')

fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
axes[0].plot(R_VALUES, [r['reference'] for r in runs], label='Data reference')
axes[0].scatter(R_VALUES, [r['energy'] for r in runs], marker='x', color='tab:red', label='Variational result')
axes[0].set(ylabel='Energy (Hartree)')
axes[0].legend()
axes[1].plot(R_VALUES, [r['error'] * 1000 for r in runs], 'o-')
axes[1].axhline(0, color='gray', linestyle='--')
axes[1].set(ylabel='Energy error (mHartree)')
for ax in axes:
    ax.set_xlabel('R data label (original unit unspecified)')
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'energy_scan.png', dpi=180)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
for run in runs:
    error = (np.asarray(run['history']) - run['reference']) * 1000
    ax.semilogy(np.maximum(error, 1e-9), label=f'R={run["r"]}')
ax.set(xlabel='Completed Adam updates', ylabel='Energy error (mHartree)')
ax.legend(ncol=3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'convergence.png', dpi=180)
plt.show()
print('结果已保存:', OUTPUT_DIR)

# %% [markdown]
# ## 7. 可选机制检查：有限采样和均匀损耗
#
# 下面固定同一组优化参数，比较无损耗和 3 dB 均匀损耗。精确条件能量应相同，
# 存活概率应分别约为 1 和 $10^{-3/10}$。有限采样结果会波动；各设置分别后选择后，
# 再对含噪的矩阵取最小特征值，可能产生低估，**采样能量偶尔低于参考值不代表突破变分下界**。
# 这时测得的矩阵不一定是同一个精确物理投影矩阵。
#
# `shots` 是每个 Pauli 项、每个子系统、每个输入制备的入射样本数。
# 当前未合并重复测量设置，62 项时一次能量评估使用 $62\times2\times9=1116$ 个设置，
# 因此 `shots=1000` 对应约 111.6 万个入射样本，而不是整个能量估计只用 1000 个光子。
# 软件中的 batch 只加快模拟，不意味着真实芯片能够同时完成所有设置。

# %%
fixed_theta = example_run['theta'].to(device=DEVICE, dtype=DTYPE)
exact_energies = []
for loss_db in (None, 3.0):
    loss_model = PhotonicProjectedEnergy(problems[example_r], loss_db=loss_db)
    with torch.no_grad():
        exact = loss_model(fixed_theta)
    exact_energies.append(exact['energy'])
    expected_survival = 1.0 if loss_db is None else 10 ** (-loss_db / 10)
    torch.testing.assert_close(
        exact['survival'], torch.full_like(exact['survival'], expected_survival), atol=1e-10, rtol=1e-10
    )
    print(
        f'loss={loss_db}: 精确条件能量={exact["energy"].item():.9f}, 平均存活率={exact["survival"].mean().item():.4f}'
    )
    for shots in (100, 1000):
        values = []
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(SEED + shots)
            for _ in range(5):
                values.append(loss_model(fixed_theta, shots=shots)['energy'].item())
        print(
            f'  shots/setting={shots}: 5 次独立采样均值={np.mean(values):.9f}, '
            f'标准差={np.std(values, ddof=1):.6f} Hartree'
        )
torch.testing.assert_close(exact_energies[0], exact_energies[1], atol=1e-10, rtol=1e-10)
