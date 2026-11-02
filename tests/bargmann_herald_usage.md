# Bargmann 原生预报：使用与测试

`backend='bargmann'` 的 `cir.heralded_state()` 现在分派到 `heralded_state_bargmann()`，返回 `(BargmannState, probability[, info])`。现有 Fock/Gaussian 路径保持张量返回值不变。不增加 `output` 参数，也不增加电路内 `cir.herald()` 操作。

```python
import torch
import deepquantum as dq

cir = dq.QumodeCircuit(3, 'vac', backend='bargmann')
cir.s2([0, 1], 0.3, 0.0)
cir.to(torch.float64)
cir()
state, p1, info = cir.heralded_state(0, 1, return_info=True)
# state 有两个模式，局部编号为 (0, 1)，分别对应原模式 (1, 2)。
next_cir = dq.QumodeCircuit(2, state, backend='bargmann')
next_cir.bs([0, 1], [0.3, 0.0])
next_cir.to(torch.float64)
next_cir()
final, p12 = next_cir.heralded_state(1, 0)
fock = final.to_fock(8)
```

预报不改变原电路的 `cir.state`，不会重跑门。必须至少测量一个且保留一个模式。`cutoff` 仅设置返回状态默认读取维度，不截断保留模式。输入若本来来自有限 Fock 张量，其原有截断不会消失。

被测 CV 变量转成新的 derived 变量，系数轴选取指定光子数。密度矩阵同时处理 ket/bra。已有 derived 轴与线性叠加轴保留，支持再次预报。`BargmannState._project_pnr` 是内部已验证参数辅助方法；公共入口负责校验。

返回 `probability` 是未归一化输出的范数平方或迹。对于未归一化输入，它是事件权重；上述示例中 `p12` 是两次事件的联合概率。`normalize=True` 除以该权重（纯态取平方根），零概率时报错。`info['input_weight']` 给出输入权重，`info['conditional_probability']` 给出本轮条件概率；零输入权重时后者为 NaN。

`info['remaining_wires']` 是源状态标签；返回态重新编号以便进入新电路。`probability_in_cutoff`、`retained_fraction` 为 None，因为未做 Fock 读取。`source_cutoff=None` 表示本路径不记录 Fock 源截断历史，不保证输入从未截断。

验证命令（仓库根目录，需安装依赖）：

```bash
PYTHONPATH=src python -m pytest -q tests/test_photonic_bargmann_herald.py tests/test_photonic_bargmann.py tests/test_photonic_heralded_state.py
```

新增测试涵盖纯态/密度矩阵、无序模式、批次、旧 derived 轴、多叠加项、概率及归一化、读取 cutoff 独立性、零概率与缓存、损耗混态、两轮预报的解析联合概率，以及前后电路参数的 gradcheck。

本次验证：在 `dq_202603` 环境运行上述三个测试文件，86 passed、12 skipped；修改文件的 Ruff 检查通过。
