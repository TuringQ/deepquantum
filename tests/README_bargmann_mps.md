# Bargmann macOS MPS 测试

这里的 MPS 指 PyTorch 的 Metal GPU 设备，不是矩阵乘积态表示。测试入口为 `tests/run_bargmann_mps.py`，复用 `tests/run_bargmann_cuda.py` 的七组解析对照、CPU 对照和梯度用例。复制到另一台机器时，两个 Python 文件需要放在同一目录；具体用例见 `README_bargmann_cuda.md`。

## 运行

在 DeepQuantum 仓库根目录，用安装了 PyTorch 和当前 DeepQuantum 依赖的 Python 环境运行：

```bash
python tests/run_bargmann_mps.py --output bargmann_mps_report.json
```

也可以直接使用公共脚本：

```bash
python tests/run_bargmann_cuda.py --device mps --output bargmann_mps_report.json
```

默认 `--dtype auto` 在 MPS 上选择 float32/complex64，共七个用例，训练 20 步。MPS 不支持 float64/complex128，显式指定 `--dtype float64` 或 `both` 会报错退出，不会静默转换精度。CUDA/CPU 的 auto 仍运行两种精度，共 14 项。

```bash
# 缩短训练，仅用于初步排查
python tests/run_bargmann_mps.py --steps 3 --output bargmann_mps_quick_report.json

# 增大 batch 和 Fock 读取 cutoff
python tests/run_bargmann_mps.py --batch-size 8 --cutoff 8 --output bargmann_mps_report.json
```

CPU 参考与 MPS 使用相同精度。float32 容差为 atol=3e-5、rtol=5e-4；不为 MPS 放宽。没有 MPS 时退出码为 2，不会自动转 CPU。严格测试要求未启用 `PYTORCH_ENABLE_MPS_FALLBACK=1`；如已设置，先在终端执行 `unset PYTORCH_ENABLE_MPS_FALLBACK`，再启动新进程。脚本拒绝开启该选项的运行，以免把算子 CPU 回退当作 MPS 支持。

## 报告含义

每个用例独立记录 PASS/FAIL、完整异常链及失败阶段：`cpu_reference`、`target_execution` 或 `cpu_target_comparison`。全部通过退出码为 0，有失败退出码为 1。报告记录操作系统、PyTorch 版本、源码 commit、导入路径、MPS 可用性和相关环境变量。

成功用例的 MPS 耗时在计时前后同步设备，但包含电路构造、断言和 CPU 调度；没有预热与重复采样，不是性能基准。内存记录为用例结束时 `current_allocated_mib` 和 `driver_allocated_mib` 的快照，**不是峰值**；驱动分配可能包含此前用例的缓存。

## 本机实测与当前限制

2026-11-03 在 macOS 26.6.2 / arm64、Python 3.14.3、PyTorch 2.10.0 上，测试源码基于 commit `aa77d7a`，七个 CPU 参考均完成，但七个 MPS 用例均失败在 `target_execution`。当前无法宣称 Bargmann 演化、预报和训练闭环已支持此 MPS 环境。

共同底层异常为：

```text
RuntimeError: linalg.lu_factor(): MPS only supports floats.
```

触发位置为 `src/deepquantum/photonic/bargmann/integrals.py` 的 `gaussian_contract()` 中复数 `torch.linalg.solve(omega - a_ii, rhs)`。创建 complex64 张量及简单复数运算可执行，并不意味着复数线性代数算子可执行。

当前实现捕获所有 RuntimeError 并包装为 `ValueError: Singular Gaussian integral; ...`。这次外层报错不能解释为输入态或高斯积分真的奇异，应查看报告中保留的底层异常链。

本次仅增加测试支持，未修改计算内核。若继续适配 MPS，需要处理复数线性求解及后续行列式等算子，并重新验证前向值和梯度；七项测试均在首个不支持的算子处停止，因此不能排除后续还有兼容问题。开启 CPU 回退也不能视作原生 MPS 测试通过。
