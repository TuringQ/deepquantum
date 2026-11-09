# Bargmann CUDA 测试

脚本：`tests/run_bargmann_cuda.py`。需使用包含 `heralded_state_bargmann()` 的 DeepQuantum 源码或安装版本，并安装适合 GPU 驱动的 CUDA 版 PyTorch。脚本只依赖 PyTorch 与 DeepQuantum，不需要 pytest 或 The Walrus。

## 运行

将脚本放在目标仓库的 `tests` 目录，在仓库根目录运行：

```bash
python tests/run_bargmann_cuda.py --device cuda:0 --output bargmann_cuda_report.json
```

默认 `--dtype auto` 在 CUDA/CPU 上运行 float32/complex64、float64/complex128 两套，共 14 个用例。默认 cutoff=6、batch=3，训练用例每种精度运行 20 步。

```bash
# 先快速检查双精度
python tests/run_bargmann_cuda.py --device cuda:0 --dtype float64 --steps 3

# 调整读取范围和 batch；建议先跑默认规模
python tests/run_bargmann_cuda.py --device cuda:0 --cutoff 8 --batch-size 8 --steps 20
```

脚本放在仓库 `tests` 内时优先导入相邻的 `src/deepquantum`，否则使用已安装的包。启动时打印实际导入路径，检查是否为预期版本。仅复制脚本不会更新远端机器上的 DeepQuantum 实现。

没有 CUDA 时默认报错退出，不会自动退回 CPU。macOS Metal GPU 测试使用 `run_bargmann_mps.py`，见 [MPS 测试说明](README_bargmann_mps.md)。也可以显式做 CPU 流程自检：

```bash
python tests/run_bargmann_cuda.py --device cpu --output bargmann_cpu_selfcheck.json
```

这种运行会明确标为 `CPU self-check only`，不代表 GPU 已通过。

## 检查内容

| 用例 | 验证 |
|---|---|
| `pure_native_herald` | 原生预报概率、复相位、单位范数，默认 cutoff=1 不丢失原生态信息 |
| `mixed_loss_herald` | 预报前损耗形成的混态，与解析概率和条件密度矩阵对照；对压缩/透射率求梯度 |
| `encoded_batch` | 编码 batch、位移、预报、Fock 读取与输入梯度 |
| `existing_derived_unsorted` | 非高斯 Fock 输入，已有辅助轴和无序测量模式，与源张量切片对照 |
| `linear_superposition` | 两个相干指数项及其干涉概率 |
| `two_rounds_training` | 三模、两轮预报、中间分束器和损耗，联合/条件概率、前后参数梯度及 Adam 更新 |
| `zero_and_invalid` | 零概率态、拒绝零概率归一化、必须至少测量且保留一个模式 |

每项先执行相同精度的 CPU 参考，再执行 GPU 版本。检查状态的 `a,b,c` 及返回张量是否在指定设备，检查所有结果是否有限，并比较输出与梯度。概率和部分状态还有独立解析对照，避免只检查 CPU/GPU 是否一致。

两轮训练目标为最大化联合成功概率，验证 20 步中每步的解析概率、梯度有限且非零、最终损失下降；这不是保真度优化功能的验收。连续参数通过完整计算图传播，未对预报态 detach。

## 结果

终端逐项显示 PASS/FAIL。JSON 记录环境、源码路径、可获取的仓库 commit、参数、每项耗时/误差、GPU 峰值 allocated/reserved 显存，以及失败堆栈。每完成一项就保存报告，便于排查中途失败。全部通过时退出码 0，有测试失败时退出码 1；启动参数或 CUDA 配置错误退出码 2。

CPU/GPU 比较容差：float32 的 atol=3e-5、rtol=5e-4；float64 的 atol=2e-9、rtol=2e-7。失败时先保留报告，不建议直接放宽容差。

耗时包含电路创建、断言、CPU 调度及必要同步，是一次端到端运行耗时，不是预热后的纯 kernel 基准。CPU 先运行可能预热递推计划缓存；小规模计算 GPU 不一定更快。显存统计每项重置峰值，但 reserved 值可能包含此前用例的缓存。脚本不会放宽源码中的递推内存上限。

本地只能执行 CPU 自检；CUDA 的正确性、兼容性和性能需以目标机器生成的报告为准。运行后可提供终端 FAIL 部分或 JSON 报告进行定位。
