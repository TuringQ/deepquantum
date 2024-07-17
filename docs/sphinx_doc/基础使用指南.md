# 基础使用指南

导入DeepQuantum以及相关库

```python
import deepquantum as dq
import deepquantum.photonic as dqp
import torch
import torch.nn as nn
```

## 量子线路对象

DeepQuantum的基本对象是QubitCircuit和QumodeCircuit。
QubitCircuit用量子比特数（n）来初始化量子线路，即cir=dq.QubitCircuit(n)。
QumodeCircuit用模式数（n）来初始化光量子线路，即cir=dq.QumodeCircuit(n)。
DeepQuantum可以帮助用户很方便地实现参数化量子线路，从而进行量子机器学习。

## QubitCircuit基础使用说明

### 量子态

QubitState是表示量子态的类。
比如，我们可以利用QubitState准备一个单比特量子态，其数据是torch的tensor，存在属性state中。
便捷地将经典数据编码为一个量子态

```python
qstate = dq.QubitState(nqubit=1, state=[0,1])
print(qstate.state)
```

此外，还可以调用一些内置的量子态。
默认初始化为全零态

```python
qstate = dq.QubitState(nqubit=2, state='zeros')
print(qstate.state)
```

初始化为等权叠加态

```python
qstate = dq.QubitState(nqubit=2, state='equal')
print(qstate.state)
```

初始化为GHZ态

```python
qstate = dq.QubitState(nqubit=3, state='ghz')
print(qstate.state)
```

### 基本的量子门

我们可以将各种量子门作用在QubitCircuit上。
例如，我们可以将Hadamard门作用于1号量子比特：cir.h(1)；将Rx门作用于2号量子比特：cir.rx(2, inputs=0.2)；多量子比特门也一样：cir.cnot(0, 1)。
同时，我们还可以一次性在量子线路上放置一层量子门，如cir.rxlayer()。
如果量子门没有指定输入参数，则会自动初始化变分参数。

```python
cir = dq.QubitCircuit(4)
```

第一个参数wires指定放置的线路，自动初始化内部的变分参数

```python
cir.rxlayer(wires=[0,2])
```

我们也可以手动初始化参数，具体如下所示。
利用inputs手动初始化固定参数

```python
cir.rxlayer(wires=[0, 1, 2, ...], inputs=[theta_0, theta_1, ...])
```

如果要求参数是可训练的，可以参考以下示例

```python
class MyCircuit(nn.Module):
    def __init__(self, nqubit):
        super().__init__()
        # 手动初始化变分参数为1
        self.params = nn.Parameter(torch.ones(nqubit))
        self.cir = self.circuit(nqubit)

    def circuit(self, nqubit):
        cir = dq.QubitCircuit(nqubit)
        cir.hlayer()
        # 利用encode，指定变分参数编码到量子线路中的位置
        cir.rylayer(encode=True)
        cir.cnot_ring()
        for i in range(nqubit):
            cir.observable(i)
        return cir

    def forward(self):
        # 在前向过程中，变分参数作为data加入量子线路
        self.cir(data=self.params)
        return self.cir.expectation().mean()
```

DeepQuantum支持灵活地使用多控制位的量子门（即控制位均为1时，受控的量子门才起作用），具体如下所示。

```python
cir = dq.QubitCircuit(4)
cir.toffoli(0, 1, 2) # 按顺序指定控制位和目标位
cir.fredkin(0, 1, 2)
# 一般的量子门都可以通过controls参数来指定任意控制位
cir.x(3, controls=[0,1,2])
cir.crx(0, 1)
cir.crxx(0, 1, 2)
cir.u3(2, controls=[0,1,3])
cir.draw()
```

### 测量与期望

#### 测量

测量是量子计算的核心操作之一。
我们以对GHZ态进行测量作为例子。

```python
cir = dq.QubitCircuit(3)
cir.h(0)
cir.cnot(0, 1)
cir.cnot(0, 2)
cir.barrier()
cir()
# 对QubitCircuit中的末态进行测量，返回的结果是字典或者字典的列表
# 字典的key是比特串，value是对应测量到的次数
# 测量总数shots默认为1024
# 比特串从左到右对应于线路从小到大
# 即第一个qubit在最高位，最后一个qubit在最低位
print(cir.measure())
# 我们也可以设定采样次数、进行部分测量以及显示理想的概率。
print(cir.measure(shots=100, wires=[1,2], with_prob=True))
```

#### 期望

使用参数化量子线路进行变分量子算法时，往往涉及对末态求某个力学量的期望值，下面我们就用最简单的量子线路演示一下。

```python
cir = dq.QubitCircuit(4)
cir.xlayer([0,2])
# 可以添加多个observable，各个期望值的结果会自动拼接起来
# 还可以使用列表形式的组合来灵活地指定测量线路和测量基底
# 如wires=[0,1,2]、basis='xyz'
# 即表示0、1、2号线路上分别对应Pauli-X、Pauli-Y、Pauli-Z的力学量
for i in range(4):
    cir.observable(i)
cir() # 线路演化后才能得到期望值
print(cir.expectation())
```

#### 条件测量

使用condition参数可以实现条件测量，此时条件测量的位置由控制位controls决定。

```python
cir = dq.QubitCircuit(3)
cir.h(0)
cir.x(1, controls=0, condition=True)
cir.x(2, controls=1, condition=True)
print(cir())
# 然后可以进行随机的延迟测量
state, measure_rst, prob = cir.defer_measure(with_prob=True)
print(state)
# 也可以选择特定的测量结果
print(cir.post_select(measure_rst))
cir.draw()
```

注意，defer_measure和post_select并不改变QubitCircuit所保存的末态state，因此measure和expectation目前并不兼容条件测量。

## QumodeCircuit基础使用说明

### Fock后端的使用

使用Fock后端可以构建基于离散变量的光量子线路，basis=True对应着量子态用Fock基矢态表示，basis=False对应着量子态用Fock态张量表示。

#### 量子态

Fock基矢态表示只支持单一的量子态输入，Fock态张量表示支持叠加态输入。

```python
qstate1 = dq.FockState(state=[1,0,1,0], basis=True)
qstate2 = dq.FockState(state=[(0.6, [1,0,1,0]), (0.8, [1,1,0,0])], basis=False)
print(qstate1, qstate2)
```

#### 基本量子门

我们可以将各种量子门作用在QumodeCircuit上。
例如，我们可以将移相器作用于第1个mode：cir.ps(0)；将分束器作用于第1、2个mode：cir.bs([0,1])。
如果量子门没有指定输入参数，则会自动初始化变分参数。

```python
init_state = [1,0]
cir = dq.QumodeCircuit(nmode=2, init_state=init_state, cutoff=3, backend='fock', basis=True)
cir.ps(0, torch.pi)
cir.ps(1, torch.pi)
cir.bs([0,1], [torch.pi/4, torch.pi/4])
cir.mzi([0,1], [torch.pi/4, torch.pi/4])
```

#### 线路演化和测量

```python
state = cir()
sample = cir.measure()
print(state, sample)
```

#### 线路可视化

通过cir.draw()可以将搭建的光量子线路可视化并保存

```python
cir.draw(filename='circuit.svg')
```

### Clements 的使用

构建基于Clements架构的光量子线路，可以通过Clements架构实现任意的光量子酉矩阵线路映射，下面是一个简单CNOT门的例子。
这里使用6模线路来实现一个1/9成功率的CNOT门，第1、2个mode对应控制比特，第3、4个mode对应目标比特，第5、6个mode为辅助mode。

```python
u6x6 = torch.tensor([[1, 0, 1, -1, 0, 0],
                     [0, 1, 0, 0, 0, 2 ** 0.5],
                     [1, 0, 0, 1, 1, 0],
                     [-1, 0, 1, 0, 1, 0],
                     [0, 0, 1, 1, -1, 0],
                     [0, 2 ** 0.5, 0, 0, 0, -1]]) / 3 ** 0.5
# 将酉矩阵分解成Clements对应的参数
ud = dq.UnitaryDecomposer(u6x6)
angle_dict = ud.decomp()[2]
clements = dq.Clements(nmode=6, init_state=[1,0,1,0,0,0], cutoff=3)
data = clements.dict2data(angle_dict) # 将线路参数编码到6x6的Clements架构中
state = clements(data=data)
clements.draw()
```

通过后选择挑选出对应的量子态，后选择规则是第1、2个mode探测到光子数之和为1，同时第3、4个mode探测到光子数之和也为1，挑选出的结果为|101000>，对应的概率为1/9。

### Gaussian后端的使用

使用Gaussian后端可以构建基于连续变量的光量子线路，量子态通过正交算符 $\hat{x}$ 和 $\hat{p}$ 的协方差矩阵和平均值矢量来表示。

#### 量子态

高斯态通过协方差矩阵cov和平均值矢量mean来表示，'vac'对应真空态，也可以设置cov和mean。

```python
qstate1 = dq.GaussianState(nmode=2, state='vac')
cov = torch.eye(4)
mean = torch.zeros(4)
qstate2 = dq.GaussianState(nmode=2, state=[cov, mean])
print(qstate1, qstate2)
```

#### 基本量子门

高斯操作对应的量子门有单模压缩门s、位移门d、旋转门r和分束器bs等。

```python
cir = dq.QumodeCircuit(nmode=2, init_state='vac', cutoff=3, backend='gaussian')
cir.s(0, r=0.1)
cir.s(1, r=0.2)
cir.d(0, r=0.3)
cir.d(1, r=0.4)
cir.bs([0,1], [torch.pi/4, torch.pi/4])
#线路可视化
cir.draw()
```

上面的压缩门和位移门具有r和theta两个参数。
当用户只指定r时，theta默认为0。
当用户只指定theta时，r默认随机初始化。

#### 线路演化和测量

这里measure_homodyne测量对应的物理量是正交算符 $\hat{x}$ 和 $\hat{p}$ 的值，photon_number_mean_var对应的是每个mode的光子数的平均值和方差。

```python
state = cir()
sample = cir.measure_homodyne()
photon_number = cir.photon_number_mean_var()
print(sample, photon_number)
```

### 高斯玻色采样（GBS）

通过压缩门和线性光学器件构造高斯玻色采样的线路。
这里采用6个节点的图演示简单的高斯玻色采样。

```python
adj_mat = torch.tensor([[0., 1., 1., 0., 0., 0.],
                        [1., 0., 0., 1., 0., 1.],
                        [1., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 1., 1.],
                        [0., 0., 0., 1., 0., 0.],
                        [0., 1., 0., 1., 0., 0.]])
gbs = dqp.GBS_Graph(adj_mat=adj_mat, cutoff=2)
state = gbs()
sample = gbs.measure() # 高斯玻色采样
print(sample)
```

通过gbs.draw()可以可视化高斯玻色采样线路。

```python
gbs.draw()
```
