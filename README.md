# DeepQuantum

DeepQuantum aims at establishing a bridge between artificial intelligence (AI) and quantum computing (QC). It provides strong support for the scientific research community and developers in the field to easily develop QML applications.

## Install DeepQuantum

Install DeepQuantum with the following commands.

> git clone http://gitlab.turingq.com/deepquantum/deepquantum.git
>
> cd deepquantum
>
> pip install -e .

# Example

```python
from deepquantum import *
import torch
import torch.nn as nn

# use common gates

# 所有的Gate、Layer都是Operation的子类，都有`den_mat`和`tsr_mode`
# 参数`den_mat`表示这个操作是处理密度矩阵，还是态矢
# 参数`tsr_mode`表示这个操作的输入输出是张量态，还是态矢或密度矩阵
# 初始化一个态矢以及量子门，量子门默认不计算梯度
x = torch.tensor([0,0,0,0,0,0,1,0]) + 0j
cnot = CNOT(nqubit=3, wires=[1,0]) # wires分别对应control和target
# 不指定初始参数，则会自动随机初始化
rx = Rx(nqubit=3, wires=1)
# nqubit一致时可以直接作用
print('cnot(x)', cnot(x))
print('rx(x)', rx(x))
# 也可以手动要求计算梯度
ry = Ry(requires_grad=True)
# 获得量子门的表示，matrix属性的结果不包含梯度
print('ry.matrix', ry.matrix)
theta = torch.randn(1, requires_grad=True)
phi = torch.randn(1, requires_grad=True)
lambd = torch.randn(1, requires_grad=True)
# 可以指定初始参数并获得量子门的表示
print('rz无梯度', Rz(inputs=2.).matrix)
print('rz参数无梯度', Rz().get_matrix(2.))
print('rz有梯度', Rz().get_matrix(theta))

theta = torch.tensor(1., requires_grad=True)
phi = torch.tensor(1., requires_grad=True)
lambd = torch.tensor(1., requires_grad=True)
u3 = U3Gate(inputs=[theta, phi, lambd], requires_grad=True)
print('u3.matrix', u3.matrix)
print('U3Gate().get_matrix', U3Gate().get_matrix(theta, phi, lambd))
# Layer默认计算梯度，不指定初始参数，则会自动随机初始化
# wires可以用列表指定放置哪几条线路
# 单比特门的Layer，默认是所有线路
rxl = RxLayer(nqubit=2)
# 得到Gate或Layer的完整表示
print('rxl.get_unitary()', rxl.get_unitary())

# PQC
N = 4
batch = 2
# amplitude encoding
print('amplitude encoding')
data = torch.randn(batch, 2 ** N)
cir = Circuit(N)
# 振幅编码，多余的数据会被舍弃，自动做了归一化
# 通过替换init_state实现，与线路本身的构建无关
cir.amplitude_encoding(data)
# 加一层rx
cir.rxlayer()
# 加一层ry并指定线路
cir.rylayer(wires=0)
# 加一层cnot ring，可以用minmax参数指定线路范围，step设定每一对control和target相隔的距离，reverse指定是否从大到小
cir.cnot_ring()
# 添加测量，同样记录在列表中，可以添加多个，自动得到所有测量结果
# 可以指定测量的线路和观测量，包括用列表形式的组合
cir.measure(wires=0, observables='x')
# forward得到末态
state = cir()
# 得到测量期望，shape为(batch, 测量次数)
exp = cir.expectation()
print('state', state, state.norm(dim=-2))
print(cir.state.shape)
print('expectation', exp)

# angle encoding
print('angle encoding')
data = torch.sin(torch.tensor(list(range(batch * N))).float()).reshape(batch, N)
cir = Circuit(N)
cir.hlayer()
# 角度编码只需对相应的Layer指定encode=True，自动将数据的特征依次加入编码层，多余的会被舍弃
cir.rxlayer(encode=True)
cir.rylayer(wires=[0, 2])
cir.rxlayer()
cir.cnot_ring()
for i in range(N):
    cir.measure(wires=i)
state = cir(data)
exp = cir.expectation()
print(cir.state_dict())
print('state', state, state.norm(dim=-2))
print(cir.state.shape)
print('expectation', exp)
# 如果不重新初始化编码层，数据与对应编码的门参数是一致的
# 但编码层的参数基于数据，结合vmap将无法保存模型
# for i in range(N):
#     print(data[:,i])
#     print(cir.operators[1].gates[i].theta)

# hybrid
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)
        self.cir = Circuit(N)
        self.build_circuit()
    
    def build_circuit(self):
        self.cir.hlayer()
        self.cir.rxlayer(encode=True)
        self.cir.rylayer(wires=[0, 2])
        self.cir.rxlayer()
        self.cir.cnot_ring()
        for i in range(N):
            self.cir.measure(wires=i)
    
    @Time() # 帮助计时的装饰器
    def forward(self, x):
        x = torch.arctan(self.fc(x))
        self.cir(x)
        exp = self.cir.expectation()
        return exp

nfeat = 8
x = torch.sin(torch.tensor(list(range(batch * nfeat))).float()).reshape(batch, nfeat)
net = Net()
y = net(x)
print(net.state_dict())
print('y', y)
# 同一个Layer的id一致
print(id(net.cir.operators[1]))
print(id(net.cir.encoders[0]))
```

