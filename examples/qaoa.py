# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dq
#     language: python
#     name: python3
# ---

# %%
import deepquantum as dq
import torch
import torch.nn as nn


class QAOA(nn.Module):
    def __init__(self, nqubit, pairs, coefs, step):
        super().__init__()
        self.nqubit = nqubit
        self.pairs = pairs
        self.coefs = torch.tensor(coefs)
        self.step = step
        self.gamma = nn.Parameter(0.1 * torch.ones(step))
        self.beta = nn.Parameter(torch.ones(step))

        self.cir = dq.QubitCircuit(nqubit)
        self.cir.hlayer()
        self.cir.barrier()
        for _ in range(step):  # step量子网络演化步数，步数越高计算越精确
            for wires in pairs:  # Hp项
                self.cir.cnot(wires[0], wires[1])
                self.cir.rz(wires[1], encode=True)
                self.cir.cnot(wires[0], wires[1])
                self.cir.barrier()
            for i in range(nqubit):  # Hb项
                self.cir.rx(i, encode=True)
            self.cir.barrier()
        for wires in pairs:  # 测量每一项Hp
            self.cir.observable(wires)

    @property
    def params(self):
        params = torch.empty(0)
        for i in range(self.step):
            gammas = 2 * self.gamma[i] * self.coefs
            betas = 2 * self.beta[i].repeat(self.nqubit)
            params = torch.cat([params, gammas, betas])
        return params

    def draw(self):  # 画出量子线路图
        self.cir.encode(self.params)
        return self.cir.draw()

    def measure(self):  # 测量线路
        return self.cir.measure()

    def forward(self):
        self.cir(self.params)
        return sum(self.coefs * self.cir.expectation())  # 测量的哈密顿量可能需要根据具体的问题进行一定的修改


# %%
def trainer(model, epoch, lr):
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    train_loss_list = []
    for e in range(epoch):
        y_pred = model()
        loss = y_pred
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss_list.append(loss.detach().numpy())
        print(f'Iteration: {e} loss {loss.item()}')
    # torch.save(model.state_dict(), save_path+model_type+'.pt')   # 保存训练好的模型参数，用于后续的推理或测试
    metrics = {'epoch': list(range(1, len(train_loss_list) + 1)), 'train_loss': train_loss_list}
    return model, metrics


# %%
# 定义问题，有若干个元素，它们如果放在一起，则两两之间有一定的矛盾值，求将他们分为两组，使总矛盾值最小。
problem = {
    'Z0 Z4': 0.73,
    'Z0 Z5': 0.33,
    'Z0 Z6': 0.5,
    'Z1 Z4': 0.69,
    'Z1 Z5': 0.36,
    'Z2 Z5': 0.88,
    'Z2 Z6': 0.58,
    'Z3 Z5': 0.67,
    'Z3 Z6': 0.43,
}

pairs = []
coefs = []

for key, value in problem.items():
    temp = []
    for item in key.split():
        if item[0] == 'Z':
            temp.append(int(item[1:]))
    if len(temp) == 2:
        pairs.append(temp)
        coefs.append(value)

print('pairs:', pairs)
print('coefs:', coefs)

nqubit = 7  # 比特的总个数，应与problem中的比特数保持一致
step = 4  # 量子网络演化步数，步数越高计算越精确
lr = 0.01  # 定义学习率
epoch = 100  # 定义迭代次数

qaoa_model = QAOA(nqubit, pairs, coefs, step)

# %%
optim_model, metrics = trainer(qaoa_model, epoch, lr)

# %%
res = qaoa_model.measure()
print(res)
max_key = max(res, key=res.get)
print('最佳分割方案' + str(max_key))

# %%
qaoa_model.draw()
