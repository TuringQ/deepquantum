import torch


class Qubit(object):
    def __init__(self, state=None, wires=0, order_cur=0, order_nxt=None) -> None:
        if state == None:
            state = torch.tensor([[1], [0]], dtype=torch.cfloat)
        elif type(state) != torch.Tensor:
            state = torch.tensor(state, dtype=torch.cfloat)
        self.state = [state] # list of tensor or SubQubit
        self.wires = wires
        self.order_cur = order_cur # current decomposition level
        self.order_nxt = order_nxt # next decomposition level


class SubQubit(object):
    def __init__(self, state=None, order_cur=0, order_nxt=None) -> None:
        if state == None:
            state = torch.tensor([[1], [0]], dtype=torch.cfloat)
        elif type(state) != torch.Tensor:
            state = torch.tensor(state, dtype=torch.cfloat)
        self.state = [state] # list of tensor or SubQubit
        self.order_cur = order_cur # current decomposition level
        self.order_nxt = order_nxt # next decomposition level


class Qubits(object):
    def __init__(self, qubits=None, nqubit=1) -> None:
        self.qubits = {}
        for i in range(nqubit):
            self.qubits[i] = Qubit(wires=i)
        if qubits != None:
            for qubit in qubits:
                self.qubits[qubit.wires] = qubit
        self.nqubit = nqubit

    def update(self, qubit):
        self.qubits[qubit.wires] = qubit