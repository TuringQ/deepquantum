from operation import Layer
from gate import Rx


class RxLayer(Layer):
    def __init__(self, nqubit=1, wires=None, first_layer=False, last_layer=False):
        super().__init__(name='RxLayer', nqubit=nqubit, wires=wires, first_layer=first_layer, last_layer=last_layer)
        for i in wires:
            self.gates.append(Rx(nqubit, wires=i, MPS=True, requires_grad=True))