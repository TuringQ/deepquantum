"""
Adjoint differentiation
"""

from copy import deepcopy
from typing import Tuple
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.autograd import Function

from .distributed import dist_one_targ_gate, dist_many_ctrl_one_targ_gate, dist_many_targ_gate, inner_product_dist
from .gate import SingleGate, CombinedSingleGate
from .operation import Gate
from .state import DistributedQubitState

if TYPE_CHECKING:
    from .layer import Observable


class AdjointExpectation(Function):
    """Adjoint differentiation

    See https://arxiv.org/pdf/2009.02823

    Args:
        state (DistributedQubitState): The final quantum state.
        operators (nn.Sequential): The quantum operations.
        observable (Observable): The observable.
        *parameters (torch.Tensor): The parameters of the quantum circuit.
    """
    @staticmethod
    def forward(
        ctx,
        state: DistributedQubitState,
        operators: nn.Sequential,
        observable: 'Observable',
        *parameters: torch.Tensor
    ) -> torch.Tensor:
        ctx.state_phi = state
        ctx.operators = operators
        ctx.observable = observable
        ctx.state_lambda = observable(deepcopy(state))
        ctx.save_for_backward(*parameters)
        return inner_product_dist(ctx.state_lambda, ctx.state_phi).real

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[None, ...]:
        parameters = [*ctx.saved_tensors]
        grads = []
        idx = 1
        for op in ctx.operators[::-1]:
            if isinstance(op, CombinedSingleGate):
                gates = op.gates
            elif isinstance(op, Gate):
                gates = [op]
            for gate in gates[::-1]:
                if gate.npara > 0:
                    gate.init_para(parameters[-idx])
                gate_dagger = gate.inverse()
                ctx.state_phi = gate_dagger(ctx.state_phi)
                if gate.npara > 0:
                    if parameters[-idx].requires_grad:
                        du_dx = gate.get_derivative(parameters[-idx]).unsqueeze(0).flatten(0, -3) # (npara, 2**n, 2**n)
                        wires = gate.controls + gate.wires
                        targets = [gate.nqubit - wire - 1 for wire in wires]
                        grads_gate = []
                        for mat in du_dx:
                            state_mu = deepcopy(ctx.state_phi)
                            if isinstance(gate, SingleGate):
                                if len(gate.controls) == 0:
                                    state_mu = dist_one_targ_gate(state_mu, targets[0], mat)
                                else:
                                    state_mu = dist_many_ctrl_one_targ_gate(state_mu, targets[:-1], targets[-1], mat,
                                                                            True)
                            else:
                                zeros = mat.new_zeros(2 ** len(wires) - 2 ** len(gate.wires)).diag_embed()
                                matrix = torch.block_diag(zeros, mat)
                                state_mu = dist_many_targ_gate(state_mu, targets, matrix)
                            grad = grad_out * 2 * inner_product_dist(ctx.state_lambda, state_mu).real
                            grads_gate.append(grad)
                        grads.append(torch.stack(grads_gate).reshape(parameters[-idx].shape))
                    else:
                        grads.append(None)
                    idx += 1
                ctx.state_lambda = gate_dagger(ctx.state_lambda)
        return None, None, None, *grads[::-1]
