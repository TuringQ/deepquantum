from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import copy
import torch
from torch import nn
from operation import Operation, Gate
from gate import PhaseShift, BeamSplitter, BeamSplitter_1, BeamSplitter_2, UAnyGate
from draw import DrawCircuit
from photonic_qmath import FockOutput, CreateSubmat
from state import FockState, TensorState

class QumodeCircuit(Operation):
    """
    Constructing quantum optical circuit
    Args:
        nmode: int, the total wires of the circuit
        cutoff: int, the maximum number of photons in each mode
        init_state: fock state list example: [1,0,0], 
        fock state tensor example:[[sqrt(2)/2,(1,0)], [sqrt(2)/2,(0,1)] ]
        name: str, the name of the circuit
        us_full_tensor: if true using fock state tensor else using fock state list 

    """
    def __init__(
        self,
        nmode: int,
        cutoff: int=None,
        init_state: Any = None,
        name: Optional[str] = None,
        use_full_tensor: bool = False
    ) -> None:
        super().__init__(name=name, nmode=nmode, wires=None )
        assert( init_state is not None), "initial state is necessary for circuit input"
        self.operators = nn.Sequential()
        self.encoders = []
        self.cutoff = cutoff
        self.npara = 0
        self.u = None
        if use_full_tensor:
            self.ini_state = TensorState(nmode=nmode,  cutoff=cutoff, state=init_state)
        else:                               # initial matrix for the circuit
            assert(nmode==len(init_state)), "nmode should be equal to length of fockstate"
            self.ini_state = FockState(nmode, init_state)      # the initial state of the circuit
        self.state = None        # the current state of the circut
        self.npara = 0
        self.ndata = 0
        self.depth = np.array([0] * nmode)

    def add(
        self,
        op: Operation,
        encode: bool = False,
        wires: Union[int, List[int], None] = None
    ) -> None:
        """A method that adds an operation to the quantum circuit.

        The operation can be a gate, a layer, or another quantum circuit. The method also updates the
        attributes of the quantum circuit. If ``wires`` is specified, the parameters of gates are shared.

        Args:
            op (Operation): The operation to add. It is an instance of ``Operation`` class or its subclasses,
                such as ``Gate``, ``Layer``, or ``QubitCircuit``.
            encode (bool): Whether the gate or layer is to encode data. Default: ``False``
            wires (Union[int, List[int], None]): The wires to apply the gate on. It can be an integer
                or a list of integers specifying the indices of the wires. Default: ``None`` (which means
                the gate has its own wires)

        Raises:
            AssertionError: If the input arguments are invalid or incompatible with the quantum circuit.
        """
        assert isinstance(op, Operation)
        if wires is not None:
            assert isinstance(op, Gate)
            # if controls is None:
            #     controls = []
            wires = self._convert_indices(wires)
            # controls = self._convert_indices(controls)
            # for wire in wires:
            #     assert wire not in controls, 'Use repeated wires'
            assert len(wires) == len(op.wires), 'Invalid input'
            op = copy(op)
            op.wires = wires
            # op.controls = controls
        if isinstance(op, QumodeCircuit):
            assert self.nmode == op.nmode
            self.operators += op.operators
            self.encoders  += op.encoders
            self.observables = op.observables
            self.npara += op.npara
            self.ndata += op.ndata
            self.depth += op.depth
            self.wires_measure = op.wires_measure
            # self.wires_condition += op.wires_condition
            # self.wires_condition = list(set(self.wires_condition))
        else:
            # op.tsr_mode = True
            self.operators.append(op)
            if isinstance(op, Gate):
                for i in op.wires:  ## here no control
                    self.depth[i] += 1
                # if op.condition:
                #     self.wires_condition += op.controls
                #     self.wires_condition = list(set(self.wires_condition))
            # elif isinstance(op, Layer):
            #     for wire in op.wires:
            #         for i in wire:
            #             self.depth[i] += 1
            if encode:
                assert not op.requires_grad, 'Please set requires_grad of the operation to be False'
                self.encoders.append(op)
                self.ndata += op.npara
            else:
                self.npara += op.npara


    def ps(
        self,
        inputs: Any = None,
        wires: Union[int, List[int], None] = None,
        encode: bool = False
    ) -> None:
        """ Add a phaseshifter"""
        nmode = self.nmode
        requires_grad = not encode
        ps_ = PhaseShift(inputs=inputs, nmode=nmode, wires=wires, cutoff=self.cutoff, requires_grad=requires_grad)
        self.add(ps_, encode = encode)

    def bs(
        self,
        inputs: Any = None,
        wires: Union[int, List[int], None] = None,
        which_bs: int = 1,
        encode: bool = False
    ) -> None:
        """ Add a phaseshifter
        Args:
            inputs:
            nmode:
            wires:
            which_bs: two types of beamsplitter, which_bs=1 for BS fixing phi at pi/2,
            which_bs=2 for fixing theta at pi/4.
            encode:
        """
        nmode = self.nmode
        requires_grad = not encode
        if which_bs == 1:
            BS_ = BeamSplitter_1
        if which_bs ==2:
            BS_ = BeamSplitter_2
        bs_ = BS_(inputs=inputs, nmode=nmode, wires=wires, cutoff=self.cutoff, requires_grad=requires_grad)
        self.add(bs_, encode = encode)
    
    def any(
        self,
        unitary: Any,
        wires: Union[int, List[int], None] = None,
        minmax: Optional[List[int]] = None,
        name: str = 'uany'
    ) -> None:
        """Add an arbitrary unitary gate."""
        uany = UAnyGate(unitary=unitary, nmode=self.nmode, wires=wires, cutoff=self.cutoff)
        self.add(uany)
    
    def draw(self):
        """
        circuit plotting
        """
        if self.nmode > 50:
            print("too many wires in the circuit, run circuit.save for the complete circuit")
        self.draw_circuit = DrawCircuit(self.name, self.nmode, self.operators, self.depth)
        self.draw_circuit.draw()
        return self.draw_circuit.draw_

    def save(self, filename):
        """
        save the circuit in svg
        filename: "example.svg"
        """
        self.draw_circuit.save(filename)

    ##############################
    #### for probs and amplitude using fock state
    #############################

    def get_unitary(self) -> torch.tensor:
        """
        Get the unitary matrix of the nmode optical quantum circuit.
        """
        u = None
        for op in self.operators:
            if u is None:
                u = op.get_unitary()
            else:
                u = op.get_unitary() @ u         ## here we use the left multiplication
        self.u = u
        return u

    def op_fock_state_list(self):
        """
        get all possible output
        """
        fock_out = FockOutput(self.ini_state)
        output_list = fock_out.fock_outputs

        return output_list

    def get_amplitude(self, final_state):
        """
        Calculating the transfer amplitude of given final state
        final_state: fock state, list or torch.tensor
        """
        assert(max(final_state) < self.cutoff), " the number of photons in the  final state must be less than cutoff"
        assert(sum(final_state) == self.ini_state.photons), " the number of photons should be conserved"
        if not isinstance(final_state, torch.Tensor):
            final_state = torch.tensor(final_state, dtype=torch.int)
        ini_state = self.ini_state.state

        if self.u is None:
            u = self.get_unitary()
        else:
            u = self.u

        sub_mat = CreateSubmat.sub_matrix(u, ini_state, final_state)
        # print(sub_mat)
        per = CreateSubmat.permanent(sub_mat)
        amp = per / np.sqrt((CreateSubmat.product_factorial(ini_state) * CreateSubmat.product_factorial(final_state)))
        return amp

    def get_prob(self, final_state):
        """
        Calculating the transfer probability of given final state
        final_state: fock state, list or torch.tensor
        """
        amplitude = self.get_amplitude(final_state)
        prob = torch.abs(amplitude)**2
        return prob

    def evolve(self):
        """
        evolving the circuit with the cutoff, if max number of photons in each mode < cutoff, the state  is accepted, 
        #else the state is not included, return amplitudes without normalisation if cutoff existing
        """
        out_dic = {}
        output_list = self.op_fock_state_list()
        for ii in output_list:
            f_state = ii.state
            if max(f_state)< self.cutoff:
                out_dic[ii] = self.get_amplitude(f_state)
            # else:
            #     out_dic[ii] = self.get_prob(f_state)*0
        sorted_out_dic = QumodeCircuit.sort_dic(out_dic)
        return sorted_out_dic

    def get_all_probs(self):
        """
        evolving the circuit with the cutoff, if max number of photons in each mode < cutoff, the state  is accepted, 
        #else the state is not included, return probs without normalisation if cutoff existing
        """
        out_dic = {}
        output_list = self.op_fock_state_list()
        for ii in output_list:
            f_state = ii.state
            if max(f_state)< self.cutoff:
                out_dic[ii] = self.get_prob(f_state)
            # else:
            #     out_dic[ii] = self.get_prob(f_state)*0
        sorted_out_dic = QumodeCircuit.sort_dic(out_dic)
        return sorted_out_dic

    @staticmethod
    def sort_dic(dict_):
        """
        sort the dictionary based on decreasing values
        """
        sort_list = sorted(dict_.items(),  key=lambda t: abs(t[1]), reverse=True)
        sorted_dict = {}
        for key, value in sort_list:
            sorted_dict[key] = value
        return sorted_dict

    ##############################
    #### for evolving the circuit
    #############################

    # pylint: disable=arguments-renamed
    def forward(self) -> Union[torch.Tensor, TensorState, FockState]:
        """Perform a forward pass of the quantum circuit and return the final state."""
        if isinstance(self.ini_state, FockState):
            x = self.evolve()
        if isinstance(self.ini_state, TensorState):
            x = self.operators(self.ini_state.state)
        self.state = x
        return self.state