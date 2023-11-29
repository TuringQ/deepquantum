from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import copy
import torch
from torch import nn, vmap
from operation import Operation, Gate
from gate import PhaseShift, BeamSplitter, BeamSplitter_1, BeamSplitter_2, UAnyGate
from draw import DrawCircuit
from photonic_qmath import FockOutput, CreateSubmat
from state import FockState
from collections import defaultdict, Counter
import random
import itertools

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
        cutoff: int = None,
        basis: bool = False,
        init_state: Any = None,
        name: Optional[str] = None,
        noise_model: bool = True
    ) -> None:
        super().__init__(name=name, nmode=nmode, wires=list(range(nmode)) )
        assert( init_state is not None), "initial state is necessary for circuit input"
        self.noise_model = noise_model
        self.operators = nn.Sequential()
        self.encoders = []
        self.cutoff = cutoff
        self.u = None
        self.basis = basis
        self.init_state = FockState(nmode=nmode,  cutoff=cutoff, state=init_state, basis=basis)
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

    def get_unitary_op(self) -> torch.tensor:
        """
        Get the unitary matrix of the nmode optical quantum circuit.
        """
        u = None
        for op in self.operators:
            if u is None:
                u = op.get_unitary_op()
            else:
                u = op.get_unitary_op() @ u         ## here we use the left multiplication
        self.u = u
        return u

    def op_fock_state_list(self, init_state=None):
        """
        get all possible output
        """
        if init_state is None:
            init_state = self.init_state
        fock_out = FockOutput(init_state)
        output_list = fock_out.fock_outputs

        return output_list

    def get_amplitude(self, init_state, final_state):
        """
        Calculating the transfer amplitude of given final state, first need run the circuit 
        final_state: fock state, list or torch.tensor
        """
        assert(max(final_state) < self.cutoff), " the number of photons in the  final state must be less than cutoff"
        assert(sum(final_state) == sum(self.init_state.state)), " the number of photons should be conserved"
        if not isinstance(final_state, torch.Tensor):
            final_state = torch.tensor(final_state, dtype=torch.int)
        if init_state is None:
           init_state = self.init_state

        if self.u is None:
            u = self.get_unitary_op()
        else:
            u = self.u
        sub_mat = CreateSubmat.sub_matrix(u, init_state.state, final_state)
        # print(sub_mat)
        nphotons = self.init_state.state.sum()
        if nphotons == 0:   # cosider all modes 0 inputs
            amp = torch.tensor(1.)
        else:
            per = CreateSubmat.permanent(sub_mat)   # take most of the time
            # if len(sub_mat.size()) == 0 :   # for the single photon case
            #     per = sub_mat
            # else:
            #     per = self.per_function(sub_mat)
            amp = per / np.sqrt((CreateSubmat.product_factorial(init_state.state) * CreateSubmat.product_factorial(final_state)))
            
        return amp

    def get_prob(self, final_state):
        """
        Calculating the transfer probability of given final state
        final_state: fock state, list or torch.tensor
        """
        # nphotons = self.init_state.state.sum()
        # self.per_function = torch.jit.trace(CreateSubmat.permanent, torch.eye(nphotons)) 

        amplitude = self.get_amplitude(final_state)
        prob = torch.abs(amplitude)**2
        return prob

    def evolve(self, init_state=None):
        """
        evolving the circuit with the cutoff, if max number of photons in each mode < cutoff, the state  is accepted, 
        #else the state is not included, return amplitudes without normalisation if cutoff existing
        """
        out_dic = {}
        output_list = self.op_fock_state_list(init_state)
        # nphotons = self.init_state.state.sum()
        # self.per_function = torch.jit.trace(CreateSubmat.permanent, torch.eye(nphotons)) 
        # #here jit for computing acceleration, torch.eye(nphotons) for example input 
        for ii in output_list:
            f_state = ii.state
            if max(f_state)< self.cutoff:
                out_dic[ii] = self.get_amplitude(init_state, f_state).reshape(1)
            # else:
            #     out_dic[ii] = self.get_prob(f_state)*0
        # sorted_out_dic = QumodeCircuit.sort_dic(out_dic)
        return out_dic

    def get_all_probs(self):
        """
        evolving the circuit with the cutoff, if max number of photons in each mode < cutoff, the state  is accepted, 
        #else the state is not included, return probs without normalisation if cutoff existing
        """
        out_dic = {}
        output_list = self.op_fock_state_list()
        # nphotons = self.init_state.state.sum()
        # self.per_function = torch.jit.trace(CreateSubmat.permanent, torch.eye(nphotons)) 
        for ii in output_list:
            f_state = ii.state
            if max(f_state)< self.cutoff:
                out_dic[ii] = self.get_prob(f_state)
            # else:
            #     out_dic[ii] = self.get_prob(f_state)*0
        # sorted_out_dic = QumodeCircuit.sort_dic(out_dic)
        return out_dic

    @staticmethod
    def sort_dic(dict_, idx_=0):
        """
        sort the dictionary based on decreasing values
        """
        sort_list = sorted(dict_.items(),  key=lambda t: abs(t[1][idx_]), reverse=True)
        sorted_dict = {}
        for key, value in sort_list:
            sorted_dict[key] = value
        return sorted_dict

    ##############################
    #### for evolving the circuit
    #############################

    # pylint: disable=arguments-renamed
    def forward(self, state=None, data=None) -> Union[torch.Tensor, FockState]:
        """Perform a forward pass of the quantum circuit and return the final state
        with batch data.
        Args:
            state: the state to be evolved.  Default: ``None``
            data: the circuit parameters(angles).  Default: ``None``
        """
        if state is None:
            state = self.init_state
        else:
            state = FockState(nmode=self.nmode,  cutoff=self.cutoff, state=state, basis=self.basis)
    
        if data is None:
            if self.basis:
                state_out = self._forward_helper_1(state=state)
                self.state = self.sort_dic(state_out)   # sort the dict based on decreasing probs

            else:
                self.state = self._forward_helper_2(state=state.state)
        
        else:    
            if self.basis: # construct a vmap for basis list input
                 state_out = vmap(self._forward_helper_1, in_dims=(0, None))(data, state)
                 self.state = self.sort_dic(state_out)   # sort the dict based on decreasing probs
                 self.encode(data[-1])       # for plotting the last data in the circuit
                 self.u = self.get_unitary_op()
            else:                            
                if state.state.shape[0] == 1:
                    self.state = vmap(self._forward_helper_2, in_dims=(0, None))(data, state.state)
                else:
                    self.state = vmap(self._forward_helper_2)(data, state.state)
                    
                self.encode(data[-1])
        return self.state

    def _forward_helper_1(self, data=None, state=None): 
        """Perform a forward pass for one sample if the input is basis list."""
        self.encode(data)   # encoding the parameter
        self.u = self.get_unitary_op()  # updata the parameters
        x = self.evolve(state)
        return x


    def _forward_helper_2(self, data=None, state=None):
        """Perform a forward pass for one sample if the input is tensor."""
        self.encode(data)   # encoding the parameter
        x = self.operators(state).squeeze(0)
        return x

    def encode(self, data: torch.Tensor) -> None:
        """Encode the input data into thecircuit parameters.

        This method iterates over the ``encoders`` of the circuit and initializes their parameters
        with the input data. Here we assume phaseshifter and beamsplitter with single parameters

        Args:
            data (torch.Tensor): The input data for the ``encoders``, must be a 1D tensor.
        """
        if data is None:
            return
        count = 0
        for op in self.encoders:
            count_up = count + op.npara
            para_temp = data[count: count_up]
            # print(para_temp)
            op.init_para(para_temp)
            count = count_up
    
    def ps(
        self,
        inputs: Any = None,
        wires: Union[int, List[int], None] = None,
        noise_mean=0,
        noise_std=0,
        encode: bool = False
    ) -> None:
        """ Add a phaseshifter"""
        nmode = self.nmode
        requires_grad = not encode
        if inputs is not None:
            requires_grad = False
        ps_ = PhaseShift(inputs=inputs,
                         nmode=nmode,
                         wires=wires,
                         cutoff=self.cutoff,
                         requires_grad=requires_grad,
                         noise_model = self.noise_model,
                         noise_mean = noise_mean,
                         noise_std = noise_std)
        self.add(ps_, encode = encode)

    def bs(
        self,
        inputs: Any = None,
        wires: Union[int, List[int], None] = None,
        noise_mean=0,
        noise_std =0,
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
        if inputs is not None:
            requires_grad = False
        if which_bs == 1:
            BS_ = BeamSplitter_1
        if which_bs ==2:
            BS_ = BeamSplitter_2
        bs_ = BS_(inputs=inputs,
                  nmode=nmode,
                  wires=wires,
                  cutoff=self.cutoff,
                  requires_grad=requires_grad,
                  noise_model = self.noise_model,
                  noise_mean = noise_mean,
                  noise_std = noise_std)
        self.add(bs_, encode = encode)
    
    def any(
        self,
        unitary: Any,
        wires: Union[int, List[int], None] = None,
        name: str = 'uany'
    ) -> None:
        """Add an arbitrary unitary gate."""
        uany = UAnyGate(unitary=unitary, nmode=self.nmode, wires=wires, cutoff=self.cutoff)
        self.add(uany)

    def measure(self, measure_wires=None, state=None, shots=1024):
        """
        measure several wires outputs, default shots = 1024
        Args:
             measure_wires: list, the wires indicies to be measured
             state: the initial state, default None
             shots: total measurement times, default 1024
        """
        if measure_wires is None:
            measure_wires = self.wires
        if self.state is None:
            print(" please evolve the circuit")
            return 
        else:
            prob_dis = self.state 
        all_results = []
        if self.basis:
            batch = len(prob_dis[list(prob_dis.keys())[0]])
            for k in range(batch):
                prob_measure_dict = defaultdict(list)
                for key in prob_dis.keys():
                    s_ = key.state[(measure_wires)]
                    s_ = FockState(state=s_, basis=self.basis)
                    temp = abs(prob_dis[key][k])**2
                    prob_measure_dict[s_].append(temp) 
                for key in prob_measure_dict.keys():
                    prob_measure_dict[key] = sum(prob_measure_dict[key])
                samples = random.choices(list(prob_measure_dict.keys()), list(prob_measure_dict.values()), k =shots)
                results = dict(Counter(samples))
                all_results.append(results)

    
        else:  # tensor state with batch
            state_tensor = self.tensor_rep(prob_dis)

            # if len(state_tensor.shape) == self.nmode:
            #     state_tensor = state_tensor.unsqueeze(0)
            batch = state_tensor.shape[0]
            combi = list(itertools.product(range(self.cutoff), repeat = len(measure_wires)))
          
            for i in range(batch):
                dict_ = {}
                state_i = state_tensor[i]
                probs_i = abs(state_i)**2
                if measure_wires == self.wires:
                    ptrace_probs_i = probs_i   # no need for ptrace if measure all
                else:
                    sum_idx = list(range(self.nmode))
                    for idx in measure_wires:
                        sum_idx.remove(idx)
                    ptrace_probs_i = probs_i.sum(dim=sum_idx)  # here partial trace for the measurement wires,此处可能需要归一化
                for p_state in combi:  
                    lst1=list(map(lambda x:str(x),p_state))
                    state_str =  ''.join(lst1)
                    p_str = ('|' + state_str + '>')
                    dict_[p_str] = ptrace_probs_i[tuple(p_state)]
                
                samples = random.choices(list(dict_.keys()), list(dict_.values()), k =shots)
                results = dict(Counter(samples))
                all_results.append(results)
        return all_results