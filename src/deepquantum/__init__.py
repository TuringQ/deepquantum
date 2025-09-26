"""
This is the top level module from which all basic functions and classes of
DeepQuantum can be directly imported.
"""

__version__ = '4.3.0'


from . import adjoint
from . import ansatz
from . import bitmath
from . import channel
from . import circuit
from . import communication
from . import cutting
from . import distributed
from . import gate
from . import layer
from . import operation
from . import optimizer
from . import qasm3
from . import qmath
from . import qpd
from . import state
from . import utils

from . import mbqc
from . import photonic

from .ansatz import (
    Ansatz,
    ControlledMultiplier,
    ControlledUa,
    HHL,
    NumberEncoder,
    PhiAdder,
    PhiModularAdder,
    QuantumConvolutionalNeuralNetwork,
    QuantumFourierTransform,
    QuantumPhaseEstimationSingleQubit,
    RandomCircuitG3,
    ShorCircuit,
    ShorCircuitFor15
)
from .channel import BitFlip, PhaseFlip, Depolarizing, Pauli, AmplitudeDamping, PhaseDamping
from .channel import GeneralizedAmplitudeDamping
from .circuit import QubitCircuit, DistributedQubitCircuit
from .communication import setup_distributed, cleanup_distributed
from .gate import U3Gate, PhaseShift, Identity, PauliX, PauliY, PauliZ, Hadamard
from .gate import SGate, SDaggerGate, TGate, TDaggerGate
from .gate import Rx, Ry, Rz, ProjectionJ, CombinedSingleGate
from .gate import CNOT, Swap, Rxx, Ryy, Rzz, Rxy, ReconfigurableBeamSplitter, Toffoli, Fredkin
from .gate import UAnyGate, LatentGate, HamiltonianGate, Barrier
from .layer import Observable, U3Layer, XLayer, YLayer, ZLayer, HLayer, RxLayer, RyLayer, RzLayer
from .layer import CnotLayer, CnotRing
from .qasm3 import dq_to_qasm3, qasm3_to_dq
from .qmath import multi_kron, partial_trace, amplitude_encoding, measure, expectation
from .qmath import meyer_wallach_measure
from .state import QubitState, MatrixProductState, DistributedQubitState

from .mbqc import SubGraphState, GraphState
from .mbqc import Pattern

from .photonic import permanent, takagi, hafnian, torontonian
from .photonic import FockState, GaussianState, BosonicState, CatState, GKPState, FockStateBosonic
from .photonic import QumodeCircuit, QumodeCircuitTDM, Clements, GaussianBosonSampling
from .photonic import UnitaryMapper, UnitaryDecomposer, DrawClements
from .photonic import DistributedFockState, DistributedQumodeCircuit
