"""
This is the top level module from which all basic functions and classes of
DeepQuantum can be directly imported.
"""

__version__ = '3.5.0'


from . import ansatz
from . import channel
from . import circuit
from . import gate
from . import layer
from . import operation
from . import optimizer
from . import qmath
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
from .circuit import QubitCircuit
from .gate import U3Gate, PhaseShift, Identity, PauliX, PauliY, PauliZ, Hadamard
from .gate import SGate, SDaggerGate, TGate, TDaggerGate
from .gate import Rx, Ry, Rz, ProjectionJ, CombinedSingleGate
from .gate import CNOT, Swap, Rxx, Ryy, Rzz, Rxy, ReconfigurableBeamSplitter, Toffoli, Fredkin
from .gate import UAnyGate, LatentGate, HamiltonianGate, Barrier
from .layer import Observable, U3Layer, XLayer, YLayer, ZLayer, HLayer, RxLayer, RyLayer, RzLayer
from .layer import CnotLayer, CnotRing
from .qmath import multi_kron, partial_trace, amplitude_encoding, measure, expectation
from .qmath import meyer_wallach_measure
from .state import QubitState, MatrixProductState

from .mbqc import Pattern

from .photonic import permanent, takagi, hafnian, torontonian
from .photonic import FockState, GaussianState, BosonicState, CatState
from .photonic import QumodeCircuit, QumodeCircuitTDM, Clements, GaussianBosonSampling
from .photonic import UnitaryMapper, UnitaryDecomposer, DrawClements
