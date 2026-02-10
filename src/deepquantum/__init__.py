"""
This is the top level module from which all basic functions and classes of
DeepQuantum can be directly imported.
"""

__version__ = '4.4.0'


from . import (
    adjoint,
    ansatz,
    bitmath,
    channel,
    circuit,
    communication,
    cutting,
    distributed,
    gate,
    layer,
    mbqc,
    operation,
    optimizer,
    photonic,
    qasm3,
    qmath,
    qpd,
    state,
    utils,
)
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
    ShorCircuitFor15,
)
from .channel import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    GeneralizedAmplitudeDamping,
    Pauli,
    PhaseDamping,
    PhaseFlip,
)
from .circuit import DistributedQubitCircuit, QubitCircuit
from .communication import cleanup_distributed, setup_distributed
from .gate import (
    Barrier,
    CNOT,
    CombinedSingleGate,
    Fredkin,
    Hadamard,
    HamiltonianGate,
    Identity,
    ImaginarySwap,
    LatentGate,
    PauliX,
    PauliY,
    PauliZ,
    PhaseShift,
    ProjectionJ,
    ReconfigurableBeamSplitter,
    Rx,
    Rxx,
    Rxy,
    Ry,
    Ryy,
    Rz,
    Rzz,
    SDaggerGate,
    SGate,
    Swap,
    TDaggerGate,
    TGate,
    Toffoli,
    U3Gate,
    UAnyGate,
)
from .layer import CnotLayer, CnotRing, HLayer, Observable, RxLayer, RyLayer, RzLayer, U3Layer, XLayer, YLayer, ZLayer
from .mbqc import GraphState, Pattern, SubGraphState
from .photonic import (
    BosonicState,
    CatState,
    Clements,
    DistributedFockState,
    DistributedQumodeCircuit,
    DrawClements,
    FockState,
    FockStateBosonic,
    GKPState,
    GaussianBosonSampling,
    GaussianState,
    QumodeCircuit,
    QumodeCircuitTDM,
    UnitaryDecomposer,
    UnitaryMapper,
    hafnian,
    permanent,
    takagi,
    torontonian,
    williamson,
)
from .qasm3 import cir_to_qasm3, qasm3_to_cir
from .qmath import amplitude_encoding, expectation, measure, meyer_wallach_measure, multi_kron, partial_trace
from .state import DistributedQubitState, MatrixProductState, QubitState
