"""
Photonic Module
"""

from . import ansatz
from . import channel
from . import circuit
from . import decompose
from . import distributed
from . import draw
from . import gate
from . import hafnian_
from . import mapper
from . import measurement
from . import operation
from . import qmath
from . import state
from . import tdm
from . import torontonian_
from . import utils

from .ansatz import Clements, GaussianBosonSampling, GBS_Graph
from .channel import PhotonLoss
from .circuit import QumodeCircuit, DistributedQumodeCircuit
from .decompose import UnitaryDecomposer
from .draw import DrawClements
from .gate import PhaseShift, BeamSplitter, MZI, BeamSplitterTheta, BeamSplitterPhi, BeamSplitterSingle, UAnyGate
from .gate import Squeezing, Squeezing2, Displacement, DisplacementPosition, DisplacementMomentum
from .gate import QuadraticPhase, ControlledX, ControlledZ, CubicPhase, Kerr, CrossKerr, DelayBS, DelayMZI, Barrier
from .hafnian_ import hafnian
from .mapper import UnitaryMapper
from .measurement import Generaldyne, Homodyne, GeneralBosonic, PhotonNumberResolvingBosonic
from .qmath import permanent, takagi, xxpp_to_xpxp, xpxp_to_xxpp, quadrature_to_ladder, ladder_to_quadrature, williamson
from .state import FockState, GaussianState, BosonicState, CatState, GKPState, FockStateBosonic, DistributedFockState
from .tdm import QumodeCircuitTDM
from .torontonian_ import torontonian
from .utils import set_hbar, set_kappa, set_perm_chunksize

hbar = 2
kappa = 2 ** (-0.5)

perm_chunksize_dict = {}
