"""
Photonic Module
"""

from collections import defaultdict

from . import ansatz
from . import channel
from . import circuit
from . import decompose
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
from .circuit import QumodeCircuit
from .decompose import UnitaryDecomposer
from .draw import DrawClements
from .gate import PhaseShift, BeamSplitter, MZI, BeamSplitterTheta, BeamSplitterPhi, BeamSplitterSingle, UAnyGate
from .gate import Squeezing, Squeezing2, Displacement, DisplacementPosition, DisplacementMomentum, DelayBS, DelayMZI
from .hafnian_ import hafnian
from .mapper import UnitaryMapper
from .measurement import Generaldyne, Homodyne
from .qmath import permanent, takagi, xxpp_to_xpxp, xpxp_to_xxpp, quadrature_to_ladder, ladder_to_quadrature
from .state import FockState, GaussianState
from .tdm import QumodeCircuitTDM
from .torontonian_ import torontonian
from .utils import set_hbar, set_kappa, set_perm_chunksize

hbar = 2
kappa = 2 ** (-0.5)

perm_chunksize_dict = defaultdict(lambda: None)
