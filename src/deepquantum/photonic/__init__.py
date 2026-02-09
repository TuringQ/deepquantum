"""
Photonic Module
"""

from . import (
    ansatz,
    channel,
    circuit,
    decompose,
    distributed,
    draw,
    gate,
    hafnian_,
    mapper,
    measurement,
    operation,
    qmath,
    state,
    tdm,
    torontonian_,
    utils,
)
from .ansatz import Clements, GaussianBosonSampling, GBS_Graph
from .channel import PhotonLoss
from .circuit import DistributedQumodeCircuit, QumodeCircuit
from .decompose import UnitaryDecomposer
from .draw import DrawClements
from .gate import (
    Barrier,
    BeamSplitter,
    BeamSplitterPhi,
    BeamSplitterSingle,
    BeamSplitterTheta,
    ControlledX,
    ControlledZ,
    CrossKerr,
    CubicPhase,
    DelayBS,
    DelayMZI,
    Displacement,
    DisplacementMomentum,
    DisplacementPosition,
    Kerr,
    MZI,
    PhaseShift,
    QuadraticPhase,
    Squeezing,
    Squeezing2,
    UAnyGate,
)
from .hafnian_ import hafnian
from .mapper import UnitaryMapper
from .measurement import GeneralBosonic, Generaldyne, Homodyne, PhotonNumberResolvingBosonic
from .qmath import (
    cv_to_wigner,
    fock_to_wigner,
    ladder_to_quadrature,
    permanent,
    quadrature_to_ladder,
    schur_anti_symm_even,
    sqrtm_herm,
    takagi,
    williamson,
    xpxp_to_xxpp,
    xxpp_to_xpxp,
)
from .state import BosonicState, CatState, DistributedFockState, FockState, FockStateBosonic, GaussianState, GKPState
from .tdm import QumodeCircuitTDM
from .torontonian_ import torontonian
from .utils import set_hbar, set_kappa, set_perm_chunksize

hbar = 2
kappa = 2 ** (-0.5)

perm_chunksize_dict = {}
