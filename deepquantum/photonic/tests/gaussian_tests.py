import torch
import numpy as np
import sys
sys.path.append("..")
from qumode import QumodeCircuit, GaussianState



n_modes = 4
batch_size = 1
## initialize a gaussian state
gs = GaussianState(batch_size, n_modes)

