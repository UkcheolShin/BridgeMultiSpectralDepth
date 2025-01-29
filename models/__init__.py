from .registry import MODELS

# Supervised Monocular Depth Network 
from .trainers.mono_depth.DORN import DORN
from .trainers.mono_depth.BTS import BTS
from .trainers.mono_depth.AdaBins import AdaBins
from .trainers.mono_depth.Midas import Midas
from .trainers.mono_depth.NewCRF import NewCRF

# Multi-Spectral Depth Network 
from .trainers.multispectral_depth.MSDepth import MSDepth
