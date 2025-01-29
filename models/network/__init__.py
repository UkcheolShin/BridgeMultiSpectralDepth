# The code for each network is migrated from their code base
# DORN: https://github.com/liviniuk/DORN_depth_estimation_Pytorch
# BTS: https://github.com/cleinc/bts
# MiDaS: https://github.com/isl-org/MiDaS
# AdaBins: https://github.com/shariqfarooq123/AdaBins
# NeWCRF: https://github.com/aliyun/NeWCRFs

# monocular depth network
from .dorn.dorn import DeepOrdinalRegression
from .bts.bts import BtsModel
from .midas import DPTDepthModel, MidasNet, MidasNet_small
from .adabin import UnetAdaptiveBins
from .newcrf import NewCRFDepth
