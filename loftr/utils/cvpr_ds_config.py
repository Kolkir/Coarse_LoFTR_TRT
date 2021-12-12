from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


_CN = CN()
_CN.BACKBONE_TYPE = 'ResNetFPN'
_CN.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 2)]
_CN.INPUT_WIDTH = 640
_CN.INPUT_HEIGHT = 480
_CN.INPUT_BATCH_SIZE = 1

# 1. LoFTR-backbone (local feature CNN) config
_CN.RESNETFPN = CN()
_CN.RESNETFPN.INITIAL_DIM = 128
_CN.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3

# 2. LoFTR-coarse module config
_CN.COARSE = CN()
_CN.COARSE.D_MODEL = 256
_CN.COARSE.D_FFN = 256
_CN.COARSE.NHEAD = 8
_CN.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.COARSE.TEMP_BUG_FIX = False

# 3. Coarse-Matching config
_CN.MATCH_COARSE = CN()

_CN.MATCH_COARSE.BORDER_RM = 2
_CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1

default_cfg = lower_config(_CN)
