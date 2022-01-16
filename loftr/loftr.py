import torch
import torch.nn as nn

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer
from .utils.coarse_matching import CoarseMatching


class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(self.config['input_batch_size'], config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'], config['coarse']['d_model'])

    def backbone_forward(self, img0, img1):
        """
            'img0': (torch.Tensor): (N, 1, H, W)
            'img1': (torch.Tensor): (N, 1, H, W)
        """

        # we assume that data['hw0_i'] == data['hw1_i'] - faster & better BN convergence
        feats_c, feats_f= self.backbone(torch.cat([img0, img1], dim=0))

        bs = self.config['input_batch_size']
        (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(bs), feats_f.split(bs)

        return feat_c0, feat_f0, feat_c1, feat_f1

    def forward(self, img0, img1):
        """ 
            'img0': (torch.Tensor): (N, 1, H, W)
            'img1': (torch.Tensor): (N, 1, H, W)
        """
        # 1. Local Feature CNN
        feat_c0, feat_f0, feat_c1, feat_f1 = self.backbone_forward(img0, img1)

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        # feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        # feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        feat_c0 = torch.flatten(self.pos_encoding(feat_c0), 2, 3).permute(0, 2, 1)
        feat_c1 = torch.flatten(self.pos_encoding(feat_c1), 2, 3).permute(0, 2, 1)

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1)

        # 3. match coarse-level
        conf_matrix, sim_matrix = self.coarse_matching(feat_c0, feat_c1)

        return conf_matrix, sim_matrix

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
