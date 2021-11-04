import torch
import torch.nn as nn

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


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
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching(config)
        self.data = dict()

    def backbone_forward(self, img0, img1):
        """
            'img0': (torch.Tensor): (N, 1, H, W)
            'img1': (torch.Tensor): (N, 1, H, W)
        """

        self.data.update({
            'image0': img0,
            'image1': img1,
            'bs': 1,  # batch size only for inference
            'hw0_i': (self.config["input_height"], self.config["input_width"]),
            'hw1_i': (self.config["input_height"], self.config["input_width"])
        })

        # we assume that data['hw0_i'] == data['hw1_i'] - faster & better BN convergence
        feats_c, feats_i, feats_f = self.backbone(torch.cat([self.data['image0'], self.data['image1']], dim=0))

        feats_c, feats_f = self.backbone.complete_result(feats_c, feats_i, feats_f)
        (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(self.data['bs']), feats_f.split(self.data['bs'])

        return feat_c0, feat_f0, feat_c1, feat_f1

    def forward(self, img0, img1):
        """ 
            'img0': (torch.Tensor): (N, 1, H, W)
            'img1': (torch.Tensor): (N, 1, H, W)
        """
        # 1. Local Feature CNN
        feat_c0, feat_f0, feat_c1, feat_f1 = self.backbone_forward(img0, img1)

        self.data.update({
            'hw0_c': (self.config["input_height"] // self.config["resolution"][0],
                      self.config["input_width"] // self.config["resolution"][0]),  # 8
            'hw1_c': (self.config["input_height"] // self.config["resolution"][0],
                      self.config["input_width"] // self.config["resolution"][0]),
            'hw0_f': (self.config["input_height"] // self.config["resolution"][1],
                      self.config["input_width"] // self.config["resolution"][1]),  # 2
            'hw1_f': (self.config["input_height"] // self.config["resolution"][1],
                      self.config["input_width"] // self.config["resolution"][1])
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        # feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        # feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        feat_c0 = torch.flatten(self.pos_encoding(feat_c0), 2, 3).permute(0, 2, 1)
        feat_c1 = torch.flatten(self.pos_encoding(feat_c1), 2, 3).permute(0, 2, 1)

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, self.data)

        # 4. fine-level refinement
        self.data.update({'W': self.config['fine_window_size']})
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, self.data['b_ids'],
                                                              self.data['i_ids'], self.data['j_ids'])
        if feat_f0_unfold.nelement() != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        #self.fine_matching(feat_f0_unfold, feat_f1_unfold, self.data)

        # return data
        #return self.data['mkpts0_f'], self.data['mkpts1_f'], self.data['mconf']

        return feat_f0_unfold, feat_f1_unfold

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
