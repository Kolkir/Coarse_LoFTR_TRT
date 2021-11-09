import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        self.temperature = config['dsmax_temperature']

        self.feature_num = 4800

    def forward(self, feat_c0, feat_c1, data):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5, [feat_c0, feat_c1])

        # sim_matrix_t = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        sim_matrix = torch.matmul(feat_c0, feat_c1.permute((0, 2, 1)))
        sim_matrix /= self.temperature
        # assert(torch.allclose(sim_matrix_t, sim_matrix, atol=1e-05))

        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        data.update({'conf_matrix': conf_matrix})

        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        # axes_lengths = {
        #     'h0c': data['hw0_c'][0],
        #     'w0c': data['hw0_c'][1],
        #     'h1c': data['hw1_c'][0],
        #     'w1c': data['hw1_c'][1]
        # }
        # 1. confidence thresholding
        # mask = conf_matrix > self.thr

        # mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c', **axes_lengths)
        # mask_view = mask.view(-1, axes_lengths['h0c'], axes_lengths['w0c'], axes_lengths['h1c'], axes_lengths['w1c'])
        # mask_border(mask_view, self.border_rm, False)

        # mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)', **axes_lengths)
        # NOTICE: mask - is already updated due to the fact that we use a view

        # 2. mutual nearest
        # mask = mask \
        #     & (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
        #     & (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask = conf_matrix # mask.to(dtype=torch.float)  # ONNX+TensorRT
        mask_v, all_j_ids = mask.max(dim=2)
        j_ids = all_j_ids.squeeze(0)
        b_ids = torch.zeros_like(j_ids, dtype=torch.long, device=mask.device)
        i_ids = torch.arange(self.feature_num, device=mask.device, dtype=torch.long)

        # mask_v, all_j_ids = mask.max(dim=2)
        # b_ids, i_ids = torch.where(mask_v)
        # j_ids = all_j_ids[b_ids, i_ids]

        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], torch.div(i_ids, data['hw0_c'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], torch.div(j_ids, data['hw1_c'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'mkpts0_c': mkpts0_c,  # [mconf != 0],
            'mkpts1_c': mkpts1_c,  # [mconf != 0],
            'mconf': mconf  # [mconf != 0]
        })

        return coarse_matches
