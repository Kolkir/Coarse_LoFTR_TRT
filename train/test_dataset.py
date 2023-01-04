import argparse

import cv2
import numpy as np
from torch.utils.data import DataLoader
from trainer import tensor_to_image

from loftr.utils.cvpr_ds_config import default_cfg
from train.mvsdataset import MVSDataset
from train.settings import TrainSettings
from utils import get_coarse_match, make_student_config
from webcam import draw_features

parser = argparse.ArgumentParser(description="LoFTR knowledge distillation.")
parser.add_argument(
    "--path", type=str, default="/data_sets/BlendedMVS", help="Path to the dataset."
)

opt = parser.parse_args()
print(opt)

settings = TrainSettings()
batch_size = settings.batch_size // settings.batch_size_divider
student_cfg = make_student_config(default_cfg)
img_size = (student_cfg["input_width"], student_cfg["input_height"])
loftr_coarse_resolution = student_cfg["resolution"][0]
dataset = MVSDataset(opt.path, img_size, loftr_coarse_resolution, epoch_size=5000)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

win_name = "BlendedMVS dataset"
cv2.namedWindow(win_name)

for batch in dataloader:
    image1, image2, conf_matrix = batch
    image1 = tensor_to_image(image1)
    image2 = tensor_to_image(image2)
    conf_matrix = conf_matrix[0].cpu().numpy()

    mkpts0, mkpts1, mconf = get_coarse_match(
        conf_matrix, img_size[1], img_size[0], loftr_coarse_resolution
    )
    n_top = 20
    indices = np.argsort(mconf)[::-1]
    indices = indices[:n_top]
    mkpts0 = mkpts0[indices, :]
    mkpts1 = mkpts1[indices, :]

    draw_features(image1, mkpts0, img_size, color=(255, 255, 255))
    draw_features(image2, mkpts1, img_size, color=(255, 255, 255))

    res_img = np.hstack((image1, image2))

    cv2.imshow(win_name, res_img)
    key = cv2.waitKey(delay=0)
    if key == ord("q"):
        print("Quitting, 'q' pressed.")
        break
