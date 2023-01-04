import itertools
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Dataset

from utils import make_query_image, ratio_preserving_resize

# Dataset parsing routines were taken from https://github.com/YoYo000/MVSNet/


def get_view_pairs(file_name, image_files, cams_files, depth_files):
    view_pairs = []
    with open(file_name) as file:
        lines = file.readlines()
        for line in lines[1:]:
            if len(line) > 3:
                tokens = line.split()
                pair_files = []
                for token in tokens[1::2]:
                    img_id = token.zfill(8)
                    for img_file_name, cam_file_name, depth_file_name in zip(
                        image_files, cams_files, depth_files
                    ):
                        text_name = str(img_file_name)
                        if img_id in text_name and "mask" not in text_name:
                            pair_files.append(
                                (img_file_name, cam_file_name, depth_file_name)
                            )
                pairs = itertools.permutations(pair_files, r=2)
                view_pairs.extend(pairs)
    return view_pairs


class DataCamera:
    def __init__(self):
        self.extrinsic = np.zeros((4, 4), dtype=np.float)
        self.intrinsic = np.zeros((3, 3), dtype=np.float)
        self.depth_min = 0
        self.depth_interval = 0
        self.depth_num = 0
        self.depth_max = 0

    def get_dir(self):
        r = self.get_rot_matrix()
        r_inv = np.linalg.inv(r)
        z = np.array([0, 0, 1, 1])
        dir = z.dot(r_inv.T)
        return dir[:3]

    def get_pos(self):
        t = self.extrinsic[:, 3]
        return t

    def get_pos_inv(self):
        r = self.get_rot_matrix()
        r_inv = np.linalg.inv(r)
        t = self.extrinsic[:, 3]
        camera_pos = t.dot(r_inv.T)
        camera_pos[:3] *= -1
        return camera_pos[:3]

    def get_rot_matrix(self):
        r = np.eye(4)
        r[0:3, 0:3] = self.extrinsic[0:3, 0:3]
        return r

    def project_points(self, coordinates_3d):
        coordinates_cam = coordinates_3d.dot(self.extrinsic.T)
        coordinates_cam = coordinates_cam / coordinates_cam[:, [3]]

        intrinsic_ex = np.pad(
            self.intrinsic,
            ((0, 0), (0, 1)),
            "constant",
            constant_values=((0, 0), (0, 0)),
        )
        coordinates_2d = coordinates_cam.dot(intrinsic_ex.T)
        coordinates_2d = coordinates_2d / coordinates_2d[:, [2]]
        return coordinates_2d, coordinates_cam[:, [2]]

    def back_project_points(self, coordinates_2d, depth):
        # from pixel to camera space
        intrinsic_inv = np.linalg.inv(self.intrinsic)
        coordinates_2d = coordinates_2d * depth
        coordinates_cam = coordinates_2d.dot(intrinsic_inv.T)  # [x, y, z]

        # make homogeneous
        coordinates_cam = np.hstack(
            [coordinates_cam, np.ones_like(coordinates_cam[:, [0]])]
        )

        # from camera to world space
        r = self.get_rot_matrix()
        r_inv = np.linalg.inv(r)
        t = self.extrinsic[:, 3]

        coordinates_cam[:, :3] -= t[:3]
        coordinates_world = coordinates_cam.dot(r_inv.T)

        return coordinates_world


def load_camera_matrices(file_name):
    with open(file_name) as file:
        camera = DataCamera()
        words = file.read().split()
        assert len(words) == 31
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                camera.extrinsic[i][j] = words[extrinsic_index]

        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                camera.intrinsic[i][j] = words[intrinsic_index]

        camera.depth_min = float(words[27])
        camera.depth_interval = float(words[28])
        camera.depth_num = int(float(words[29]))
        camera.depth_max = float(words[30])
        return camera


def load_pfm(file_name):
    with open(file_name, mode="rb") as file:
        header = file.readline().decode("UTF-8").rstrip()

        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("UTF-8"))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")
        scale = float((file.readline()).decode("UTF-8").rstrip())
        if scale < 0:  # little-endian
            data_type = "<f"
        else:
            data_type = ">f"  # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
        return data


class MVSDataset(Dataset):
    def __init__(
        self,
        path,
        image_size,
        resolution,
        depth_tolerance=0.005,
        seed=0,
        epoch_size=0,
        return_cams_info=False,
    ):
        self.path = path
        self.image_size = image_size
        self.items = []
        self.epoch_size = epoch_size
        self.resolution = resolution
        self.return_cams_info = return_cams_info
        self.depth_tolerance = depth_tolerance

        mvs_folders = list(Path(self.path).glob("*"))
        for folder_name in mvs_folders:
            images_folder = os.path.join(folder_name, "blended_images")
            image_files = list(Path(images_folder).glob("*[0-9].*"))
            image_files.sort()

            cams_folder = os.path.join(folder_name, "cams")
            cams_files = list(Path(cams_folder).glob("*cam.*"))
            cams_files.sort()

            depth_folder = os.path.join(folder_name, "rendered_depth_maps")
            depth_files = list(Path(depth_folder).glob("*.*"))
            depth_files.sort()

            pairs_file = os.path.join(folder_name, "cams", "pair.txt")
            if os.path.exists(pairs_file):
                view_pairs = get_view_pairs(
                    pairs_file, image_files, cams_files, depth_files
                )
                self.items.extend(view_pairs)

        self.rng = default_rng(seed)
        self.rng.shuffle(self.items)
        if epoch_size != 0:
            self.epoch_items = self.items[:epoch_size]

    def reset_epoch(self):
        self.rng.shuffle(self.items)
        if self.epoch_size != 0:
            self.epoch_items = self.items[: self.epoch_size]

    def __getitem__(self, index):
        (img_file_name1, cam_file_name1, depth_file_name1), (
            img_file_name2,
            cam_file_name2,
            depth_file_name2,
        ) = self.items[index]
        img1 = cv2.imread(str(img_file_name1))
        img_size_orig = np.array([img1.shape[1], img1.shape[0]])
        img1 = make_query_image(img1, self.image_size)
        img2 = cv2.imread(str(img_file_name2))
        img2 = make_query_image(img2, self.image_size)

        img1 = torch.from_numpy(img1)[None] / 255.0
        img2 = torch.from_numpy(img2)[None] / 255.0

        conf_matrix, camera1, camera2 = self.generate_groundtruth_confidence(
            cam_file_name1, depth_file_name1, cam_file_name2, depth_file_name2
        )
        conf_matrix = torch.from_numpy(conf_matrix)[None]

        if self.return_cams_info:
            return (
                img1,
                img2,
                conf_matrix,
                img_size_orig,
                camera1.intrinsic,
                camera1.get_rot_matrix(),
                camera1.get_pos_inv(),
                camera2.intrinsic,
                camera2.get_rot_matrix(),
                camera2.get_pos_inv(),
            )
        else:
            return img1, img2, conf_matrix

    def generate_groundtruth_confidence(
        self, cam_file_name1, depth_file_name1, cam_file_name2, depth_file_name2
    ):
        data_camera1 = load_camera_matrices(cam_file_name1)
        data_camera2 = load_camera_matrices(cam_file_name2)

        depth_hw1 = load_pfm(depth_file_name1)
        depth_hw2 = load_pfm(depth_file_name2)

        original_image_size = depth_hw1.shape

        w = original_image_size[1] // self.resolution
        h = original_image_size[0] // self.resolution

        coordinates_2d = np.array(list(np.ndindex(w, h))) * self.resolution
        coordinates_2d = np.hstack(
            [coordinates_2d, np.ones_like(coordinates_2d[:, [0]])]
        )

        depth1 = depth_hw1[coordinates_2d[:, 1], coordinates_2d[:, 0], np.newaxis]
        coordinates1_3d = data_camera1.back_project_points(coordinates_2d, depth1)

        coordinates2, depth2_computed = data_camera2.project_points(coordinates1_3d)

        # check depth consistency
        coordinates2_clipped = np.around(coordinates2)
        mask = np.where(
            np.all(
                (coordinates2_clipped[:, :2] >= (0, 0))
                & (
                    coordinates2_clipped[:, :2]
                    < (original_image_size[1], original_image_size[0])
                ),
                axis=1,
            )
        )
        coordinates2_clipped = coordinates2_clipped[mask].astype(np.long)
        coordinates2 = coordinates2[mask]
        coordinates_2d = coordinates_2d[mask]
        depth2_computed = depth2_computed[mask]

        depth2 = depth_hw2[
            coordinates2_clipped[:, 1], coordinates2_clipped[:, 0], np.newaxis
        ]
        depth2[depth2 == 0.0] = np.finfo(depth2.dtype).max
        depth_consistency_mask, _ = np.where(
            np.absolute((depth2 - depth2_computed) / depth2) < self.depth_tolerance
        )

        coordinates2 = coordinates2[depth_consistency_mask]
        coordinates_2d = coordinates_2d[depth_consistency_mask]

        # filter image coordinates to satisfy the grid property
        region_threshold = self.resolution / 3  # pixels
        grid_mask = np.where(
            np.all(
                (self.resolution - coordinates2[:, :2] % self.resolution)
                <= region_threshold,
                axis=1,
            )
        )
        coordinates2 = coordinates2[grid_mask]
        coordinates_2d = coordinates_2d[grid_mask]

        # scale coordinates to the training size
        scale_w = self.image_size[0] / original_image_size[1]
        scale_h = self.image_size[1] / original_image_size[0]

        # scale the first image coordinates
        coordinates1 = coordinates_2d.astype(np.float)
        coordinates1[:, :2] *= np.array([scale_w, scale_h])
        coordinates1[:, :2] /= self.resolution
        coordinates1 = np.around(coordinates1)

        # scale the second image coordinates
        coordinates2[:, :2] *= np.array([scale_w, scale_h])
        coordinates2[:, :2] /= self.resolution
        coordinates2 = np.around(coordinates2)

        # check bounds correctness
        w = self.image_size[0] // self.resolution
        h = self.image_size[1] // self.resolution

        mask = np.where(
            np.all(
                (coordinates2[:, :2] >= (0, 0)) & (coordinates2[:, :2] < (w, h)), axis=1
            )
        )
        coordinates2 = coordinates2[mask][:, :2]
        coordinates1 = coordinates1[mask][:, :2]

        # fill the confidence matrix
        coordinates1 = coordinates1[:, 1] * w + coordinates1[:, 0]
        coordinates2 = coordinates2[:, 1] * w + coordinates2[:, 0]

        conf_matrix = np.zeros((h * w, h * w), dtype=float)
        conf_matrix[coordinates1.astype(np.long), coordinates2.astype(np.long)] = 1.0

        return conf_matrix, data_camera1, data_camera2

    def __len__(self):
        if self.epoch_size != 0:
            return self.epoch_size
        return len(self.items)
