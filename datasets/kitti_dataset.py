# Based on the code of Monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/datasets/kitti_dataset.py

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import scipy.misc
import numpy as np
import PIL.Image as pil
from utils.kitti_utils import generate_depth_map, read_calib_file
from .mono_dataset import MonoDataset

np.random.seed(10)

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    min_depth = 1e-3
    max_depth = 80.0

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(self.data_path,
                                     scene_name,
                                     "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_filename = os.path.join(self.data_path,
                                      scene_name,
                                      "proj_depth/groundtruth/image_02/{:010d}.png".format(int(frame_index)))

        return os.path.isfile(velo_filename) or os.path.isfile(depth_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    crop_rate = [0.40810811, 0.99189189, 0.03594771, 0.96405229]  # garg/eigen crop rate
    default_crop = [153, 371, 44, 1197]  # crop with default shape [375, 1242]
    default_full_res_shape = [375, 1242]

    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

        self.K = {}
        self.full_res_shape = {}

        self.dates = [date for date in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, date))]

        for date in self.dates:
            calib_filename = os.path.join(self.data_path, date, 'calib_cam_to_cam.txt')
            if os.path.isfile(calib_filename):
                cam2cam = read_calib_file(calib_filename)
                K = np.eye(4, dtype=np.float32)
                K[0:3, :] = cam2cam['P_rect_00'].reshape([3, 4])
                width = cam2cam['S_rect_00'][0]
                height = cam2cam['S_rect_00'][1]
                K[0, :] /= width
                K[1, :] /= height
                self.K[date] = K
                self.full_res_shape[date] = [height, width]

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(self.data_path,
                                     scene_name,
                                     "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def check_pose(self):
        return False

    def check_object_mask(self):
        return False

    def get_intrinsic(self, folder):
        date = folder.split('/')[0]
        return self.K[date]

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = scipy.misc.imresize(depth_gt, self.default_full_res_shape, "nearest")

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """

    crop_rate = [0.0, 1.0, 0.0, 1.0]
    default_crop = [0, 375, 0, 1242]
    default_full_res_shape = [375, 1242]

    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

        self.K = {}
        self.full_res_shape = {}

        self.dates = [date for date in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, date))]

        for date in self.dates:
            calib_filename = os.path.join(self.data_path, date, 'calib_cam_to_cam.txt')
            if os.path.isfile(calib_filename):
                cam2cam = read_calib_file(calib_filename)
                K = np.eye(4, dtype=np.float32)
                K[0:3, :] = cam2cam['P_rect_00'].reshape([3, 4])
                width = cam2cam['S_rect_00'][0]
                height = cam2cam['S_rect_00'][1]
                K[0, :] /= width
                K[1, :] /= height
                self.K[date] = K
                self.full_res_shape[date] = [height, width]

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        depth_filename = os.path.join(self.data_path,
                                      scene_name,
                                      "proj_depth/groundtruth/image_02/{:010d}.png".format(int(frame_index)))
        return os.path.isfile(depth_filename)

    def check_pose(self):
        return False

    def check_object_mask(self):
        return False

    def get_intrinsic(self, folder):
        date = folder.split('/')[0]
        return self.K[date]

    def get_image_path(self, folder, frame_index, side):
        if not self.is_test:
            f_str = "{:010d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(
                self.data_path,
                folder,
                "image_0{}/data".format(self.side_map[side]),
                f_str)
        else:
            f_str = "{:010d}{}".format(int(frame_index), self.img_ext) \
                if frame_index.isdigit() else frame_index+self.img_ext
            image_path = os.path.join(
                self.data_path,
                folder,
                "image",
                f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.default_full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

        self.K = {}
        self.full_res_shape = {}

        self.folders = ["{:02d}".format(folder) for folder in range(22)]

        for folder in self.folders:
            calib_filename = os.path.join(self.data_path, "sequences", folder, 'calib.txt')
            if os.path.isfile(calib_filename):
                cam2cam = read_calib_file(calib_filename)
                K = np.eye(4, dtype=np.float32)
                K[0:3, :] = cam2cam['P0'].reshape([3, 4])
                image = pil.open(os.path.join(self.data_path, "sequences", folder, "image_2/000000.png"))
                width, height = image.size
                K[0, :] /= width
                K[1, :] /= height
                self.K[folder] = K
                self.full_res_shape[folder] = [height, width]

    def check_depth(self):
        return False

    def check_pose(self):
        line = self.filenames[0].split()
        if line[0].isdigit():
            scene_name = int(line[0])
        else:
            return False
        frame_index = int(line[1])

        f_str = "{:06d}.txt".format(frame_index)
        pose_filename = os.path.join(self.data_path,
                                     "sequences/{:02d}".format(scene_name),
                                     "poses",
                                     f_str)

        return os.path.isfile(pose_filename)

    def check_object_mask(self):
        return False

    def get_intrinsic(self, folder):
        return self.K["{:02d}".format(int(folder))]

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path

    def get_pose(self, folder, frame_index, do_flip):
        f_str = "{:06d}.txt".format(frame_index)
        pose_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "poses",
            f_str)
        pose_gt = np.eye(4)
        pose_gt[0:3, :] = np.loadtxt(pose_path).reshape(3, 4)
        if do_flip:
            pose_gt[0:3, 1:] *= -1
        return pose_gt


class KITTIDepthTestDataset(KITTIDataset):
    """KITTI benchmark test dataset
    """

    crop_rate = [0.0, 1.0, 0.0, 1.0]
    default_crop = [0, 352, 0, 1216]
    default_full_res_shape = [352, 1216]

    def __init__(self, *args, **kwargs):
        super(KITTIDepthTestDataset, self).__init__(*args, **kwargs)

    def check_depth(self):
        return False

    def check_pose(self):
        return False

    def check_object_mask(self):
        return False

    def get_image_path(self, folder, frame_index, side=None):
        f_str = "{:010d}{}".format(int(frame_index), self.img_ext) \
                if frame_index.isdigit() else frame_index+self.img_ext
        image_path = os.path.join(self.data_path,
                                  folder,
                                  "image",
                                  f_str)
        return image_path

    def __getitem__(self, index):
        inputs = {}
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            image_frame_index = line[1] + "_image_" + line[2]
        else:
            image_frame_index = line[1]

        inputs[("color", 0, -1)] = self.get_color(folder, image_frame_index, None, False)
        color_aug = (lambda x: x)
        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        return inputs
