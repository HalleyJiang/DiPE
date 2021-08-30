from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class CityScapesDataset(MonoDataset):
    """Class for CityScapes dataset loaders
    """

    fx = 2262.52
    fy = 2265.3017905988554
    cx = 1096.98
    cy = 513.137
    base_line = 0.209313

    img_w = 2048
    img_h = 1024
    logo_h = 256

    full_res_shape = [img_h, img_w]
    default_full_res_shape = [img_h - logo_h, img_w]
    crop_rate = [0.0, 1.0, 0.0, 1.0]
    default_crop = [0, img_h - logo_h, 0, img_w]
    focus_length_x_baseline = fx * base_line

    min_depth = 1e-3
    max_depth = 80.0

    def __init__(self, *args, **kwargs):
        super(CityScapesDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[self.fx / self.img_w, 0, 0.5, 0],
                           [0, self.fy / (self.img_h-self.logo_h), self.cy / (self.img_h-self.logo_h), 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.side_map = {"l": "left", "r": "right"}

    def get_color(self, folder, frame_index, side, do_flip):

        seq_id = frame_index//1000000
        fram_id = frame_index % 1000000
        city = folder.split("/")[1]
        image_name = city + "_" + '{:06d}'.format(seq_id) + "_" + '{:06d}'.format(fram_id) + \
                     "_" + self.side_map[side] + "Img8bit" + self.img_ext

        folder =self.side_map[side] + "Img8bit_sequence/" + folder

        image_path = os.path.join(self.data_path, folder, image_name)
        color = self.loader(image_path)
        color = color.crop((0, 0, self.default_full_res_shape[1], self.default_full_res_shape[0]))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_intrinsic(self, folder):
        return self.K

    def check_depth(self):
        if self.is_train:
            return False
        line = self.filenames[0].split()
        city = line[0].split("/")[1]
        frame_index = int(line[1])
        seq_id = frame_index // 1000000
        fram_id = frame_index % 1000000

        disparity_folder = "disparity/" + line[0]
        disparity_name = city + "_" + '{:06d}'.format(seq_id) + "_" + '{:06d}'.format(fram_id) + \
                     "_" + "disparity" + ".png"
        disparity_path = os.path.join(self.data_path, disparity_folder, disparity_name)

        return os.path.isfile(disparity_path)

    def get_depth(self, folder, frame_index, side, do_flip):

        seq_id = frame_index//1000000
        fram_id = frame_index % 1000000
        city = folder.split("/")[1]

        disparity_folder = "disparity/" + folder
        disparity_name = city + "_" + '{:06d}'.format(seq_id) + "_" + '{:06d}'.format(fram_id) + \
                     "_" + "disparity" + ".png"
        disparity_path = os.path.join(self.data_path, disparity_folder, disparity_name)

        disparity_gt = pil.open(disparity_path)
        disparity_gt = disparity_gt.crop((0, 0, self.default_full_res_shape[1], self.default_full_res_shape[0]))
        disparity_gt = (np.array(disparity_gt).astype(np.float32)-1)/256

        depth_gt = np.zeros((self.default_full_res_shape[0], self.default_full_res_shape[1]), dtype=np.float32)
        val_mask = disparity_gt > 0
        depth_gt[val_mask] = self.focus_length_x_baseline/disparity_gt[val_mask]

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def check_object_mask(self):
        if self.is_train:
            return False
        line = self.filenames[0].split()
        city = line[0].split("/")[1]
        frame_index = int(line[1])
        seq_id = frame_index // 1000000
        fram_id = frame_index % 1000000

        label_folder = "gtFine/" + line[0]
        label_name = city + "_" + '{:06d}'.format(seq_id) + "_" + '{:06d}'.format(fram_id) + \
                     "_" + "gtFine_labelIds" + ".png"
        label_path = os.path.join(self.data_path, label_folder, label_name)

        if not os.path.isfile(label_path):
            return False
        label = pil.open(label_path)
        label = np.asarray(label)

        if label.max() <= 3:
            return False
        else:
            return True

    def get_object_mask(self, folder, frame_index, side, do_flip):

        seq_id = frame_index//1000000
        fram_id = frame_index % 1000000
        city = folder.split("/")[1]

        label_folder = "gtFine/" + folder
        label_name = city + "_" + '{:06d}'.format(seq_id) + "_" + '{:06d}'.format(fram_id) + \
                     "_" + "gtFine_labelIds" + ".png"
        label_path = os.path.join(self.data_path, label_folder, label_name)

        label_gt = pil.open(label_path)
        label_gt = label_gt.crop((0, 0, self.default_full_res_shape[1], self.default_full_res_shape[0]))
        label_gt = np.asarray(label_gt)

        # 24: person, 25: rider, 26: car, 27: truck, 28: bus, 29: caravan,
        # 30: trailer, 31: train, 32: motorcycle, 33: bicycle
        object_mask = (label_gt > 23) * (label_gt < 33)
        object_mask = object_mask.astype("uint8")

        if do_flip:
            object_mask = np.fliplr(object_mask)

        return object_mask

    def check_pose(self):
        return False

