# Based on the code of Monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py

from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
import argparse

from layers import disp_to_depth, transformation_from_parameters
import datasets
import networks
from options import MonodepthOptions

from utils.utils import readlines
from utils.kitti_utils import export_gt_depths, load_labels
from utils.evaluation_utils import *

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

datasets_dict = {"kitti_eigen": datasets.KITTIRAWDataset}


def evaluate_depth(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    device = torch.device("cpu" if opt.no_cuda else "cuda")

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        depth_encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        depth_decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(depth_encoder_path)

        img_ext = '.png' if opt.png else '.jpg'
        dataset = datasets_dict[opt.eval_split](opt.data_path, filenames,
                                                encoder_dict['height'], encoder_dict['width'],
                                                [0], 4, is_train=False, img_ext=img_ext)

        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        depth_encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(depth_encoder.num_ch_enc)

        model_dict = depth_encoder.state_dict()
        depth_encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(depth_decoder_path))

        depth_encoder.to(device)
        depth_encoder.eval()
        depth_decoder.to(device)
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].to(device)

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(depth_encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0, 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    if not os.path.exists(gt_path):
        gt_depths = export_gt_depths(opt.data_path, opt.eval_split)
    else:
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    labels = load_labels(opt.eval_split)

    print("-> Evaluating")

    if opt.eval_stereo:
        print("Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("Mono evaluation - using median scaling")

    ratios = []

    background_errors = []
    background_pixel_nums = []

    object_errors = []
    object_pixel_nums = []

    similar_errors = []
    similar_pixel_nums = []

    dissimilar_errors = []
    dissimilar_pixel_nums = []

    slow_errors = []
    slow_pixel_nums = []

    predestrain_errors = []
    predestrain_pixel_nums = []

    cyclist_errors = []
    cyclist_pixel_nums = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        background_mask = np.logical_and(labels[i] == 0, mask)
        background_pred_depth = pred_depth[background_mask]
        background_gt_depth = gt_depth[background_mask]

        object_mask = np.logical_and(labels[i] > 0, mask)
        object_pred_depth = pred_depth[object_mask]
        object_gt_depth = gt_depth[object_mask]

        similar_mask = np.logical_and(labels[i] == 1, mask)
        similar_pred_depth = pred_depth[similar_mask]
        similar_gt_depth = gt_depth[similar_mask]

        dissimilar_mask = np.logical_and(labels[i] == 2, mask)
        dissimilar_pred_depth = pred_depth[dissimilar_mask]
        dissimilar_gt_depth = gt_depth[dissimilar_mask]

        slow_mask = np.logical_and(labels[i] == 3, mask)
        slow_pred_depth = pred_depth[slow_mask]
        slow_gt_depth = gt_depth[slow_mask]

        predestrain_mask = np.logical_and(labels[i] == 4, mask)
        predestrain_pred_depth = pred_depth[predestrain_mask]
        predestrain_gt_depth = gt_depth[predestrain_mask]

        clylist_mask = np.logical_and(labels[i] == 5, mask)
        cyclist_pred_depth = pred_depth[clylist_mask]
        cyclist_gt_depth = gt_depth[clylist_mask]

        background_pred_depth *= opt.pred_depth_scale_factor
        object_pred_depth *= opt.pred_depth_scale_factor
        similar_pred_depth *= opt.pred_depth_scale_factor
        dissimilar_pred_depth *= opt.pred_depth_scale_factor
        slow_pred_depth *= opt.pred_depth_scale_factor
        predestrain_pred_depth *= opt.pred_depth_scale_factor
        cyclist_pred_depth *= opt.pred_depth_scale_factor

        if not opt.disable_median_scaling:
            ratio = np.median(background_gt_depth) / np.median(background_pred_depth)
            ratios.append(ratio)
            background_pred_depth *= ratio
            object_pred_depth *= ratio
            similar_pred_depth *= ratio
            dissimilar_pred_depth *= ratio
            slow_pred_depth *= ratio
            predestrain_pred_depth *= ratio
            cyclist_pred_depth *= ratio

        background_pred_depth[background_pred_depth < MIN_DEPTH] = MIN_DEPTH
        background_pred_depth[background_pred_depth > MAX_DEPTH] = MAX_DEPTH
        background_errors.append(compute_errors(background_gt_depth, background_pred_depth))
        background_pixel_nums.append(background_pred_depth.shape[0])

        object_pred_depth[object_pred_depth < MIN_DEPTH] = MIN_DEPTH
        object_pred_depth[object_pred_depth > MAX_DEPTH] = MAX_DEPTH
        object_errors.append(compute_errors(object_gt_depth, object_pred_depth))
        object_pixel_nums.append(object_pred_depth.shape[0])

        similar_pred_depth[similar_pred_depth < MIN_DEPTH] = MIN_DEPTH
        similar_pred_depth[similar_pred_depth > MAX_DEPTH] = MAX_DEPTH
        similar_errors.append(compute_errors(similar_gt_depth, similar_pred_depth))
        similar_pixel_nums.append(similar_pred_depth.shape[0])

        dissimilar_pred_depth[dissimilar_pred_depth < MIN_DEPTH] = MIN_DEPTH
        dissimilar_pred_depth[dissimilar_pred_depth > MAX_DEPTH] = MAX_DEPTH
        dissimilar_errors.append(compute_errors(dissimilar_gt_depth, dissimilar_pred_depth))
        dissimilar_pixel_nums.append(dissimilar_pred_depth.shape[0])

        slow_pred_depth[slow_pred_depth < MIN_DEPTH] = MIN_DEPTH
        slow_pred_depth[slow_pred_depth > MAX_DEPTH] = MAX_DEPTH
        slow_errors.append(compute_errors(slow_gt_depth, slow_pred_depth))
        slow_pixel_nums.append(slow_pred_depth.shape[0])

        predestrain_pred_depth[predestrain_pred_depth < MIN_DEPTH] = MIN_DEPTH
        predestrain_pred_depth[predestrain_pred_depth > MAX_DEPTH] = MAX_DEPTH
        predestrain_errors.append(compute_errors(predestrain_gt_depth, predestrain_pred_depth))
        predestrain_pixel_nums.append(predestrain_pred_depth.shape[0])

        cyclist_pred_depth[cyclist_pred_depth < MIN_DEPTH] = MIN_DEPTH
        cyclist_pred_depth[cyclist_pred_depth > MAX_DEPTH] = MAX_DEPTH
        cyclist_errors.append(compute_errors(cyclist_gt_depth, cyclist_pred_depth))
        cyclist_pixel_nums.append(cyclist_pred_depth.shape[0])

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_background_errors = np.average(np.nan_to_num(np.asarray(background_errors)),
                                        axis=0, weights=np.asarray(background_pixel_nums))
    print("\n Background: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_background_errors.tolist()) + "\\\\")

    mean_object_errors = np.average(np.nan_to_num(np.asarray(object_errors)),
                                        axis=0, weights=np.asarray(object_pixel_nums))
    print("\n Moving Objects: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_object_errors.tolist()) + "\\\\")

    mean_similar_errors = np.average(np.nan_to_num(np.asarray(similar_errors)),
                                    axis=0, weights=np.asarray(similar_pixel_nums))
    print("\n Similar Moving Vehicles: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_similar_errors.tolist()) + "\\\\")

    mean_dissimilar_errors = np.average(np.nan_to_num(np.asarray(dissimilar_errors)),
                                    axis=0, weights=np.asarray(dissimilar_pixel_nums))
    print("\n Dissimilar Moving Vehicles: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_dissimilar_errors.tolist()) + "\\\\")

    mean_slow_errors = np.average(np.nan_to_num(np.asarray(slow_errors)),
                                   axis=0, weights=np.asarray(slow_pixel_nums))
    print("\n Slow Moving Vehicles: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_slow_errors.tolist()) + "\\\\")

    mean_predestrain_errors = np.average(np.nan_to_num(np.asarray(predestrain_errors)),
                                         axis=0, weights=np.asarray(predestrain_pixel_nums))
    print("\n Predestrains/Persons: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log",
                                                                  "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_predestrain_errors.tolist()) + "\\\\")

    mean_cyclist_errors = np.average(np.nan_to_num(np.asarray(cyclist_errors)),
                                     axis=0, weights=np.asarray(cyclist_pixel_nums))
    print("\n Cyclists/Bikers: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_cyclist_errors.tolist()) + "\\\\")

    print("\n-> Done!")

    print(sum(background_pixel_nums))
    print(sum(object_pixel_nums))
    print(sum(similar_pixel_nums))
    print(sum(dissimilar_pixel_nums))
    print(sum(slow_pixel_nums))
    print(sum(predestrain_pixel_nums))
    print(sum(cyclist_pixel_nums))

    result_path = os.path.join(opt.load_weights_folder, "moving_result_{}_split.txt".format(opt.eval_split))
    f = open(result_path, 'w+')
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)), file=f)

    print("\n Background: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log",
                                                        "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 7).format(*mean_background_errors.tolist()) + "\\\\", file=f)

    print("\n Moving Objects: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log",
                                                            "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 7).format(*mean_object_errors.tolist()) + "\\\\", file=f)

    print("\n Similar Moving Vehicles: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log",
                                                             "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 7).format(*mean_similar_errors.tolist()) + "\\\\", file=f)

    print("\n Dissimilar Moving Vehicles: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log",
                                                             "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 7).format(*mean_dissimilar_errors.tolist()) + "\\\\", file=f)

    print("\n Slow Moving Vehicles: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log",
                                                            "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 7).format(*mean_slow_errors.tolist()) + "\\\\", file=f)

    print("\n Predestrains/Persons: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log",
                                                                  "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 7).format(*mean_predestrain_errors.tolist()) + "\\\\", file=f)

    print("\n Cyclists/Bikers: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log",
                                                             "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 7).format(*mean_cyclist_errors.tolist()) + "\\\\", file=f)

    print("\n-> Done!", file=f)

    f.close()


if __name__ == "__main__":

    options = MonodepthOptions()
    opt = options.parse()

    evaluate_depth(opt)

