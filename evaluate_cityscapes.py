from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
import argparse

from layers import disp_to_depth
import datasets
import networks
from utils.utils import readlines
from utils.evaluation_utils import *

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


STEREO_SCALE_FACTOR = 10*datasets.CityScapesDataset.base_line


def evaluate_depth(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = datasets.CityScapesDataset.min_depth
    MAX_DEPTH = datasets.CityScapesDataset.max_depth

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    device = torch.device("cpu" if opt.no_cuda else "cuda")

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, "cityscapes", "test_files.txt"))
        depth_encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        depth_decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(depth_encoder_path)

        img_ext = '.png'
        dataset = datasets.CityScapesDataset(opt.data_path, filenames,
                                             encoder_dict['height'], encoder_dict['width'],
                                             [0], 4, is_train=False, img_ext=img_ext)

        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        depth_encoder = networks.ResnetEncoder(18, False)
        depth_decoder = networks.DepthDecoder(depth_encoder.num_ch_enc)

        model_dict = depth_encoder.state_dict()
        depth_encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(depth_decoder_path))

        depth_encoder.to(device)
        depth_encoder.eval()
        depth_decoder.to(device)
        depth_decoder.eval()

        pred_disps = []
        gt_depths = []
        object_masks = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].to(device)

                gt_depth = data[("depth_gt", 0)]
                gt_depth = gt_depth.cpu()[:, 0].numpy()
                gt_depths.append(gt_depth)

                object_mask = data[("object_mask", 0)]
                object_mask = object_mask.cpu()[:, 0].numpy()
                object_masks.append(object_mask)

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
        gt_depths = np.concatenate(gt_depths)
        object_masks = np.concatenate(object_masks)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format("cityscapes"))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    gt_path = os.path.join(splits_dir, "cityscapes", "gt_depths.npz")
    object_mask_path = os.path.join(splits_dir, "cityscapes", "object_mask.npz")
    if "gt_depths" in vars() and "object_masks" in vars():
        if not os.path.exists(gt_path):
            np.savez_compressed(gt_path, data=gt_depths)
        if not os.path.exists(object_mask_path):
            np.savez_compressed(object_mask_path, data=object_masks)
    else:
        if os.path.exists(gt_path):
            gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
        else:
            raise RuntimeError('Execute prediction once to save the gt depths and object masks first.')
        if os.path.exists(object_mask_path):
            object_masks = np.load(object_mask_path, fix_imports=True, encoding='latin1')["data"]
        else:
            raise RuntimeError('Execute prediction once to save the gt depths and object masks first.')

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

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        background_mask = np.logical_and(object_masks[i] == 0, mask)
        background_pred_depth = pred_depth[background_mask]
        background_gt_depth = gt_depth[background_mask]

        object_mask = np.logical_and(object_masks[i] > 0, mask)
        object_pred_depth = pred_depth[object_mask]
        object_gt_depth = gt_depth[object_mask]

        background_pred_depth *= opt.pred_depth_scale_factor
        object_pred_depth *= opt.pred_depth_scale_factor

        if not opt.disable_median_scaling:
            ratio = np.median(background_gt_depth) / np.median(background_pred_depth)
            ratios.append(ratio)
            background_pred_depth *= ratio
            object_pred_depth *= ratio

        background_pred_depth[background_pred_depth < MIN_DEPTH] = MIN_DEPTH
        background_pred_depth[background_pred_depth > MAX_DEPTH] = MAX_DEPTH
        background_errors.append(compute_errors(background_gt_depth, background_pred_depth))
        background_pixel_nums.append(background_pred_depth.shape[0])

        object_pred_depth[object_pred_depth < MIN_DEPTH] = MIN_DEPTH
        object_pred_depth[object_pred_depth > MAX_DEPTH] = MAX_DEPTH
        object_errors.append(compute_errors(object_gt_depth, object_pred_depth))
        object_pixel_nums.append(object_pred_depth.shape[0])

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
    print("\n Objects: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_object_errors.tolist()) + "\\\\")

    print("\n-> Done!")

    print(sum(background_pixel_nums))
    print(sum(object_pixel_nums))

    result_path = os.path.join(opt.load_weights_folder, "result_{}_split.txt".format("cityscapes"))
    f = open(result_path, 'w+')
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)), file=f)

    print("\n Background: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log",
                                                        "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 7).format(*mean_background_errors.tolist()) + "\\\\", file=f)

    print("\n Objects: \n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log",
                                                            "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 7).format(*mean_object_errors.tolist()) + "\\\\", file=f)
    print("\n-> Done!", file=f)

    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Monocular Depth Evaluation Options for CityScapes")

    parser.add_argument("--data_path",
                        type=str,
                        help="path of the training data",
                        required=True)

    parser.add_argument("--load_weights_folder",
                        type=str,
                        help="name of model to load")

    parser.add_argument("--eval_stereo",
                        help="if set evaluates in stereo mode",
                        action="store_true")
    parser.add_argument("--eval_mono",
                        help="if set evaluates in mono mode",
                        action="store_true")
    parser.add_argument("--disable_median_scaling",
                        help="if set disables median scaling in evaluation",
                        action="store_true")
    parser.add_argument("--pred_depth_scale_factor",
                        help="if set multiplies predictions by this number",
                        type=float,
                        default=1.0)
    parser.add_argument("--ext_disp_to_eval",
                        type=str,
                        help="optional path to a .npy disparities file to evaluate")
    parser.add_argument("--save_pred_disps",
                        help="if set saves predicted disparities",
                        action="store_true")
    parser.add_argument("--post_process",
                        help="if set will perform the flipping post processing "
                             "from the original monodepth paper",
                        action="store_true")

    parser.add_argument("--batch_size",
                        type=int,
                        help="batch size",
                        default=16)
    parser.add_argument("--num_workers",
                        type=int,
                        help="number of dataloader workers",
                        default=8)
    parser.add_argument("--png",
                        help="if set, trains from raw png files (instead of jpgs)",
                        action="store_true")
    parser.add_argument("--no_cuda",
                        help="if set disables CUDA",
                        action="store_true")

    parser.add_argument("--min_depth",
                        type=float,
                        help="minimum depth",
                        default=0.1)
    parser.add_argument("--max_depth",
                        type=float,
                        help="maximum depth",
                        default=100.0)

    opt = parser.parse_args()

    opt.height = 192
    opt.width = 512

    evaluate_depth(opt)

