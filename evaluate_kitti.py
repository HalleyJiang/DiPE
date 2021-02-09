# Based on the code of Monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py
# https://github.com/nianticlabs/monodepth2/blob/master/evaluate_pose.py

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth, transformation_from_parameters
from options import MonodepthOptions
import datasets
import networks

from utils.utils import readlines
from utils.kitti_utils import export_gt_depths
from utils.evaluation_utils import *

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


datasets_dict = {"eigen": datasets.KITTIRAWDataset,
                 "eigen_benchmark": datasets.KITTIRAWDataset,
                 "benchmark": datasets.KITTIDepthTestDataset,
                 "odom_09": datasets.KITTIOdomDataset,
                 "odom_10": datasets.KITTIOdomDataset}


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

        if opt.eval_split == "benchmark":
            val_sel_filenames = readlines(os.path.join(splits_dir, opt.eval_split, "val_selection_files.txt"))

            val_sel_dataset = datasets_dict[opt.eval_split](opt.data_path, val_sel_filenames,
                                                            encoder_dict['height'], encoder_dict['width'],
                                                            [0], 4, is_train=False, img_ext=img_ext)

            val_sel_dataloader = DataLoader(val_sel_dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                            pin_memory=True, drop_last=False)
            val_sel_pred_disps = []

            with torch.no_grad():
                for data in val_sel_dataloader:
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

                    val_sel_pred_disps.append(pred_disp)

                val_sel_pred_disps = np.concatenate(val_sel_pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "eigen_benchmark", "eigen_to_benchmark_ids.npy"))
            pred_disps = pred_disps[eigen_to_benchmark_ids]

        if opt.eval_split == "benchmark":
            val_sel_pred_disps = np.load(os.path.join(opt.ext_disp_to_eval.split('/')[:-1] +
                                                      "val_sel_" + opt.ext_disp_to_eval.split('/')[-1]))

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

        if opt.eval_split == "benchmark":
            output_path = os.path.join(
                opt.load_weights_folder, "val_sel_disps_{}_split.npy".format(opt.eval_split))
            print("-> Saving predicted disparities to ", output_path)
            np.save(output_path, val_sel_pred_disps)

    if opt.eval_split == "benchmark":
        pred_disps_test_only = pred_disps
        pred_disps = val_sel_pred_disps

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    if not os.path.exists(gt_path):
        gt_depths = export_gt_depths(opt.data_path, opt.eval_split)
    else:
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen" \
                or opt.eval_split == "eigen_benchmark":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    scale_factor = opt.pred_depth_scale_factor
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
        scale_factor *= med

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

    result_path = os.path.join(opt.load_weights_folder, "result_{}_split.txt".format(opt.eval_split))
    f = open(result_path, 'w+')
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)), file=f)

    mean_errors = np.array(errors).mean(0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"), file=f)
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\", file=f)
    print("\n-> Done!", file=f)
    f.close()

    if opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps_test_only)):
            disp_resized = cv2.resize(pred_disps_test_only[idx], (1216, 352))
            depth = scale_factor / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)


def evaluate_pose(opt):
    """Evaluate odometry on the KITTI dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    assert opt.eval_split == "odom_09" or opt.eval_split == "odom_10", \
        "eval_split should be either odom_9 or odom_10"

    device = torch.device("cpu" if opt.no_cuda else "cuda")

    sequence_id = int(opt.eval_split.split("_")[-1])

    if opt.pose_model_input == "pairs":
        opt.frame_ids = [1, 0]  # pose network only takes two frames as input
        num_poses = 1
        filenames = readlines(
            os.path.join(os.path.dirname(__file__), "splits", "odom",
                         "test_files_{}_{:02d}.txt".format("pairs", sequence_id)))
    else:
        opt.frame_ids = [i for i in opt.frame_ids if i != "s"]
        num_poses = len(opt.frame_ids) - 1
        filenames = readlines(
            os.path.join(os.path.dirname(__file__), "splits", "odom",
                         "test_files_{}_{:02d}.txt".format("all"+str(num_poses+1), sequence_id)))

    img_ext = '.png' if opt.png else '.jpg'
    dataset = datasets_dict[opt.eval_split](opt.data_path, filenames, opt.height, opt.width,
                                            opt.frame_ids, 4, is_train=False, img_ext=img_ext)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder = networks.ResnetEncoder(opt.num_layers, False, num_poses+1)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, num_poses, 1)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_encoder.to(device)
    pose_encoder.eval()
    pose_decoder.to(device)
    pose_decoder.eval()

    pred_poses = []
    flip_pred_poses = []

    print("-> Computing pose predictions")

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)

            if opt.post_process:
                # Left-Right Flip as Post-processing to further improve accuracy of pose estimation
                all_color_aug = torch.cat((all_color_aug, torch.flip(all_color_aug, [3])), 0)

            features = pose_encoder(all_color_aug)
            axisangle, translation = pose_decoder(features)


            if opt.post_process:
                N = axisangle.shape[0] // 2
                pred_poses.append(
                    transformation_from_parameters(axisangle[:N].view(N*num_poses, 1, 3),
                                                   translation[:N].view(N*num_poses, 1, 3),
                                                   invert=True).cpu().numpy().reshape(N, num_poses, 4, 4))
                flip_pred_poses.append(
                    transformation_from_parameters(axisangle[N:].view(N*num_poses, 1, 3),
                                                   translation[N:].view(N*num_poses, 1, 3),
                                                   invert=True).cpu().numpy().reshape(N, num_poses, 4, 4))
            else:
                N = axisangle.shape[0]
                pred_poses.append(
                    transformation_from_parameters(axisangle.view(N*num_poses, 1, 3),
                                                   translation.view(N*num_poses, 1, 3),
                                                   invert=True).cpu().numpy().reshape(N, num_poses, 4, 4))

    pred_poses = np.concatenate(pred_poses)

    if opt.post_process:
        flip_pred_poses = np.concatenate(flip_pred_poses)
        flip_pred_poses[:, :, 1:3, 0] *= -1
        flip_pred_poses[:, :, 0, 1:] *= -1
        pred_poses = average_poses(np.array([pred_poses, flip_pred_poses]))

    gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i]))
    gt_local_poses = np.expand_dims(np.array(gt_local_poses), axis=1)

    ATEs = []
    REs = []
    num_frames = gt_global_poses.shape[0]
    track_length = 5
    for i in range(0, num_frames - track_length):
        gt_odometry = local_poses_to_odometry(gt_local_poses[i:i + track_length - 1])
        pred_odometry = local_poses_to_odometry(pred_poses[i:i + track_length - num_poses])
        ATE, RE = compute_pose_error(gt_odometry, pred_odometry)
        ATEs.append(ATE)
        REs.append(RE)

    print("\n Trajectory error: \n"
          "    ATE: {:0.4f}, std: {:0.4f} \n"
          "    RE: {:0.4f}, std: {:0.4f}  \n ".format(np.mean(ATEs), np.std(ATEs), np.mean(REs), np.std(REs)))

    # compute the global monocular visual odometry and save it
    global_pred_odometry = local_poses_to_odometry(pred_poses)

    save_filename = opt.eval_split
    if opt.post_process:
        save_filename = save_filename + "_pp"
    save_path = os.path.join(opt.load_weights_folder, save_filename+".txt")
    np.savetxt(save_path, global_pred_odometry[:, :-1, :].reshape(global_pred_odometry.shape[0], -1), delimiter=' ', fmt='%1.8e')
    print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()
    if opt.eval_split in ["eigen", "eigen_benchmark", "benchmark"]:
        evaluate_depth(opt)
    elif opt.eval_split in ["odom_09", "odom_10"]:
        evaluate_pose(opt)
