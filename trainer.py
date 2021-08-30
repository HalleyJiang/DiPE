# Based on the code of Monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/trainer.py

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


from __future__ import absolute_import, division, print_function

import numpy as np
import time
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


from utils.utils import *
from layers import *
import networks
import datasets
from IPython import embed

np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)


datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                 "kitti_odom": datasets.KITTIOdomDataset,
                 "kitti_depth": datasets.KITTIDepthDataset,
                 "cityscapes": datasets.CityScapesDataset}


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.epoch_training_times = []
        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)
            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_poses_to_predict_for=self.num_pose_frames-1)
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, self.opt.scheduler_milestones, self.opt.learning_rate_step_gamma)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        self.pose_metric_names = ["ate"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.val()
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

        avg_epoch_training_time = np.array(self.epoch_training_times).mean()
        print("average training time per epoch:", sec_to_hm_str(avg_epoch_training_time))

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        run_step = 0
        loss_sum = 0.0
        epoch_training_time = time.time()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            duration = time.time() - before_op_time

            run_step += 1
            loss_sum += losses["loss"].cpu().data

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, loss_sum/run_step)
                if ("depth_gt", 0) in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                if ("pose_gt", 0) in inputs:
                    self.compute_pose_losses(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)

            self.step += 1
        epoch_training_time = time.time() - epoch_training_time
        self.epoch_training_times.append(epoch_training_time)
        self.val()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}

        depth_encoder_features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs.update(self.models["depth"](depth_encoder_features))

        if self.use_pose_net:
            outputs.update(self.predict_pose(inputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_pose(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id != "s":

                    # To maintain ordering we always pass frames in temporal order
                    if frame_id > 0:
                        pose_inputs = [inputs["color_aug", 0, 0], inputs["color_aug", frame_id, 0]]
                    else:
                        pose_inputs = [inputs["color_aug", frame_id, 0], inputs["color_aug", 0, 0]]

                    pose_feats = self.models["pose_encoder"](torch.cat(pose_inputs, 1))

                    axisangle, translation = self.models["pose"](pose_feats)
                    outputs[("axisangle", 0, frame_id)] = axisangle
                    outputs[("translation", 0, frame_id)] = translation

                    outputs[("cam_T_cam", 0, frame_id)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(frame_id < 0))
        else:
            # Here we input all frames to the pose net (and predict all poses) together
            pose_inputs = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)
            pose_feats = self.models["pose_encoder"](pose_inputs)
            axisangle, translation = self.models["pose"](pose_feats)

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id != "s":
                    outputs[("axisangle", 0, frame_id)] = axisangle[:, i:i+1]
                    outputs[("translation", 0, frame_id)] = translation[:, i:i+1]
                    outputs[("cam_T_cam", 0, frame_id)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])
        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        run_step = 0
        losses_sum = {"loss": 0.0}
        losses_avg = {"loss": 0.0}

        for s in self.opt.scales:
            losses_sum["loss/"+str(s)] = 0.0
            losses_avg["loss/" + str(s)] = 0.0

        if self.opt.dataset == "kitti_odom":
            for name in self.pose_metric_names:
                losses_sum[name] = 0.0
                losses_avg[name] = 0.0
        else:
            for name in self.depth_metric_names:
                losses_sum[name] = 0.0
                losses_avg[name] = 0.0

        for batch_idx, inputs in enumerate(self.val_loader):
            run_step += 1
            with torch.no_grad():
                outputs, losses = self.process_batch(inputs)

                if ("depth_gt", 0) in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                if ("pose_gt", 0) in inputs:
                    self.compute_pose_losses(inputs, outputs, losses)

                for l, v in losses.items():
                    losses_sum[l] += v

        for l, v in losses_sum.items():
            losses_avg[l] = losses_sum[l] / run_step

        self.log("val", inputs, outputs, losses_avg)

        del inputs, outputs, losses, losses_sum, losses_avg

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            if self.opt.full_resolution_multiscale:
                source_scale = 0
            else:
                source_scale = scale

            disp = outputs[("disp", 0, scale)]
            if source_scale == 0:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    outputs[("depth", 0, scale)], inputs[("inv_K", source_scale)])

                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords.permute(0, 2, 3, 1)

                outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border")

                if not self.opt.disable_masking and not self.opt.disable_occlusion_mask_from_image_boundary:
                    outputs[("occlusion_mask_from_image_boundary", frame_id, scale)] = (
                            (outputs[("sample", frame_id, scale)][..., 0].unsqueeze(1) >= -1) *
                            (outputs[("sample", frame_id, scale)][..., 0].unsqueeze(1) <= 1) *
                            (outputs[("sample", frame_id, scale)][..., 1].unsqueeze(1) >= -1) *
                            (outputs[("sample", frame_id, scale)][..., 1].unsqueeze(1) <= 1)
                            ).float().detach()

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the photometric loss, smoothness loss, dynamic depth consistency loss for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0

            if self.opt.full_resolution_multiscale:
                source_scale = 0
            else:
                source_scale = scale

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]

            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                outputs[("reprojection_losses", frame_id, scale)] = \
                    self.compute_reprojection_loss(outputs[("color", frame_id, scale)], target)

            if self.opt.disable_masking:
                for frame_id in self.opt.frame_ids[1:]:
                    loss += outputs[("reprojection_losses", frame_id, scale)].mean()*(self.opt.scale_rate**source_scale)
            else:
                for frame_id in self.opt.frame_ids[1:]:
                    outputs[("mask", frame_id, scale)] = \
                        torch.ones(outputs[("reprojection_losses", frame_id, scale)].shape,
                                   device=outputs[("reprojection_losses", frame_id, scale)].device)

                if not self.opt.disable_occlusion_mask_from_image_boundary:
                    for frame_id in self.opt.frame_ids[1:]:
                        outputs[("mask", frame_id, scale)] *= \
                            outputs[("occlusion_mask_from_image_boundary", frame_id, scale)]

                if not self.opt.disable_occlusion_mask_from_photometric_error:
                    outputs[("min_reprojection_losses", scale)] = torch.stack(
                        [outputs[("reprojection_losses", frame_id, scale)] for frame_id in self.opt.frame_ids[1:]],
                        dim=-1).min(dim=-1)[0] + 1e-7

                    for frame_id in self.opt.frame_ids[1:]:
                        outputs[("occlusion_mask_from_photometric_error", frame_id, scale)] = (
                            outputs[("reprojection_losses", frame_id, scale)] <=
                            outputs[("min_reprojection_losses", scale)]).float().detach()

                        outputs[("mask", frame_id, scale)] *= \
                            outputs[("occlusion_mask_from_photometric_error", frame_id, scale)]

                if not self.opt.disable_stationary_mask_from_photometric_error:
                    for frame_id in self.opt.frame_ids[1:]:
                        source = inputs[("color", frame_id, source_scale)]
                        outputs[("identity_reprojection_losses", frame_id, scale)] = \
                            self.compute_reprojection_loss(source, target)

                        outputs[("stationary_mask_from_photometric_error", frame_id, scale)] = (
                            outputs[("reprojection_losses", frame_id, scale)] <
                            outputs[("identity_reprojection_losses", frame_id, scale)]).float().detach()

                        outputs[("mask", frame_id, scale)] *= \
                            outputs[("stationary_mask_from_photometric_error", frame_id, scale)]

                if not self.opt.disable_outlier_mask_from_photometric_error:
                    size = self.opt.batch_size
                    outputs[("mean_reprojection_losses", scale)] = torch.stack(
                        [outputs[("reprojection_losses", frame_id, scale)] for frame_id in self.opt.frame_ids[1:]],
                        dim=-1).view(size, 1, 1, -1).mean(dim=-1, keepdim=True)

                    outputs[("std_reprojection_losses", scale)] = torch.stack(
                        [outputs[("reprojection_losses", frame_id, scale)] for frame_id in self.opt.frame_ids[1:]],
                        dim=-1).view(size, 1, 1, -1).std(dim=-1, keepdim=True)

                    for frame_id in self.opt.frame_ids[1:]:
                        outputs[("outlier_mask_from_photometric_error", frame_id, scale)] = \
                            (outputs[("reprojection_losses", frame_id, scale)] >
                             (outputs[("mean_reprojection_losses", scale)] +
                              self.opt.low_ratio_outlier_mask_from_photometric_error *
                              outputs[("std_reprojection_losses", scale)])).float().detach() * \
                            (outputs[("reprojection_losses", frame_id, scale)] <
                             (outputs[("mean_reprojection_losses", scale)] +
                              self.opt.up_ratio_outlier_mask_from_photometric_error *
                              outputs[("std_reprojection_losses", scale)])).float().detach()

                        outputs[("mask", frame_id, scale)] *= \
                            outputs[("outlier_mask_from_photometric_error", frame_id, scale)]

                for frame_id in self.opt.frame_ids[1:]:
                    loss += (outputs[("mask", frame_id, scale)] *
                             outputs[("reprojection_losses", frame_id, scale)]).sum() / \
                            (outputs[("mask", frame_id, scale)].sum() + 1e-7)*(self.opt.scale_rate**source_scale)

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training
        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, self.dataset.default_full_res_shape, mode="bilinear", align_corners=False),
            self.dataset.min_depth, self.dataset.max_depth)
        depth_pred = depth_pred.detach()

        depth_gt = inputs[("depth_gt", 0)]
        mask = depth_gt > 0

        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, self.dataset.default_crop[0]:self.dataset.default_crop[1],
                        self.dataset.default_crop[2]:self.dataset.default_crop[3]] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=self.dataset.min_depth, max=self.dataset.max_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def compute_pose_losses(self, inputs, outputs, losses):
        """Compute pose metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """

        frame_ids = self.opt.frame_ids.copy()
        if 's' in frame_ids:
            frame_ids.remove('s')
        frame_ids.sort()

        pose_gt = []
        pose_pred = []
        for frame_id in frame_ids:

            pose_gt.append(inputs[("pose_gt", frame_id)])

            if frame_id == 0:
                pose_pred.append(torch.eye(4).repeat(self.opt.batch_size, 1, 1).to(self.device))
            else:
                pose_pred.append(transformation_from_parameters(outputs[("axisangle", 0, frame_id)].detach()[:, 0],
                                                                outputs[("translation", 0, frame_id)].detach()[:, 0],
                                                                invert=(frame_id > 0 or
                                                                        self.opt.pose_model_input == 'all')))
        local_pose_gt = []
        local_pose_pred = []
        for i in range(1, len(frame_ids)):
            local_pose_gt.append(torch.matmul(inverse_poses(pose_gt[i]), pose_gt[i-1]))
            local_pose_pred.append(torch.matmul(inverse_poses(pose_pred[i]), pose_pred[i - 1]))

        ate = compute_absolute_trajectory_error(local_pose_gt, local_pose_pred)

        pose_errors = []
        pose_errors.append(ate)
        for i, metric in enumerate(self.pose_metric_names):
            losses[metric] = np.array(pose_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses, vis_num=None):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        if not vis_num:
            vis_num = min(self.opt.batch_size//2, 4)

        for j in range(vis_num):
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
                        if not self.opt.disable_masking:
                            if not self.opt.disable_occlusion_mask_from_image_boundary:
                                writer.add_image(
                                    "occlusion_mask_from_image_boundary_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("occlusion_mask_from_image_boundary", frame_id, s)][j].data, self.step)
                            if not self.opt.disable_occlusion_mask_from_photometric_error:
                                writer.add_image(
                                    "occlusion_mask_from_photometric_error_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("occlusion_mask_from_photometric_error", frame_id, s)][j].data, self.step)
                            if not self.opt.disable_stationary_mask_from_photometric_error:
                                writer.add_image(
                                    "stationary_mask_from_photometric_error_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("stationary_mask_from_photometric_error", frame_id, s)][j].data, self.step)
                            if not self.opt.disable_outlier_mask_from_photometric_error:
                                writer.add_image(
                                    "outlier_mask_from_photometric_error_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("outlier_mask_from_photometric_error", frame_id, s)][j].data, self.step)

                            writer.add_image(
                                "mask_{}_{}/{}".format(frame_id, s, j),
                                outputs[("mask", frame_id, s)][j].data, self.step)

                            writer.add_image(
                                "reprojection_losses_{}_{}/{}".format(frame_id, s, j),
                                outputs[("reprojection_losses", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", 0, s)][j]), self.step)

            if ("depth_gt", 0) in inputs:

                depth = inputs[("depth_gt", 0)][j]
                depth[depth == 0] = 1000000

                writer.add_image("gt_disp/{}".format(j),
                                 normalize_image(1 / (depth + 0.000001)), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            if model_name == 'pose_encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            if os.path.exists(path):
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
