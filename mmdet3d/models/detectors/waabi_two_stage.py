# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models import TwoStageDetector

from mmdet3d.core.bbox.transforms import bbox3d2result
from mmdet3d.models.detectors.vqvae import LidarVQGAN, VectorQuantizer
from mmdet3d.models.detectors.vqvit import LidarVQViT
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import Base3DDetector
from mmcv.ops.roi_align_rotated import RoIAlignRotated
from typing import Dict, List, Tuple, Optional
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss
import torch
from mmdet3d.core.post_processing import nms_bev
from mmcv.ops.box_iou_rotated import box_iou_rotated
from scipy.optimize import linear_sum_assignment

z_offset = 0.0
# z_offset = 1.6
# z_offset = 0.4


def _sample_logistic(shape, out=None):

    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    # U2 = out.resize_(shape).uniform_() if out is not None else th.rand(shape)

    return torch.log(U) - torch.log(1 - U)


def _sigmoid_sample(logits, tau=1):
    """
    Implementation of Bernouilli reparametrization based on Maddison et al. 2017
    """
    dims = logits.dim()
    logistic_noise = _sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    return torch.sigmoid(y / tau)


def gumbel_sigmoid(logits, tau=1, hard=False):

    # shape = logits.size()
    y_soft = _sigmoid_sample(logits, tau=tau)
    if hard:
        y_hard = torch.where(y_soft > 0.5, torch.ones_like(y_soft), torch.zeros_like(y_soft))
        y = y_hard.data - y_soft.data + y_soft
    else:
        y = y_soft
    return y


class Voxelizer(torch.nn.Module):
    """Voxelizer for converting Lidar point cloud to image"""

    def __init__(self, x_min, x_max, y_min, y_max, step, z_min, z_max, z_step):
        super().__init__()
        # self.x_min = config.x_min
        # self.x_max = config.x_max
        # self.y_min = config.y_min
        # self.y_max = config.y_max
        # self.step = config.step
        # self.z_min = config.z_min
        # self.z_max = config.z_max
        # self.z_step = config.z_step

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.step = step
        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step

        # self.h_min = config.h_min
        # self.h_max = config.h_max
        # self.h_step = config.h_step
        # self.add_intensity = config.add_intensity
        # self.add_ground = config.add_ground

        self.width = round((self.x_max - self.x_min) / self.step)
        self.height = round((self.y_max - self.y_min) / self.step)
        self.z_depth = round((self.z_max - self.z_min) / self.z_step)
        # self.h_depth = round((self.h_max - self.h_min) / self.h_step)
        self.depth = self.z_depth
        # if self.add_intensity:
        #     self.depth *= 2
        # if self.add_ground:
        #     self.depth += self.h_depth

    def voxelize_single(self, lidar, bev):
        """Voxelize a single lidar sweep into image frame
        Image frame:
        1. Increasing depth indices corresponds to increasing real world z
            values.
        2. Increasing height indices corresponds to decreasing real world y
            values.
        3. Increasing width indices corresponds to increasing real world x
            values.
        Args:
            lidar (torch.Tensor N x 4 or N x 5) x, y, z, intensity, height_to_ground (optional)
            bev (torch.Tensor D x H x W) D = depth, the bird's eye view
                raster to populate
        """
        # assert len(lidar.shape) == 2 and (lidar.shape[1] == 4 or lidar.shape[1] == 5) and lidar.shape[0] > 0
        # TODO (Quin) allow timestamps to be consumed by single lidar voxelization
        # assert not self.add_ground or (self.add_ground and lidar.shape[1] == 5)
        # 1 & 2. Convert points to tensor index location. Clamp z indices to
        # valid range.
        indices_h = torch.floor((self.y_max - lidar[:, 1]) / self.step).long()
        # indices_h = torch.floor((lidar[:, 1] - self.y_min) / self.step).long()
        indices_w = torch.floor((lidar[:, 0] - self.x_min) / self.step).long()
        indices_d = torch.floor((lidar[:, 2] - self.z_min) / self.z_step).long()
        # indices_d = torch.clamp(
        #     torch.floor((lidar[:, 2] - self.z_min) / self.z_step),
        #     0,
        #     self.z_depth - 1,
        # ).long()
        # 3. Remove points out of bound
        valid_mask = ~torch.any(
            torch.stack(
                [
                    indices_h < 0,
                    indices_h >= self.height,
                    indices_w < 0,
                    indices_w >= self.width,
                    indices_d < 0,
                    indices_d >= self.z_depth,
                ]
            ),
            dim=0,
        )
        indices_h = indices_h[valid_mask]
        indices_w = indices_w[valid_mask]
        indices_d = indices_d[valid_mask]
        # 4. Assign indices to 1
        bev[indices_d, indices_h, indices_w] = 1.0

        # # 5. Add intensity
        # if self.add_intensity:
        #     intensity = lidar[valid_mask, 3]

        #     indices_d_intensity = indices_d + self.z_depth
        #     bev.index_put_((indices_d_intensity, indices_h, indices_w), intensity, accumulate=True)
        #     intensity_norm = (
        #         torch.index_put(
        #             torch.zeros_like(bev),
        #             (indices_d_intensity, indices_h, indices_w),
        #             torch.ones_like(indices_h, dtype=torch.float),
        #             accumulate=True,
        #         )
        #         + 1e-8
        #     )
        #     bev[self.z_depth : self.z_depth * 2] /= intensity_norm[self.z_depth : self.z_depth * 2]

        # # 6. Add height channels
        # if self.add_ground:
        #     indices_d_height = (
        #         torch.clamp(
        #             torch.floor((lidar[valid_mask, 4] - self.h_min) / self.h_step),
        #             0,
        #             self.h_depth - 1,
        #         ).long()
        #         + (self.depth - self.h_depth)
        #     )
        #     bev[indices_d_height, indices_h, indices_w] = 1.0

    def forward(self, lidars: List[List[Tensor]]):
        """Voxelize multiple sweeps in the current vehicle frame into voxels
            in image frame
        Args:
            list(list(tensor)): B * T * tensor[N x 4],
                where B = batch_size, T = 5, N is variable,
                4 = [x, y, z, intensity]
        Returns:
            tensor: [B x D x H x W], B = batch_size, D = T * depth, H = height,
                W = width
        """
        batch_size = len(lidars)
        assert batch_size > 0 and len(lidars[0]) > 0
        num_sweep = len(lidars[0])

        bev = torch.zeros(
            (batch_size, num_sweep, self.depth, self.height, self.width),
            dtype=torch.float,
            device=lidars[0][0][0].device,
        )

        for b in range(batch_size):
            assert len(lidars[b]) == num_sweep
            for i in range(num_sweep):
                self.voxelize_single(lidars[b][i], bev[b][i])
        return bev.view(batch_size, num_sweep * self.depth, self.height, self.width)


def get_iou_loss(a, b, cross=False, max_cost=21.0):
    if cross:
        a = a.unsqueeze(1)
        b = b.unsqueeze(0)

    x_min = torch.max(a[..., 0] - a[..., 2] / 2.0, b[..., 0] - b[..., 2] / 2.0)
    x_max = torch.min(a[..., 0] + a[..., 2] / 2.0, b[..., 0] + b[..., 2] / 2.0)
    y_min = torch.max(a[..., 1] - a[..., 3] / 2.0, b[..., 1] - b[..., 3] / 2.0)
    y_max = torch.min(a[..., 1] + a[..., 3] / 2.0, b[..., 1] + b[..., 3] / 2.0)

    ol = x_max - x_min
    ow = y_max - y_min

    intersect = ol * ow
    union = a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - intersect

    mask = (a[..., 2] > 0) & (a[..., 3] > 0) & (b[..., 2] > 0) & (b[..., 3] > 0) & (ol > 0) & (ow > 0) & (union > 0)

    if cross:
        cost = max_cost * torch.ones((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        cost[mask] = -torch.log(intersect[mask] / union[mask])
        cost[cost > max_cost] = max_cost
        return cost
    else:
        if mask.any():
            loss = -torch.log(intersect[mask] / union[mask])
            return loss.sum()
        else:
            return torch.zeros(1, device=a.device, dtype=a.dtype).sum()


def get_3diou_loss(a, b, cross=False, max_cost=21.0):
    if cross:
        a = a.unsqueeze(1)
        b = b.unsqueeze(0)

    # xyzwlh

    x_min = torch.max(a[..., 0] - a[..., 3] / 2.0, b[..., 0] - b[..., 3] / 2.0)
    x_max = torch.min(a[..., 0] + a[..., 3] / 2.0, b[..., 0] + b[..., 3] / 2.0)
    y_min = torch.max(a[..., 1] - a[..., 4] / 2.0, b[..., 1] - b[..., 4] / 2.0)
    y_max = torch.min(a[..., 1] + a[..., 4] / 2.0, b[..., 1] + b[..., 4] / 2.0)

    zmax1 = a[..., 2] + 0.5 * a[..., 5]
    zmin1 = a[..., 2] - 0.5 * a[..., 5]
    zmax2 = b[..., 2] + 0.5 * b[..., 5]
    zmin2 = b[..., 2] - 0.5 * b[..., 5]

    ol = x_max - x_min
    ow = y_max - y_min
    z_overlap = torch.min(zmax1, zmax2) - torch.max(zmin1, zmin2)

    intersect = ol * ow * z_overlap
    union = a[..., 3] * a[..., 4] * a[..., 5] + b[..., 3] * b[..., 4] * b[..., 5] - intersect

    mask = (
        (a[..., 3] > 0)
        & (a[..., 4] > 0)
        & (a[..., 5] > 0)
        & (b[..., 3] > 0)
        & (b[..., 4] > 0)
        & (b[..., 5] > 0)
        & (ol > 0)
        & (ow > 0)
        & (union > 0)
        & (z_overlap > 0)
    )

    if cross:
        cost = max_cost * torch.ones((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        cost[mask] = -torch.log(intersect[mask] / union[mask])
        cost[cost > max_cost] = max_cost
        return cost
    else:
        if mask.any():
            loss = -torch.log(intersect[mask] / union[mask])
            return loss.sum()
        else:
            return torch.zeros(1, device=a.device, dtype=a.dtype).sum()


class DetectionLoss(nn.Module):
    def __init__(self, weights, focal, dontcare_iou=None, pos_iou=None):
        super().__init__()
        self.weights = weights
        self.alpha = focal["alpha"]
        self.gamma = focal["gamma"]
        self.dontcare_iou = dontcare_iou
        self.pos_iou = pos_iou

    @torch.jit.unused
    def forward(self, outputs: Dict[int, Dict[str, Tensor]], gt):
        loss, meta = 0, dict()
        if "ignore" in gt:
            num_bboxes = max(1, sum([(~x).sum().item() for x in gt["ignore"]]))
        else:
            num_bboxes = max(1, sum([len(x) for x in gt["bboxes"]]))

        loss_dict = {}
        # import ipdb; ipdb.set_trace()

        for c in [0]:
            for stage in ["propose", "refine"]:
                out: Dict[str, Tensor] = {
                    k.replace(stage + "_", ""): v for k, v in outputs[c].items() if k.startswith(stage)
                }
                if len(out) == 0:
                    continue

                if stage == "propose":
                    pos_idcs, neg_idcs, gt_idcs = self.matcher(out, gt, dontcare_iou=self.dontcare_iou)
                if stage == "refine":
                    pos_idcs, neg_idcs, gt_idcs = self.matcher(out, gt, pos_iou=self.pos_iou)

                cls = self.cls_loss(out, gt, pos_idcs, neg_idcs, gt_idcs, num_bboxes)
                iou = self.reg_loss(out, gt, pos_idcs, gt_idcs, num_bboxes)

                # import ipdb; ipdb.set_trace()

                # loss += self.weights["cls"] * cls + self.weights["iou"] * iou
                # meta[f"det/{c}/{stage}_cls"] = float(cls)
                # meta[f"det/{c}/{stage}_iou"] = float(iou)
                loss_dict[f"{stage}_cls_loss"] = self.weights["cls"] * cls
                loss_dict[f"{stage}_box_loss"] = self.weights["iou"] * iou[0]
                loss_dict[f"{stage}_iou_loss"] = self.weights["iou"] * iou[1]
                loss_dict[f"{stage}_angle_loss"] = self.weights["iou"] * iou[2]

                if stage == "refine" and "tracks" in out:
                    track = self.track_loss(out, gt, pos_idcs, gt_idcs, num_bboxes)

                    loss += self.weights["track"] * track
                    meta[f"det/{c}/{stage}_track"] = float(track)

                if stage == "refine" and "heading" in out:
                    heading = self.heading_loss(out, gt, pos_idcs, gt_idcs)
                    loss += self.weights["heading"] * heading
                    meta[f"det/{c}/{stage}_head"] = float(heading)
        return loss_dict

    @torch.no_grad()
    def matcher(self, out, gt, dontcare_iou=None, pos_iou=None):
        pos_idcs, gt_idcs = [], []
        zero = torch.zeros(0, device=gt["bboxes"][0].device, dtype=torch.int64)
        batch_size = len(out["bboxes"])
        for i in range(batch_size):
            if len(out["bboxes"][i]) == 0 or len(gt["bboxes"][i]) == 0:
                pos_idcs.append(zero.clone())
                gt_idcs.append(zero.clone())
                continue

            cost = 0.0

            logits = out["logits"][i][:, gt["cls_idcs"][i]]
            pos_cost = sigmoid_focal_loss(logits, torch.ones_like(logits), self.alpha, self.gamma, "none")
            neg_cost = sigmoid_focal_loss(logits, torch.zeros_like(logits), self.alpha, self.gamma, "none")
            cls_cost = pos_cost - neg_cost
            cost += self.weights["cls"] * cls_cost

            bboxes, gt_bboxes = out["bboxes"][i], gt["bboxes"][i]
            # if bboxes.shape[-1] == 5:
            #     iou_cost = get_iou_loss(bboxes[:, :4], gt_bboxes[:, [0, 1, 3, 4]], cross=True)
            # else:
            # iou_cost = get_iou_loss(bboxes[:, [0, 1, 3, 4]], gt_bboxes[:, [0, 1, 3, 4]], cross=True)
            iou_cost = get_3diou_loss(bboxes[:, :6], gt_bboxes[:, :6], cross=True)

            bboxes = bboxes.unsqueeze(1).repeat(1, len(gt_bboxes), 1)
            gt_bboxes = gt_bboxes.unsqueeze(0).repeat(len(bboxes), 1, 1)

            # if bboxes.shape[-1] == 5:
            #     box_cost = F.smooth_l1_loss(bboxes[:, :, :4], gt_bboxes[:, :, [0, 1, 3, 4]], reduction="none")
            # else:
            box_cost = F.smooth_l1_loss(bboxes[:, :, :6], gt_bboxes[:, :, :6], reduction="none")
            box_cost = box_cost.sum(2)

            theta = 2.0 * bboxes[:, :, -1]
            gt_theta = 2.0 * gt_bboxes[:, :, -1]
            sin_cost = F.smooth_l1_loss(theta.sin(), gt_theta.sin(), reduction="none")
            cos_cost = F.smooth_l1_loss(theta.cos(), gt_theta.cos(), reduction="none")
            angle_cost = sin_cost + cos_cost

            cost += self.weights["iou"] * (iou_cost + 0.1 * box_cost + angle_cost)

            idcs_a, idcs_b = linear_sum_assignment(cost.cpu())
            pos_idcs.append(torch.as_tensor(idcs_a, device=gt["bboxes"][0].device, dtype=torch.int64))
            gt_idcs.append(torch.as_tensor(idcs_b, device=gt["bboxes"][0].device, dtype=torch.int64))

        neg_idcs = []
        for i in range(batch_size):
            mask = torch.ones(len(out["logits"][i]), device=pos_idcs[i].device, dtype=torch.bool)
            mask[pos_idcs[i]] = False
            neg_idcs.append(torch.where(mask)[0])

        cls_idcs = [x.clone() for x in gt["cls_idcs"]]
        if "ignore" in gt:
            for i in range(batch_size):
                mask = ~(gt["ignore"][i][gt_idcs[i]])
                pos_idcs[i] = pos_idcs[i][mask]
                gt_idcs[i] = gt_idcs[i][mask]
                cls_idcs[i] = cls_idcs[i][mask]

        if dontcare_iou is not None:
            for i in range(batch_size):
                if len(pos_idcs[i]) == 0 or len(neg_idcs[i]) == 0:
                    continue
                # iou_mat = box_iou_rotated(out["bboxes"][i][pos_idcs[i]], out["bboxes"][i][neg_idcs[i]], clockwise=False)
                iou_mat = box_iou_rotated(
                    out["bboxes"][i][pos_idcs[i]][..., [0, 1, 3, 4, 6]],
                    out["bboxes"][i][neg_idcs[i]][..., [0, 1, 3, 4, 6]],
                    clockwise=True,
                )
                max_iou, _ = iou_mat.max(0)
                mask = max_iou < dontcare_iou
                neg_idcs[i] = neg_idcs[i][mask]

        if pos_iou is not None:
            pos_iou = torch.FloatTensor(pos_iou).to(gt["bboxes"][0].device)
            for i in range(batch_size):
                if len(pos_idcs[i]) == 0:
                    continue
                # iou_mat = box_iou_rotated(
                #     out["rois"][i][pos_idcs[i]], gt["bboxes"][i][gt_idcs[i]][..., [0, 1, 3, 4, 6]], clockwise=False
                # )
                iou_mat = box_iou_rotated(
                    out["rois"][i][pos_idcs[i]][..., [0, 1, 3, 4, 6]],
                    gt["bboxes"][i][gt_idcs[i]][..., [0, 1, 3, 4, 6]],
                    clockwise=True,
                )
                iou = iou_mat.diag()
                pos_iou_vec = pos_iou[cls_idcs[i]]
                mask = iou > pos_iou_vec
                pos_idcs[i] = pos_idcs[i][mask]
                gt_idcs[i] = gt_idcs[i][mask]
        return pos_idcs, neg_idcs, gt_idcs

    def cls_loss(self, out, gt, pos_idcs, neg_idcs, gt_idcs, num_bboxes):
        logits, labels = [], []
        batch_size = len(pos_idcs)
        for i in range(batch_size):
            pos_logits = out["logits"][i][pos_idcs[i]]
            neg_logits = out["logits"][i][neg_idcs[i]]

            pos_labels = torch.zeros_like(pos_logits)
            cls_idcs = gt["cls_idcs"][i][gt_idcs[i]]
            pos_labels[torch.arange(len(pos_labels), device=pos_logits.device), cls_idcs] = 1
            neg_labels = torch.zeros_like(neg_logits)

            logits += [pos_logits, neg_logits]
            labels += [pos_labels, neg_labels]
        logits = torch.cat(logits, 0)
        labels = torch.cat(labels, 0)
        loss = sigmoid_focal_loss(logits, labels, self.alpha, self.gamma, "sum") / num_bboxes
        return loss

    def reg_loss(self, out, gt, pos_idcs, gt_idcs, num_bboxes):
        batch_size = len(pos_idcs)
        bboxes = [out["bboxes"][i][pos_idcs[i]] for i in range(batch_size)]
        bboxes = torch.cat(bboxes, 0)
        gt_bboxes = [gt["bboxes"][i][gt_idcs[i]] for i in range(batch_size)]
        gt_bboxes = torch.cat(gt_bboxes, 0)

        # if bboxes.shape[-1] == 5:
        #     box_loss = F.smooth_l1_loss(bboxes[:, :4], gt_bboxes[:, [0, 1, 3, 4]], reduction="sum") / num_bboxes
        #     iou_loss = get_iou_loss(bboxes[:, :4], gt_bboxes[:, [0, 1, 3, 4]]) / num_bboxes
        # else:
        box_loss = F.smooth_l1_loss(bboxes[:, :6], gt_bboxes[:, :6], reduction="sum") / num_bboxes
        # iou_loss = get_iou_loss(bboxes[:, [0, 1, 3, 4]], gt_bboxes[:, [0, 1, 3, 4]]) / num_bboxes
        iou_loss = get_3diou_loss(bboxes[:, :6], gt_bboxes[:, :6]) / num_bboxes

        theta = 2.0 * bboxes[:, -1]
        gt_theta = 2.0 * gt_bboxes[:, -1]
        sin_loss = F.smooth_l1_loss(theta.sin(), gt_theta.sin(), reduction="sum")
        cos_loss = F.smooth_l1_loss(theta.cos(), gt_theta.cos(), reduction="sum")
        angle_loss = (sin_loss + cos_loss) / num_bboxes
        # return 0.1 * box_loss + iou_loss + angle_loss
        return 0.1 * box_loss, iou_loss, angle_loss

    def track_loss(self, out, gt, pos_idcs, gt_idcs, num_bboxes):
        batch_size = len(pos_idcs)

        tracks = out["tracks"]
        device = tracks.device
        tracks = [tracks[i, pos_idcs[i]] for i in range(batch_size) if pos_idcs[i].size(0) > 0]
        tracks = torch.cat(tracks, 0) if len(tracks) > 0 else torch.empty((0), device=device)

        gt_tracks = [gt["tracks"][i][gt_idcs[i]] for i in range(batch_size) if gt_idcs[i].size(0) > 0]
        gt_tracks = torch.cat(gt_tracks, 0) if len(gt_tracks) > 0 else torch.empty((0), device=device)

        gt_track_size = gt_tracks.size(0)
        track_size = tracks.size(0)

        if track_size > 0 and gt_track_size > 0:
            track_loss_xy = nn.functional.smooth_l1_loss(tracks[:, :, 0:2], gt_tracks[:, :, 0:2], reduction="none")
            gt_angle = 0.5 * torch.atan2(torch.sin(2 * gt_tracks[:, :, 2]), torch.cos(2 * gt_tracks[:, :, 2]))
            track_loss_angle = nn.functional.smooth_l1_loss(tracks[:, :, 2], gt_angle, reduction="none")
            track_loss = track_loss_xy.sum() + track_loss_angle.sum()
            track_loss = track_loss / num_bboxes
        else:
            track_loss = torch.tensor(0.0, device=device)

        if track_loss.isnan():
            track_loss = torch.tensor(0.0, device=device)

        return track_loss

    def heading_loss(self, out, gt, pos_idcs, gt_idcs):
        batch_size = len(pos_idcs)
        device = out["heading"].device
        headings = [out["heading"][i][pos_idcs[i]] for i in range(batch_size)]
        headings = torch.cat(headings, 0)
        if headings.size(0) == 0:
            return torch.tensor(0.0, device=device)

        yaws = [out["bboxes"][i][pos_idcs[i], 4] for i in range(batch_size)]
        yaws = torch.cat(yaws, 0)
        gt_yaws = [gt["bboxes"][i][gt_idcs[i], 4] for i in range(batch_size)]
        gt_yaws = torch.cat(gt_yaws, 0)

        # if yaw angle error is greater than 45 degrees, then we don't compute heading loss
        valid_idcs = torch.nonzero(wrap_angle_rad(yaws - gt_yaws, math.pi / 2).abs() <= (math.pi / 4))
        if valid_idcs.size(0) == 0:
            return torch.tensor(0.0, device=device)
        yaws = yaws[valid_idcs]
        gt_yaws = gt_yaws[valid_idcs]
        headings = headings[valid_idcs]

        gt_headings = torch.where(
            gt_yaws.sin().abs() < (1 / math.sqrt(2)), gt_yaws.cos() >= 0, gt_yaws.sin() >= 0
        ).float()
        loss = nn.functional.binary_cross_entropy_with_logits(headings, gt_headings, reduction="mean")
        return loss


def bev_reg_to_bboxes(reg: Tensor, bev_range: Tuple[float, float, float, float]) -> Tensor:
    N, C, H, W = reg.shape
    # assert C == 6
    x_min, x_max, y_min, y_max = bev_range

    bboxes = torch.empty((N, H, W, 7), device=reg.device, dtype=reg.dtype)
    x = x_min + (torch.arange(W, device=reg.device, dtype=reg.dtype) + 0.5) / W * (x_max - x_min)
    y = y_max - (torch.arange(H, device=reg.device, dtype=reg.dtype) + 0.5) / H * (y_max - y_min)
    bboxes[..., 0] = x.view(1, 1, W) + reg[:, 0]
    bboxes[..., 1] = y.view(1, H, 1) + reg[:, 1]
    bboxes[..., 2] = reg[:, 2]
    bboxes[..., 3] = reg[:, 3]
    bboxes[..., 4] = reg[:, 4]
    bboxes[..., 5] = reg[:, 5]
    bboxes[..., 6] = torch.atan2(reg[:, 6], reg[:, 7])
    bboxes = bboxes.view(N, H * W, 7)

    # bboxes = torch.empty((N, H, W, 5), device=reg.device, dtype=reg.dtype)
    # x = x_min + (torch.arange(W, device=reg.device, dtype=reg.dtype) + 0.5) / W * (x_max - x_min)
    # y = y_max - (torch.arange(H, device=reg.device, dtype=reg.dtype) + 0.5) / H * (y_max - y_min)
    # # y = y_min + (torch.arange(H, device=reg.device, dtype=reg.dtype) + 0.5) / H * (y_max - y_min)
    # bboxes[..., 0] = x.view(1, 1, W) + reg[:, 0]
    # bboxes[..., 1] = y.view(1, H, 1) + reg[:, 1]
    # bboxes[..., 2] = reg[:, 2]
    # bboxes[..., 3] = reg[:, 3]
    # bboxes[..., 4] = torch.atan2(reg[:, 4], reg[:, 5])
    # bboxes = bboxes.view(N, H * W, 5)
    return bboxes


def get_topk_idcs(logits: Tensor, k: int) -> Tensor:
    # logits: (N, K, C)
    scores, _ = logits.max(2)
    topk_idcs = []
    for i in range(len(logits)):
        sort_idcs = scores[i].argsort(descending=True)
        topk_idcs.append(sort_idcs[:k])
    return torch.stack(topk_idcs, 0)


def batch_index(batch, batch_idcs):
    data = []
    for i in range(len(batch)):
        data.append(batch[i][batch_idcs[i]])

    if isinstance(batch_idcs, Tensor):
        data = torch.stack(data, 0)
    return data


def gt_from_label(gt_bboxes, gt_labels, bev_range, classes, track=False, labels_history=None, device=None):

    if isinstance(bev_range, Tensor):
        x_min_all, x_max_all, y_min_all, y_max_all = bev_range.chunk(4, dim=-1)
    else:
        x_min, x_max, y_min, y_max = bev_range
    gt: Dict[str, List[Tensor]] = dict(bboxes=[], cls_idcs=[], ignore=[], tracks=[])
    if device is None:
        device = gt_bboxes[0].device

    batch_size = len(gt_bboxes)
    for i in range(batch_size):
        gt["bboxes"].append([])
        gt["cls_idcs"].append([])
        gt["ignore"].append([])
        gt["tracks"].append([])
        if isinstance(bev_range, Tensor):
            x_min = x_min_all[i]
            x_max = x_max_all[i]
            y_min = y_min_all[i]
            y_max = y_max_all[i]
        for j, c in enumerate(classes):

            # label = gt_labels[i] == c
            x, y, z, l, w, h, theta = gt_bboxes[i].tensor.chunk(7, dim=1)
            z += z_offset
            # x, y = label.trajectories[:, 0, 0], label.trajectories[:, 0, 1]
            # l, w = label.boxes[:, 0], label.boxes[:, 1]
            # theta = label.yaw[:, 0]
            mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)

            bboxes = torch.stack((x[mask], y[mask], z[mask], l[mask], w[mask], h[mask], theta[mask]), -1).to(device)
            cls_idcs = j * torch.ones(len(bboxes), device=device, dtype=torch.int64)
            # cls_idcs = c.value * torch.ones(len(bboxes), device=bboxes.device, dtype=torch.int64)
            # ignore = label.ignores[mask]
            ignore = torch.zeros(len(bboxes), device=device, dtype=torch.bool)
            ignore[gt_labels[i] == -1] = True

            gt["bboxes"][i].append(bboxes)
            gt["cls_idcs"][i].append(cls_idcs)
            gt["ignore"][i].append(ignore)
            if track:
                label_history = labels_history[i][c]
                x_history, y_history = label_history.trajectories[:, :4, 0], label_history.trajectories[:, :4, 1]
                theta_history = label_history.yaw[:, :4]
                tracks = torch.stack((x_history[mask], y_history[mask], theta_history[mask]), -1)
                gt["tracks"][i].append(tracks)

        gt["bboxes"][i] = torch.cat(gt["bboxes"][i], 0)
        gt["cls_idcs"][i] = torch.cat(gt["cls_idcs"][i], 0)
        gt["ignore"][i] = torch.cat(gt["ignore"][i], 0)
        if track:
            gt["tracks"][i] = torch.cat(gt["tracks"][i], 0)
    return gt


class Conv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, dilation=1, act=True):
        super().__init__()
        padding = (int(kernel_size) - 1) // 2 * dilation
        self.conv = nn.Conv2d(n_in, n_out, kernel_size, stride, padding=padding, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x: Tensor):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Linear(nn.Module):
    def __init__(self, n_in, n_out, act=True):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.norm = nn.LayerNorm(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x: Tensor):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


def get_roi_feats(fm, rois, bev_range: Tuple[float, float, float, float], op: RoIAlignRotated):
    N, C, H, W = fm.shape
    num_rois = rois.shape[1]
    if isinstance(bev_range, Tensor):
        try:
            x_min, x_max, y_min, y_max = bev_range.chunk(4, dim=-1)
        except:
            x_min, x_max, y_min, y_max, _, _ = bev_range.chunk(4, dim=-1)
    else:
        try:
            x_min, x_max, y_min, y_max = bev_range
        except:
            x_min, x_max, y_min, y_max, _, _ = bev_range

    idcs = torch.arange(N, device=fm.device, dtype=fm.dtype)
    rois = torch.cat((idcs.view(-1, 1, 1).repeat(1, num_rois, 1), rois), 2)  # (N, K, 6)
    rois[..., 1] = (rois[..., 1] - x_min) / (x_max - x_min) * W
    rois[..., 2] = (y_max - rois[..., 2]) / (y_max - y_min) * H
    # rois[..., 2] = (rois[..., 2] - y_min) / (y_max - y_min) * H
    rois[..., 3] = rois[..., 3] / (x_max - x_min) * W
    rois[..., 4] = rois[..., 4] / (y_max - y_min) * H
    rois[..., 5] = rois[..., 5]
    rois = rois.view(N * num_rois, 6)

    feats = op.forward(fm.contiguous(), rois.contiguous())
    feats = feats.view(N, num_rois, C, -1).transpose(2, 3).contiguous()
    return feats


def get_roi_coords(rois, roi_size: int):
    assert rois.ndim == 3 and rois.shape[-1] == 5
    N, num_dets, _ = rois.shape

    rois = rois.view(-1, 5)
    num_rois = len(rois)

    cx, cy, wid, hgt, theta = rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
    st = torch.sin(theta)
    ct = torch.cos(theta)

    rot = torch.empty((num_rois, 2, 2), device=rois.device, dtype=rois.dtype)
    rot[:, 0, 0] = ct
    rot[:, 0, 1] = -st
    rot[:, 1, 0] = st
    rot[:, 1, 1] = ct

    x = torch.arange(roi_size, device=rois.device, dtype=rois.dtype)
    x = (x + 0.5) / roi_size - 0.5
    y = torch.arange(roi_size - 1, -1, -1, device=rois.device, dtype=rois.dtype)
    y = (y + 0.5) / roi_size - 0.5

    x = x.view(1, 1, -1) * wid.view(-1, 1, 1)
    y = y.view(1, -1, 1) * hgt.view(-1, 1, 1)

    xy = torch.zeros((num_rois, roi_size, roi_size, 2), device=rois.device, dtype=rois.dtype)
    xy[:, :, :, 0] = x
    xy[:, :, :, 1] = y

    rot = rot.view(num_rois, 1, 1, 2, 2)
    xy = xy.view(num_rois, roi_size, roi_size, 2, 1)
    xy = torch.matmul(rot, xy).view(num_rois, roi_size, roi_size, 2)

    xy[:, :, :, 0] += cx.view(-1, 1, 1)
    xy[:, :, :, 1] += cy.view(-1, 1, 1)
    xy = xy.view(N, num_dets, -1, 2)
    return xy


class ROIAttention(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.weight = nn.Sequential(
            Linear(3 * n, n),
            nn.Linear(n, n),
        )

        self.value = nn.Sequential(
            Linear(2 * n, n),
            nn.Linear(n, n),
        )

        self.pe = nn.Sequential(
            nn.Linear(2, n),
            nn.ReLU(inplace=True),
            Linear(n, n, act=False),
        )

        self.norm = nn.LayerNorm(n)
        self.relu = nn.ReLU(inplace=True)
        self.linear = Linear(n, n, act=False)

    def forward(self, tgt, tgt_coords, src, src_coords):
        N, num_rois, num_pxls, C = src.shape
        x = torch.empty((N, num_rois, num_pxls, 3 * C), device=src.device, dtype=src.dtype)
        x[:, :, :, :C] = src
        x[:, :, :, C : 2 * C] = self.pe(src_coords - tgt_coords.unsqueeze(2))
        x[:, :, :, 2 * C :] = tgt.unsqueeze(2)

        weight = self.weight(x).sigmoid()
        value = self.value(x[:, :, :, : 2 * C])
        out = (weight * value).sum(2)

        out = self.norm(out)
        out = self.relu(out)
        out = self.linear(out)
        out += tgt
        out = self.relu(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

        self.weight = nn.Sequential(
            Linear(3 * n, n),
            nn.Linear(n, n),
        )

        self.value = nn.Sequential(
            Linear(2 * n, n),
            nn.Linear(n, n),
        )

        self.pe = nn.Sequential(
            nn.Linear(2, n),
            nn.ReLU(inplace=True),
            Linear(n, n, act=False),
        )

        self.norm = nn.LayerNorm(n)
        self.relu = nn.ReLU(inplace=True)
        self.linear = Linear(n, n, act=False)

    def forward(
        self,
        src: Tensor,
        src_coords: Tensor,
        tgt: Optional[Tensor] = None,
        tgt_coords: Optional[Tensor] = None,
        dist_th: Optional[float] = None,
        masks: Optional[Tensor] = None,
    ):
        """Forward pass"""
        if dist_th is not None and masks is not None:
            raise RuntimeError("Cannot have both inputs dist_th and masks set")

        if dist_th is None and masks is None:
            raise RuntimeError("Must set either dist_th or masks as input")

        if tgt is None:
            tgt = src

        if tgt_coords is None:
            tgt_coords = src_coords

        batch_size = len(src)
        batch = []
        for i in range(batch_size):
            out = torch.zeros_like(tgt[i])

            mask = torch.empty([tgt_coords[i].size()[0], src_coords[i].size()[0]], dtype=torch.bool)
            if dist_th is not None:
                dist = tgt_coords[i].view(-1, 1, 2) - src_coords[i].view(1, -1, 2)
                dist = torch.sqrt((dist**2).sum(2))
                mask = dist < dist_th
            elif masks is not None:
                mask = masks[i]

            idcs = torch.nonzero(mask)
            if len(idcs) > 0:
                hi, wi = idcs[:, 0], idcs[:, 1]
                dist = self.pe(tgt_coords[i][hi] - src_coords[i][wi])
                x = torch.cat((src[i][wi], dist, tgt[i][hi]), 1)
                weight = self.weight(x).sigmoid()
                value = self.value(x[:, : 2 * self.n])
                # Non deterministic
                out.index_add_(0, hi, weight * value)

            out = self.norm(out)
            out = self.relu(out)
            out = self.linear(out)
            out += tgt[i]
            out = self.relu(out)
            batch.append(out)

        return torch.stack(batch, 0)


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@DETECTORS.register_module()
class WaabiTwoStageDetector(Base3DDetector):
    """Base class of two-stage 3D detector.

    It inherits original ``:class:TwoStageDetector`` and
    ``:class:Base3DDetector``. This class could serve as a base class for all
    two-stage 3D detectors.
    """

    def __init__(
        self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, init_cfg=None, pretrained=None
    ):
        super(WaabiTwoStageDetector, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        # if neck is not None:
        #     self.neck = build_neck(neck)
        # bbox_head.update(train_cfg=train_cfg)
        # bbox_head.update(test_cfg=test_cfg)
        # self.bbox_head = build_head(bbox_head)
        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg

        self.has_heading = False
        self.has_track = False

        self.box_reg_dim = 6 if self.has_heading else 7
        # self.box_reg_dim = 6 if self.has_heading else 5
        self.reg_dim = self.box_reg_dim + 3 * 4 if self.has_track else self.box_reg_dim

        # Two-stage model related hyper-parameters. Move them to config after the merged model is fixed.
        num_classes = 1
        self.active_classes = [0]
        self.num_dets = 500  # number of first stage output
        self.nms_th = 0.7  # first stage nms
        # self.bev_range = (self.voxel_cfg.x_min, self.voxel_cfg.x_max, self.voxel_cfg.y_min, self.voxel_cfg.y_max)
        # [0, -39.68, -3, 69.12, 39.68, 1]
        # self.bev_range = [0, 69.12, -39.68, 39.68]
        # self.bev_range = [0, 72.5, -40, 40] # for kitti
        self.bev_range = [0, 80.0, -40, 40]  # for pandaset
        # self.bev_range = [-75, 75, -75, 75]
        # self.bev_range = [-74.24, 74.24, -74.24, 74.24]
        self.roi_size = 3
        self.dist_th = 20.0  # distance threshold used in spatial attention
        weights = dict(cls=1.0, iou=2.0, track=1.0, heading=1.0)  # weights of multi-task loss
        focal = dict(alpha=0.5, gamma=2)
        dontcare_iou = 0.8  # first stage dontcare iou threshold
        pos_iou = [0.5, 0.1, 0.1]
        pos_iou = [pos_iou[c] for c in self.active_classes]  # second stage, positive samples with low iou are ignored
        n = 128  # feature dimension

        # Post-processing hyperparameters
        self.conf_thresh = 0.05
        self.max_det = 100

        self.voxelizer = Voxelizer(
            # self.bev_range[0], self.bev_range[1], self.bev_range[2], self.bev_range[3], 0.32, -2, 4, 0.15
            self.bev_range[0],
            # 80,
            self.bev_range[1],
            self.bev_range[2],
            self.bev_range[3],
            0.15625,
            # 0.15625 * 2,
            -2,
            4,
            0.15,
        )
        # self.voxelizer = Voxelizer(
        #     self.bev_range[0], self.bev_range[1], self.bev_range[2], self.bev_range[3], 0.3, -2, 4, 0.3
        # )

        # First stage detection header
        propose = dict()
        propose["cls"] = nn.Sequential(
            Conv(n, n),
            nn.Conv2d(n, num_classes, 1, padding=0),
        )
        nn.init.constant_(propose["cls"][-1].bias.data, -2)
        propose["reg"] = nn.Sequential(
            Conv(n, n),
            Conv(n, n),
            nn.Conv2d(n, 8, 1, padding=0),
        )
        self.propose = nn.ModuleDict(propose)

        # Second stage detection header
        self.input2 = Conv(n, n, act=False)
        self.pe = nn.Sequential(
            nn.Linear(2, n),
            nn.ReLU(inplace=True),
            Linear(n, n, act=False),
        )
        self.roi_align_rotated = RoIAlignRotated(
            output_size=[self.roi_size, self.roi_size], spatial_scale=1.0, sampling_ratio=1
        )
        self.roi_attention = ROIAttention(n)
        self.spatial_attention = SpatialAttention(n)

        refine = dict()
        refine["cls"] = nn.Sequential(
            Linear(n, n),
            nn.Linear(n, num_classes),
        )
        refine["reg"] = nn.Sequential(
            Linear(n, n),
            Linear(n, n),
            nn.Linear(n, self.reg_dim),
        )
        self.refine = nn.ModuleDict(refine)

        # Detection loss
        self.det_loss = DetectionLoss(weights, focal, dontcare_iou, pos_iou)

        # self.pre_quant = nn.Sequential(nn.Conv2d(128, 256, 1, bias=False), LayerNorm(256))
        # self.quantizer = VectorQuantizer(2048, 256, 0.25)
        # self.post_quant = nn.Sequential(Conv(256, 128))

        # self.preprocessor = LidarVQGAN()
        # self.preprocessor = LidarVQViT()
        # print(
        #     self.preprocessor.load_state_dict(
        #         torch.load(
        #             '/mnt/remote/shared_data/users/yuwen/arch_baselines_oct/vqvit_front_2022-10-27_07-27-37_novq_8x_pandaset_front/checkpoint/model_00150e.pth.tar',
        #             # '/mnt/remote/shared_data/users/yuwen/arch_baselines_oct/vqvit_front_2022-10-23_20-47-24_8x_pandaset_front/checkpoint/model_00140e.pth.tar',
        #             # "/opt/experiments/vqvit_front_2022-10-30_20-05-48/checkpoint/model_0190e.pth.tar",
        #             map_location="cpu",
        #         )["model"],
        #         strict=False,
        #     )
        # )
        self.preprocessor = None

    def forward_dummy(self, points):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(points)
        try:
            sample_mod = self.train_cfg.sample_mod
            outs = self.bbox_head(x, sample_mod)
        except AttributeError:
            outs = self.bbox_head(x)
        return outs

    def forward_header(self, fm, bev_range=None) -> Dict[str, Dict[str, Tensor]]:
        # First stage
        logits = self.propose["cls"](fm)
        reg = self.propose["reg"](fm)
        bboxes = bev_reg_to_bboxes(reg, self.bev_range if bev_range is None else bev_range)

        propose = dict()
        propose["logits"] = logits.flatten(2, 3).transpose(1, 2)
        propose["bboxes"] = bboxes

        # Process first stage output
        topk_idcs = get_topk_idcs(propose["logits"], 1000 + self.num_dets)
        logits = batch_index(propose["logits"], topk_idcs).detach()
        max_logits, _ = logits.max(2)
        bboxes = batch_index(propose["bboxes"], topk_idcs).detach()
        roi_list = []
        for i in range(len(bboxes)):
            # keep_id = nms(torch.cat((max_logits[i][:1000].unsqueeze(1), bboxes[i][:1000]), 1), self.nms_th)
            # keep_id = nms_bev(bboxes[i][:1000], max_logits[i][:1000], self.nms_th, xywhr=True)
            keep_id = nms_bev(bboxes[i][:1000, [0, 1, 3, 4, 6]], max_logits[i][:1000], self.nms_th, xywhr=True)
            if len(keep_id) >= self.num_dets:
                keep_id = keep_id[: self.num_dets]
            else:
                padding = torch.arange(1000, 1000 + self.num_dets - len(keep_id), device=keep_id.device)
                keep_id = torch.cat((keep_id, padding))
            roi_list.append(bboxes[i][keep_id])
        rois = torch.stack(roi_list, 0)

        # Second stage
        fm = self.input2(fm)

        roi_feats = get_roi_feats(
            fm, rois[..., [0, 1, 3, 4, 6]], self.bev_range if bev_range is None else bev_range, self.roi_align_rotated
        )
        roi_coords = get_roi_coords(rois[..., [0, 1, 3, 4, 6]], self.roi_size)
        # roi_feats = get_roi_feats(fm, rois, self.bev_range if bev_range is None else bev_range, self.roi_align_rotated)
        # roi_coords = get_roi_coords(rois, self.roi_size)

        obj_feats = roi_feats[:, :, int(self.roi_size**2 // 2)].clone()
        obj_coords = roi_coords[:, :, int(self.roi_size**2 // 2)].clone()
        obj_feats += self.pe(obj_coords)  # can be removed

        obj_feats = self.roi_attention(obj_feats, obj_coords, roi_feats, roi_coords)
        obj_feats = self.spatial_attention(obj_feats, obj_coords, dist_th=self.dist_th)

        refine = dict()

        reg = self.refine["reg"](obj_feats)
        refine["logits"] = self.refine["cls"](obj_feats)
        refine["bboxes"] = reg[:, :, :7] + rois
        # refine["bboxes"] = torch.cat(
        #     [
        #         reg[:, :, :2] + rois[:, :, :2],
        #         reg[:, :, 2:3],
        #         reg[:, :, 3:5] + rois[:, :, 2:4],
        #         reg[:, :, 5:6],
        #         reg[:, :, 6:7] + rois[:, :, 4:5],
        #     ],
        #     dim=-1,
        # )
        refine["rois"] = rois

        if self.has_heading:
            refine["heading"] = reg[:, :, 5]

        if self.has_track:
            refine["tracks"] = reg[:, :, self.box_reg_dim :].view(len(rois), -1, 4, 3)
            refine["tracks"] += rois[:, :, [0, 1, 4]].unsqueeze(2).repeat([1, 1, 4, 1])

        det_out = dict(
            propose=propose,
            refine=refine,
        )
        return det_out

    def det_post_process(
        self,
        det_out: Dict[str, Dict[str, Tensor]],
        postprocess: bool = True,
    ):
        # TODO: Add support for multi-class
        bboxes: List[Dict[int, Tensor]] = []
        tracks: List[Dict[int, Tensor]] = []
        # det_model_output: Dict[int, Dict[str, Tensor]] = dict()
        # # dict format conversion: out["propose"]["logits"] --> out[0]["propose_logits"]
        # for c in [0]:
        #     class_output: Dict[str, Tensor] = dict()
        #     for k, v in det_out.items():
        #         for k1, v1 in v.items():
        #             class_output[k + "_" + k1] = v1
        #     det_model_output[c] = class_output

        if postprocess:
            stage = "refine" if "refine" in det_out else "propose"
            out = det_out[stage]
            batch_size = len(out["bboxes"])
            for i in range(batch_size):
                d: Dict[int, Tensor] = dict()
                t: Dict[int, Tensor] = dict()
                for j, c in enumerate(self.active_classes):
                    score = out["logits"][i].sigmoid()
                    score = score[:, j : j + 1].contiguous()
                    sort_idcs = score.view(-1).argsort(descending=True)
                    conf_mask = score.view(-1)[sort_idcs] > self.conf_thresh
                    sort_idcs = sort_idcs[conf_mask]  # Contains descending indices that pass confidence threshold
                    score = score[sort_idcs]
                    bb = out["bboxes"][i][sort_idcs].clone()
                    # mask = bb[:, 2:4] < 0
                    # bb[:, 2:4][mask] = 0.0000001

                    # keep_id = nms(torch.cat((score, bb), 1), 0.5)[:200]
                    # keep_id = nms_bev(bb, score[:, 0], 0.5, xywhr=True)[:200]
                    keep_id = nms_bev(bb[:, [0, 1, 3, 4, 6]], score[:, 0], 0.5, xywhr=True)[:200]
                    if self.max_det is not None and len(keep_id) > self.max_det:
                        # Actors are already sorted by confidence, so this slicing gets the most confident post nms
                        keep_id = keep_id[: self.max_det]
                    if keep_id.numel() == 0:
                        d[c] = torch.empty((0, 6))
                    else:
                        bb = bb[keep_id]
                        score = score[keep_id]
                        d[c] = torch.cat((bb, score), 1)

                bboxes.append(d)

        return bboxes

    def forward_model(self, points):
        assert not hasattr(self.backbone, "map_channels")

        # _, input, gt = torch.load("/home/yuwen/av/model_check.pth")
        # self.load_state_dict(
        #     torch.load("/home/yuwen/mmdetection3d/work_dirs/waabi_two_stage_pandaset-3d-car-run3/epoch_31.pth")[
        #         "state_dict"
        #     ],
        #     strict=True,
        # )
        # self.eval()
        # # self.load_state_dict(state_dict)
        # points = [[p[0]] for p in input]

        for p in points:
            p[0][:, 2] += z_offset
        bev = self.voxelizer(points)
        # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        if self.preprocessor is not None:
            pad_x = bev.new_zeros((bev.shape[0], bev.shape[1], 512, 512))
            pad_x[:, :, : bev.shape[2], : bev.shape[3]] = bev
            residual, _ = self.preprocessor.forward(pad_x)
            pad_x = (pad_x * 200 + residual).sigmoid()
            pad_x[pad_x < 0.1] = 0
            bev = pad_x[:, :, : bev.shape[2], : bev.shape[3]]

            # residual, _ = self.preprocessor.forward(bev)
            # # bev = gumbel_sigmoid(bev * 200 + residual, hard=True)
            # bev = (bev * 200 + residual).sigmoid()
            # bev[bev < 0.1] = 0
            # bev[bev > 0.9] = 1
            # bev = (gumbel_sigmoid(residual, hard=True)).float()
            # bev = (residual.sigmoid()).float()
            # import ipdb; ipdb.set_trace()
        # feat = self.backbone(bev[:, :, :, :464])
        feat = self.backbone(bev)
        # import ipdb; ipdb.set_trace()
        fm = feat[0]
        # q_fm, _, _ = self.quantizer(self.pre_quant(fm))
        q_fm = fm
        # q_fm = self.post_quant(q_fm)
        header_out = self.forward_header(q_fm.float())
        return header_out

        # return self.det_post_process(fm, header_out, post_process=False)

    def extract_feat(self, points, img_metas=None):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        x = self.backbone(points)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, points, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore=None):

        # import copy
        # pts = copy.deepcopy(points)
        header_out = self.forward_model([[_] for _ in points])
        det_out: Dict[int, Dict[str, Tensor]] = dict()

        for c in [0]:
            class_output: Dict[str, Tensor] = dict()
            for k, v in header_out.items():
                for k1, v1 in v.items():
                    class_output[k + "_" + k1] = v1
            det_out[c] = class_output

        gt = gt_from_label(
            # batch_data.labels,
            gt_bboxes_3d,
            gt_labels_3d,
            self.bev_range,
            classes=self.active_classes,
            device=points[0].device,
        )
        # _, input, gt = torch.load("/home/yuwen/av/model_check.pth")

        det_loss = self.det_loss(det_out, gt)

        return det_loss

    def extract_feats(self, points, img_metas):
        """Extract features of multiple samples."""
        return [self.extract_feat(pts, img_meta) for pts, img_meta in zip(points, img_metas)]

    def simple_test(self, points, img_metas, imgs=None, rescale=False):

        self.eval()

        header_out = self.forward_model([points])

        bbox_out = self.det_post_process(header_out)

        bbox_list = []

        for i in range(len(bbox_out)):
            bboxes = torch.zeros((bbox_out[i][0].shape[0], 7))
            # # # bboxes = bbox_out[i][0][:, :-1]
            # bboxes[:, 0:2] = bbox_out[i][0][:, 0:2]
            # bboxes[:, 2] = -1
            # # # bboxes[:, 2] = bbox_out[i][0][:, 1]
            # bboxes[:, 3:5] = bbox_out[i][0][:, 2:4]
            # bboxes[:, 5] = 2.1
            # # # bboxes[:, 5] = bbox_out[i][0][:, 3]
            # bboxes[:, 6:7] = bbox_out[i][0][:, 4:5]
            bboxes = bbox_out[i][0][:, :-1]
            # bboxes[:, [4, 5]] = bboxes[:, [5, 4]]
            bboxes[:, 2] -= z_offset

            bboxes = img_metas[i]["box_type_3d"](bboxes, box_dim=7)
            scores = bbox_out[i][0][:, -1]
            labels = torch.zeros_like(scores).long()

            bbox_list.append((bboxes, scores, labels))

        # import ipdb; ipdb.set_trace()

        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        pass


# # Copyright (c) 2021 Waabi Innovation. All rights reserved.

# import math
# import time
# import warnings
# from collections import defaultdict, deque
# from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union

# import torch
# import torch.nn as nn
# from scipy.optimize import linear_sum_assignment
# from torch import Tensor, autocast
# from torch.nn import functional as F
# from torchvision.ops import sigmoid_focal_loss

# from waabi.autonomy.data.dataloader import BatchedPnPInput
# from waabi.autonomy.mp.rasterizer.config import RasterizerConfig
# from waabi.autonomy.pnp.config import EvaluatorConfig, PerceptionModelConfig
# from waabi.autonomy.pnp.perception.config import DetectorConfig
# from waabi.autonomy.pnp.perception.detection_net import PnPModelInput
# from waabi.autonomy.pnp.perception.modules import backbones, necks
# from waabi.autonomy.pnp.perception.modules.det_postprocess import (
#     convert_det_output_to_det_frame_history,
#     correct_yaw_with_heading_direction,
# )
# from waabi.autonomy.pnp.perception.modules.dynamic_voxelizer import DynamicVoxelizer
# from waabi.autonomy.pnp.perception.modules.voxelizer import IntensityVoxelizer
# from waabi.autonomy.pnp.perception.ops.nms_py import nms, rbox_iou_2sets_gpu
# from waabi.autonomy.pnp.perception.ops.voxelizer.voxelizer import Voxelizer, VoxelizerOutput
# from waabi.autonomy.pnp.perception.tracker import Tracker, convert_track_output_to_det_frame
# from waabi.autonomy.pnp.perception.voxel_configs import VoxelizerConfig
# from waabi.autonomy.pnp.prediction.postprocess.postprocessor import PredictionPostProcessor, generate_dummy_predictions
# from waabi.autonomy.pnp.type.internal import ActorClass, DetectionModelOutput
# from waabi.autonomy.pnp.type.metadata.metric_metadata import PnPMetricMetadata
# from waabi.autonomy.pnp.type.output.trajectory_output import PnPTrajectoryOutput
# from waabi.autonomy.pnp.utils.multiclass import MulticlassConfig
# from waabi.autonomy.pnp.utils.nested import update_dict_of_list
# from waabi.autonomy.type.detection_frame import BatchedDetectionFrameHistory
# from waabi.common.datasets.pandaset.data_processors.lidar import lidar_name_to_pandaset_enum
# from waabi.common.geometry.angles import wrap_angle_rad
# from waabi.common.ops.roi_align_rotated import ROIAlignRotated
# from waabi.common.training.experiments import ExperimentLogger
# from waabi.metrics.detection.detection_runner import DetectionSequentialMetricsRunner
# from waabi.sim.recorder.log_reader import LogReader
# from waabi.sim.type.lidar_frame import LidarSensorFrame


# class TwoStage(nn.Module):
#     def __init__(self, cfg: PerceptionModelConfig):
#         """Init with ..pnp_model.PerceptionModelConfig"""
#         super().__init__()

#         cfg = OmegaConf.merge(
#             PerceptionModelConfig(voxel_cfg=cfg.voxel_cfg, det_cfg=cfg.det_cfg, pred_cfg=cfg.pred_cfg), cfg
#         )  # Enables backwards compatibility after multilidar change
#         self.voxel_cfg: VoxelizerConfig = cfg.voxel_cfg
#         self.class_cfg: MulticlassConfig = cfg.class_cfg
#         self.model_cfg: DetectorConfig = cfg.det_cfg
#         self.raster_cfg = (
#             cfg.raster_cfg if isinstance(cfg.raster_cfg, RasterizerConfig) else OmegaConf.to_object(cfg.raster_cfg)
#         )
#         self.jit_raster_cfg = self.raster_cfg.to_jit_object()

#         num_lidars = cfg.num_lidars

#         # Different two-stage model variants
#         self.multi_lidar = num_lidars > 1

#         if self.multi_lidar:
#             assert (
#                 cfg.det_cfg.backbone.in_channels == 5 * num_lidars + 1 or cfg.det_cfg.backbone.in_channels == 128
#             ), "Incorrect number of channels for the given number of lidars. We expect for {} lidars to have \
#                 {} channels".format(
#                 num_lidars, 5 * num_lidars + 1
#             )

#         self.num_input_sweeps = self.model_cfg.backbone.in_channels if not self.multi_lidar else 5
#         # whether to add heading direction classification to 2nd-stage reg out
#         self.has_heading = self.model_cfg.header.add_heading
#         # whether to add past tracks to 2nd-stage reg out
#         self.has_track = cfg.has_track
#         self.box_reg_dim = 6 if self.has_heading else 5
#         self.reg_dim = self.box_reg_dim + 3 * 4 if self.has_track else self.box_reg_dim

#         # Two-stage model related hyper-parameters. Move them to config after the merged model is fixed.
#         num_classes = len(cfg.class_cfg.active_classes)
#         self.active_classes = [int(x) for x in self.class_cfg.active_classes]
#         self.num_dets = 500  # number of first stage output
#         self.nms_th = 0.7  # first stage nms
#         self.bev_range = (self.voxel_cfg.x_min, self.voxel_cfg.x_max, self.voxel_cfg.y_min, self.voxel_cfg.y_max)
#         self.roi_size = 3
#         self.dist_th = 20.0  # distance threshold used in spatial attention
#         weights = dict(cls=1.0, iou=2.0, track=1.0, heading=1.0)  # weights of multi-task loss
#         focal = dict(alpha=0.5, gamma=2)
#         dontcare_iou = 0.8  # first stage dontcare iou threshold
#         pos_iou = [0.5, 0.1, 0.1]
#         pos_iou = [pos_iou[c] for c in self.active_classes]  # second stage, positive samples with low iou are ignored
#         n = 128  # feature dimension

#         # Post-processing hyperparameters
#         self.conf_thresh = self.model_cfg.score_threshold
#         self.max_det = self.model_cfg.nms_topk

#         # Build voxelizer
#         if "sparse" in self.model_cfg.backbone.name.lower():
#             if not self.multi_lidar:
#                 self.voxelizer: Union[
#                     IntensityVoxelizer, Voxelizer, MultiLidarVoxelize, DynamicVoxelizer
#                 ] = DynamicVoxelizer(self.voxel_cfg)
#             else:
#                 self.voxelizer = MultiLidarVoxelize(
#                     self.voxel_cfg, n_out=self.model_cfg.backbone.in_channels, num_sensors=num_lidars, num_sweeps=5
#                 )
#         else:
#             if not self.multi_lidar:
#                 self.voxelizer = (
#                     IntensityVoxelizer(self.voxel_cfg)
#                     if self.voxel_cfg.add_intensity
#                     or (self.voxel_cfg.add_ground and (self.voxel_cfg.z_max - self.voxel_cfg.z_min) > 0)
#                     else Voxelizer(self.voxel_cfg)
#                 )
#             else:
#                 self.voxelizer = MultiLidarVoxelize(
#                     self.voxel_cfg,
#                     n_out=self.model_cfg.backbone.in_channels,
#                     num_sensors=num_lidars,
#                     num_sweeps=5,
#                     dense=True,
#                 )

#         # Build backbone
#         self.backbone = backbones.build_backbone(self.model_cfg.backbone)
#         self.neck = necks.build_neck(self.model_cfg.neck)

#         # First stage detection header
#         propose = dict()
#         propose["cls"] = nn.Sequential(
#             Conv(n, n),
#             nn.Conv2d(n, num_classes, 1, padding=0),
#         )
#         nn.init.constant_(propose["cls"][-1].bias.data, -2)
#         propose["reg"] = nn.Sequential(
#             Conv(n, n),
#             Conv(n, n),
#             nn.Conv2d(n, 6, 1, padding=0),
#         )
#         self.propose = nn.ModuleDict(propose)

#         # Second stage detection header
#         self.input2 = Conv(n, n, act=False)
#         self.pe = nn.Sequential(
#             nn.Linear(2, n),
#             nn.ReLU(inplace=True),
#             Linear(n, n, act=False),
#         )
#         self.roi_align_rotated = ROIAlignRotated(
#             output_size=[self.roi_size, self.roi_size], spatial_scale=1.0, sampling_ratio=1
#         )
#         self.roi_attention = ROIAttention(n)
#         self.spatial_attention = SpatialAttention(n)

#         refine = dict()
#         refine["cls"] = nn.Sequential(
#             Linear(n, n),
#             nn.Linear(n, num_classes),
#         )
#         refine["reg"] = nn.Sequential(
#             Linear(n, n),
#             Linear(n, n),
#             nn.Linear(n, self.reg_dim),
#         )
#         self.refine = nn.ModuleDict(refine)

#         # Detection loss
#         self.det_loss = DetectionLoss(weights, focal, dontcare_iou, pos_iou)

#         # PnP output converter
#         # not necessary, but needed in compute_metrics() because detection metric runner needs PnPTrajectoryOutput
#         self.num_pred_modes = cfg.pred_cfg.model_cfg.num_modes
#         self.pred_len = cfg.pred_cfg.model_cfg.pred_len
#         self.pred_target_dim = cfg.pred_cfg.model_cfg.pred_target
#         self.pnp_converter = PredictionPostProcessor(
#             cfg.pred_cfg.postprocess_cfg,
#             self.class_cfg,
#             cfg.pred_label_type,
#             cfg.pred_cfg.sweep_duration_secs,
#             cfg.pred_cfg.pred_delta_t_secs,
#         )

#         # Tracker
#         # called in track_inference() to perform joint detection and tracking, requires self.has_track = True
#         if self.has_track:
#             trackers = []
#             track_classes = []
#             trackers.append(Tracker(track_steps=10, min_score=0.3, match_th=0.5, momentum=0.9, grow_decay=0.9))
#             track_classes.append(ActorClass.VEHICLE)
#             self.trackers = nn.ModuleList(trackers)
#             self.track_classes = track_classes

#     def forward(
#         self, model_input: PnPModelInput, post_process: bool = True, preprocessing_model=None
#     ) -> DetectionModelOutput:
#         assert not hasattr(self.backbone, "map_channels")

#         bev = self.voxelizer(model_input.batched_lidar)
#         if preprocessing_model is not None:
#             # updated_bev, _ = preprocessing_model(preprocessing_model.patchify(bev.voxel_features))
#             # updated_bev = gumbel_sigmoid(preprocessing_model.unpatchify(updated_bev), hard=True)
#             updated_bev, _ = preprocessing_model(bev.voxel_features)
#             updated_bev = (updated_bev + bev.voxel_features * 20).sigmoid()
#             fm = self.neck(self.backbone(VoxelizerOutput(updated_bev)))
#         else:
#             fm = self.neck(self.backbone(bev))
#         header_out = self.forward_header(fm.float())

#         return self.det_post_process(fm, header_out, post_process)

#     def forward_header(self, fm, bev_range=None) -> Dict[str, Dict[str, Tensor]]:
#         with autocast("cuda", dtype=torch.float16, enabled=False):
#             # First stage
#             logits = self.propose["cls"](fm)
#             reg = self.propose["reg"](fm)
#             bboxes = bev_reg_to_bboxes(reg, self.bev_range if bev_range is None else bev_range)

#             propose = dict()
#             propose["logits"] = logits.flatten(2, 3).transpose(1, 2)
#             propose["bboxes"] = bboxes

#             # Process first stage output
#             topk_idcs = get_topk_idcs(propose["logits"], 1000 + self.num_dets)
#             logits = batch_index(propose["logits"], topk_idcs).detach()
#             max_logits, _ = logits.max(2)
#             bboxes = batch_index(propose["bboxes"], topk_idcs).detach()
#             roi_list = []
#             for i in range(len(bboxes)):
#                 keep_id = nms(torch.cat((max_logits[i][:1000].unsqueeze(1), bboxes[i][:1000]), 1), self.nms_th)
#                 if len(keep_id) >= self.num_dets:
#                     keep_id = keep_id[: self.num_dets]
#                 else:
#                     padding = torch.arange(1000, 1000 + self.num_dets - len(keep_id), device=keep_id.device)
#                     keep_id = torch.cat((keep_id, padding))
#                 roi_list.append(bboxes[i][keep_id])
#             rois = torch.stack(roi_list, 0)

#             # Second stage
#             fm = self.input2(fm)

#             roi_feats = get_roi_feats(
#                 fm, rois, self.bev_range if bev_range is None else bev_range, self.roi_align_rotated
#             )
#             roi_coords = get_roi_coords(rois, self.roi_size)

#             obj_feats = roi_feats[:, :, int(self.roi_size ** 2 // 2)].clone()
#             obj_coords = roi_coords[:, :, int(self.roi_size ** 2 // 2)].clone()
#             obj_feats += self.pe(obj_coords)  # can be removed

#             obj_feats = self.roi_attention(obj_feats, obj_coords, roi_feats, roi_coords)
#             obj_feats = self.spatial_attention(obj_feats, obj_coords, dist_th=self.dist_th)

#             refine = dict()

#             reg = self.refine["reg"](obj_feats)
#             refine["logits"] = self.refine["cls"](obj_feats)
#             refine["bboxes"] = reg[:, :, :5] + rois
#             refine["rois"] = rois

#             if self.has_heading:
#                 refine["heading"] = reg[:, :, 5]

#             if self.has_track:
#                 refine["tracks"] = reg[:, :, self.box_reg_dim :].view(len(rois), -1, 4, 3)
#                 refine["tracks"] += rois[:, :, [0, 1, 4]].unsqueeze(2).repeat([1, 1, 4, 1])

#             det_out = dict(
#                 propose=propose,
#                 refine=refine,
#             )
#             return det_out

#     def convert_to_pnp_output(
#         self, det_output: DetectionModelOutput, sweep_end_ns: List[Tensor]
#     ) -> List[PnPTrajectoryOutput]:
#         # generate dummy pred_output for detection model
#         bboxes = det_output.bboxes
#         assert bboxes is not None
#         pred_output = generate_dummy_predictions(bboxes, self.num_pred_modes, self.pred_len, self.pred_target_dim)

#         return self.pnp_converter(
#             [int(lidar_timestamps_example[-1].item()) for lidar_timestamps_example in sweep_end_ns],
#             det_output,
#             pred_output,
#         )

#     @torch.jit.unused
#     def loss(
#         self, det_output: DetectionModelOutput, batch_data: BatchedPnPInput, bev_range=None
#     ) -> Tuple[Tensor, Dict[str, float]]:
#         metas = {}

#         gt = gt_from_label(
#             batch_data.labels,
#             self.bev_range if bev_range is None else bev_range,
#             classes=self.active_classes,
#             track=self.has_track,
#             labels_history=batch_data.label_history_dict,
#         )

#         det_loss, det_metas = self.det_loss(det_output.det_outs, gt)

#         metas.update(det_metas)
#         metas["detection_loss"] = det_loss.item()
#         metas["total_loss"] = det_loss.item()
#         return det_loss, metas

#     @torch.jit.unused
#     def train_iter(self, batched_frames: BatchedPnPInput) -> Tuple[Tensor, Dict[str, float]]:
#         self.train()

#         model_input = PnPModelInput.from_batched_frames(
#             batched_frames, self.raster_cfg, is_multi_lidar=self.multi_lidar
#         )

#         det_output = self.forward(model_input, post_process=False)

#         total_loss, metas = self.loss(det_output, batched_frames)
#         return total_loss, metas

#     @torch.jit.export
#     def inference(
#         self, model_input: PnPModelInput, sweep_end_ns: Optional[List[Tensor]] = None
#     ) -> BatchedDetectionFrameHistory:
#         """Perform detection model inference."""
#         det_output = self.forward(model_input, post_process=True)
#         if sweep_end_ns is not None and not self.has_track:
#             sweep_end_ns = None
#             warnings.warn("TwoStage is not a tracklet model, therefore sweep_end_ns is ignored during inference.")
#         det_frame_history = convert_det_output_to_det_frame_history(det_output, self.active_classes, sweep_end_ns)
#         return det_frame_history

#     @torch.jit.export
#     def track_inference(
#         self,
#         model_input: PnPModelInput,
#         world_to_vehicle: List[Tensor],
#         sequence_id: Optional[List[int]] = None,
#         track_length: int = 5,
#         world_frame: bool = False,
#     ) -> BatchedDetectionFrameHistory:
#         """Perform discrete tracking after detection model inference, works with batch-size 1 only.
#         If sequence_id is not given, then track_reset() needs to be manually called when applied to a new sequence.
#         """
#         assert len(model_input.batched_lidar) == len(world_to_vehicle) == 1
#         if sequence_id is not None:
#             assert len(sequence_id) == 1
#         assert self.has_track, "track_inference only supports tracklet model."

#         # detection inference
#         det_output = self.forward(model_input, post_process=True)
#         bboxes = det_output.bboxes
#         past_pred = det_output.tracks
#         assert bboxes is not None
#         assert past_pred is not None

#         # NOTE: currently considers only one class but designed for multi-class
#         i = 0
#         trackers: List[Tracker] = []
#         for tracker in self.trackers:
#             c = self.track_classes[i]
#             if sequence_id is not None and sequence_id[0] != tracker.sequence_id:
#                 tracker.reset(sequence_id[0])
#             tracker.update(
#                 bboxes[0][c],
#                 past_pred[0][c],
#                 world_to_vehicle[0].to(device=bboxes[0][c].device),
#             )
#             i += 1
#             trackers.append(tracker)

#         return convert_track_output_to_det_frame(trackers, track_length, world_frame=world_frame)

#     @torch.jit.export
#     def track_reset(self):
#         for tracker in self.trackers:
#             tracker.reset()

#     @staticmethod
#     @torch.jit.unused
#     def eval_iter(
#         model: Union[nn.Module, torch.jit.ScriptModule],
#         batched_frames: BatchedPnPInput,
#         compute_loss: bool,
#         track: bool = False,
#         extrapolate_to_sweep_end: bool = False,
#         **kwargs,  # pylint: disable=unused-argument
#     ):
#         """Returns (model_output, model_output_label, model_output_loss) per batch.
#         Outputs are aggregated in ../evaluator.py, evaluation metrics are then computed by self.compute_metrics().
#         """
#         model.eval()

#         world_frame = kwargs["world_frame"] if "world_frame" in kwargs else False

#         if hasattr(model, "raster_cfg"):
#             model_input = PnPModelInput.from_batched_frames(
#                 batched_frames, model.raster_cfg, is_multi_lidar=model.multi_lidar
#             )
#         else:
#             model_input = PnPModelInput.from_batched_frames(batched_frames, is_multi_lidar=model.multi_lidar)

#         det_output = model.forward(model_input, True, kwargs.get("preprocessing_model", None))
#         # det_output = model.forward(model_input, post_process=True)

#         pnp_traj_output = model.convert_to_pnp_output(det_output, batched_frames.sweep_end_ns)

#         if track and model.has_track:
#             world_to_vehicle = [x.get_matrix() for x in batched_frames.world_to_vehicle]
#             sequence_id = batched_frames.sequence_id
#             det_frame_output = model.track_inference(
#                 model_input, world_to_vehicle, sequence_id, world_frame=world_frame
#             )
#         else:
#             sweep_end_ns = batched_frames.sweep_end_ns if hasattr(model, "has_track") and model.has_track else None
#             det_frame_output = convert_det_output_to_det_frame_history(
#                 det_output, model.active_classes, sweep_end_ns, extrapolate_to_sweep_end
#             )

#         loss_metas: Dict[str, float] = {}
#         if compute_loss and batched_frames.dense_labels is not None:
#             assert not isinstance(model, torch.jit.ScriptModule)
#             _, loss_metas = model.loss(det_output, batched_frames)

#         return {"pnp_traj": pnp_traj_output, "det_frame": det_frame_output}, batched_frames.labels, loss_metas

#     def det_post_process(
#         self,
#         backbone_fm: Tensor,
#         det_out: Dict[str, Dict[str, Tensor]],
#         postprocess: bool = True,
#     ) -> DetectionModelOutput:
#         # TODO: Add support for multi-class
#         bboxes: List[Dict[int, Tensor]] = []
#         tracks: List[Dict[int, Tensor]] = []
#         det_model_output: Dict[int, Dict[str, Tensor]] = dict()
#         # dict format conversion: out["propose"]["logits"] --> out[0]["propose_logits"]
#         for c in [0]:
#             class_output: Dict[str, Tensor] = dict()
#             for k, v in det_out.items():
#                 for k1, v1 in v.items():
#                     class_output[k + "_" + k1] = v1
#             det_model_output[c] = class_output

#         if postprocess:
#             stage = "refine" if "refine" in det_out else "propose"
#             out = det_out[stage]
#             batch_size = len(out["bboxes"])
#             for i in range(batch_size):
#                 d: Dict[int, Tensor] = dict()
#                 t: Dict[int, Tensor] = dict()
#                 for j, c in enumerate(self.active_classes):
#                     score = out["logits"][i].sigmoid()
#                     score = score[:, j : j + 1].contiguous()
#                     sort_idcs = score.view(-1).argsort(descending=True)
#                     conf_mask = score.view(-1)[sort_idcs] > self.conf_thresh
#                     sort_idcs = sort_idcs[conf_mask]  # Contains descending indices that pass confidence threshold
#                     score = score[sort_idcs]
#                     bb = out["bboxes"][i][sort_idcs].clone()
#                     mask = bb[:, 2:4] < 0
#                     bb[:, 2:4][mask] = 0.0000001

#                     keep_id = nms(torch.cat((score, bb), 1), 0.5)[:200]
#                     if self.max_det is not None and len(keep_id) > self.max_det:
#                         # Actors are already sorted by confidence, so this slicing gets the most confident post nms
#                         keep_id = keep_id[: self.max_det]
#                     bb = bb[keep_id]
#                     score = score[keep_id]

#                     if stage == "refine" and "heading" in out:
#                         heading = out["heading"][i][sort_idcs[keep_id]].clone()
#                         bb[:, 4] = correct_yaw_with_heading_direction(bb[:, 4], heading)

#                     d[c] = torch.cat((bb, score), 1)

#                     if stage == "refine" and "tracks" in out:
#                         trks = out["tracks"][i][sort_idcs].clone()
#                         t[c] = trks[keep_id].reshape((-1, 4, 3))
#                         if "heading" in out:
#                             heading = out["heading"][i][sort_idcs[keep_id]].clone()
#                             for ti in range(4):
#                                 t[c][:, ti, 2] = correct_yaw_with_heading_direction(t[c][:, ti, 2], heading)

#                 bboxes.append(d)
#                 if len(t) > 0:
#                     tracks.append(t)

#         return DetectionModelOutput(
#             bboxes=bboxes if len(bboxes) > 0 else None,
#             tracks=tracks if len(tracks) > 0 else None,
#             preds=None,
#             det_outs=det_model_output,
#             score_threshold=0.0,
#             nms_threshold=0.5,
#             pre_nms_topk=1000,
#             nms_topk=200,
#             det_feat=backbone_fm,
#             det_pair_feat=None,
#             adv_loss=None,
#         )

#     @staticmethod
#     @torch.jit.unused
#     def compute_metrics(
#         config: EvaluatorConfig,
#         model_outputs: Dict[str, List],
#         labels: List,
#         logger: ExperimentLogger,
#         metric_metadata: Optional[Sequence[PnPMetricMetadata]] = None,
#     ):
#         del metric_metadata  # Unused
#         metrics = {}
#         pnp_outputs = model_outputs["pnp_traj"]
#         if config.detection_metrics_config is not None:
#             start = time.time()
#             detection_metrics_runner = DetectionSequentialMetricsRunner.build(config.detection_metrics_config)
#             detection_metrics = detection_metrics_runner.run(pnp_outputs, labels)

#             logger.log_val_detection(detection_metrics)
#             logger.print(f"det_metrics_time: {(time.time() - start):.3f}")
#             metrics.update(detection_metrics)

#         return metrics

#     @staticmethod
#     @torch.jit.unused
#     def eval_with_logreader(
#         model: Union[nn.Module, torch.jit.ScriptModule],
#         logreader: LogReader,
#         lidar_sensor_names: List[str],
#         device="cuda",
#         visualizer=None,
#         future_frames: int = 0,
#         sequence_id: int = 0,
#         track: bool = False,
#         **kwargs,
#     ):

#         model_outputs: Dict[str, List] = defaultdict(list)
#         cnt = 0
#         for pnp_input in logreader_loader(
#             logreader,
#             lidar_sensor_names,
#             model.multi_lidar,
#             model.num_input_sweeps,
#             device,
#             future_frames,
#             sequence_id,
#         ):
#             with torch.no_grad():
#                 pnp_output, _, _ = model.eval_iter(model, pnp_input, compute_loss=False, track=track, **kwargs)
#             update_dict_of_list(model_outputs, pnp_output, keys_to_ignore=["full_motion_pred", "full_motion_gt"])
#             update_dict_of_list(model_outputs, {"log_id": pnp_input.sequence_id})

#             if visualizer:
#                 visualizer.visualize_batch(pnp_input, pnp_output["det_frame"])

#             cnt += 1
#             print(f"processed {cnt} frames")
#         return model_outputs


# class MultiLidarVoxelize(nn.Module):
#     def __init__(self, cfg: VoxelizerConfig, n_out: int, num_sensors: int, num_sweeps: int, dense=False):
#         super().__init__()
#         self.lidar_range = [cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max, cfg.z_min, cfg.z_max]  # z: (-1.5, 5.5)
#         x_min, x_max, y_min, y_max, z_min, z_max = self.lidar_range

#         self.voxel_size = [cfg.step, cfg.step, cfg.z_step]
#         self.num_sensors = num_sensors
#         self.n_out = n_out
#         self.num_sweeps = num_sweeps
#         self.dense = dense

#         self.num_x = round((x_max - x_min) / cfg.step)
#         self.num_y = round((y_max - y_min) / cfg.step)
#         self.num_z = round((z_max - z_min) / cfg.z_step)

#         n_feat = 16
#         if not dense:
#             self.block = nn.Sequential(
#                 Linear(num_sweeps * (5 + num_sensors), n_feat),
#                 nn.Linear(n_feat, n_out),
#             )
#             self.norm = nn.LayerNorm(n_out)
#         else:
#             self.block = nn.Sequential(
#                 Linear(num_sweeps * (5 + num_sensors), n_feat),
#                 nn.Linear(n_feat, n_feat),
#             )
#             self.norm = nn.LayerNorm(n_feat)
#             self.pos_embedding = nn.Embedding(self.num_z, n_feat)
#             self.dense_block = nn.Sequential(Linear(2 * n_feat, n_feat), nn.Linear(n_feat, n_out))
#             self.dense_norm = nn.LayerNorm(n_out)

#     def forward(self, lidar: List[List[torch.Tensor]]) -> VoxelizerOutput:
#         # sweep: (x, y, z, intensity, dt, sensor_id starting from 0)
#         batch_size = len(lidar)
#         num_sweeps = len(lidar[0])
#         assert num_sweeps == self.num_sweeps
#         x_min, x_max, y_min, y_max, z_min, _ = self.lidar_range
#         x_size, y_size, z_size = self.voxel_size

#         feats_list, coords_list = [], []
#         for i, sweeps in enumerate(lidar):
#             for j, sweep in enumerate(sweeps):
#                 mask = (
#                     (sweep[:, 0] > x_min + 1e-6)
#                     & (sweep[:, 0] < x_max - 1e-6)
#                     & (sweep[:, 1] > y_min + 1e-6)
#                     & (sweep[:, 1] < y_max - 1e-6)
#                     # & (sweep[:, 2] > z_min + 1e-6)
#                     # & (sweep[:, 2] < z_max - 1e-6)
#                 )
#                 sweep = sweep[mask]

#                 x = (sweep[:, 0] - x_min) / x_size
#                 y = (y_max - sweep[:, 1]) / y_size
#                 z = (sweep[:, 2] - z_min) / z_size

#                 feat = torch.zeros(
#                     (len(sweep), num_sweeps, 5 + self.num_sensors), dtype=sweep.dtype, device=sweep.device
#                 )
#                 feat[:, j, 0] = x - x.floor() - 0.5
#                 feat[:, j, 1] = y - y.floor() - 0.5
#                 feat[:, j, 2] = z - torch.clamp(z.floor(), min=0, max=self.num_z - 1) - 0.5
#                 feat[:, j, 3] = 1.0
#                 feat[:, j, 4] = sweep[:, 4] * 10.0
#                 idcs = torch.arange(len(feat), device=feat.device)
#                 feat[idcs, j, 5 + sweep[:, 5].long()] = 1.0
#                 feat = feat.flatten(1, 2)
#                 feats_list.append(feat)

#                 coord = torch.zeros((len(sweep), 4), dtype=torch.int64, device=sweep.device)
#                 coord[:, 0] = i
#                 coord[:, 1] = x.long()
#                 coord[:, 2] = y.long()
#                 coord[:, 3] = torch.clamp(z.long(), min=0, max=self.num_z - 1)
#                 coords_list.append(coord)

#         feats = torch.cat(feats_list, 0)
#         coords = torch.cat(coords_list, 0)
#         feats = self.block(feats)

#         coords = (
#             coords[:, 0] * (self.num_x * self.num_y * self.num_z)
#             + coords[:, 1] * self.num_y * self.num_z
#             + coords[:, 2] * self.num_z
#             + coords[:, 3]
#         )
#         coords, idcs = torch.unique(coords, return_inverse=True)

#         buff = torch.zeros(coords.shape[0], feats.shape[1], device=feats.device, dtype=feats.dtype)
#         buff.index_add_(0, idcs, feats)
#         feats = buff

#         buff = torch.zeros(coords.shape[0], 4, device=coords.device, dtype=coords.dtype)
#         buff[:, 0] = torch.div(coords, self.num_x * self.num_y * self.num_z, rounding_mode="floor")
#         buff[:, 1] = torch.div(
#             coords % (self.num_x * self.num_y * self.num_z), self.num_y * self.num_z, rounding_mode="floor"
#         )
#         buff[:, 2] = torch.div(coords % (self.num_y * self.num_z), self.num_z, rounding_mode="floor")
#         buff[:, 3] = coords % self.num_z
#         coords = buff

#         # size = (batch_size, self.num_x, self.num_y, self.num_z, self.n_out)
#         # out = torch.sparse_coo_tensor(coords.T, feats, size).coalesce()
#         # feats = out.values()
#         # coords = out.indices().T

#         feats = self.norm(feats)
#         if not self.dense:
#             out = VoxelizerOutput(
#                 voxel_features=feats,
#                 voxel_coords=coords[:, [0, 3, 2, 1]].contiguous().int(),
#                 batch_size=batch_size,
#                 spatial_shape=torch.tensor((self.num_z, self.num_y, self.num_x)),
#                 sparse=True,
#             )

#         else:
#             batch_size = len(lidar)
#             out_tensor = torch.zeros(
#                 (batch_size * self.num_y * self.num_x, self.n_out), device=feats.device, dtype=feats.dtype
#             )
#             bi = coords[:, 0]
#             hi = coords[:, 2]
#             wi = coords[:, 1]
#             zi = coords[:, 3]
#             pos = self.pos_embedding(zi)
#             feats = torch.cat((feats, pos), 1)
#             feats = self.dense_block(feats)
#             out_tensor.index_add_(0, bi * self.num_x * self.num_y + hi * self.num_x + wi, feats)
#             out_tensor = self.dense_norm(out_tensor)
#             out = VoxelizerOutput(
#                 out_tensor.view(batch_size, -1, self.n_out)
#                 .transpose(1, 2)
#                 .contiguous()
#                 .view(batch_size, self.n_out, self.num_y, self.num_x)
#             )
#         return out
