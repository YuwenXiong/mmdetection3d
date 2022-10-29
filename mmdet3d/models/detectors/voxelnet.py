# Copyright (c) OpenMMLab. All rights reserved.
from turtle import xcor
from typing import List
import torch
from mmcv.ops import Voxelization
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from .. import builder
from ..builder import DETECTORS
from .single_stage import SingleStage3DDetector
from mmdet3d.models.detectors.vqvae import LidarVQGAN
from mmdet3d.models.detectors.vqvit import LidarVQViT

z_offset = 0.0 # waymo train/val
# z_offset = 0.4
# z_offset = 1.6

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
        # indices_h = torch.floor((self.y_max - lidar[:, 1]) / self.step).long()
        indices_h = torch.floor((lidar[:, 1] - self.y_min) / self.step).long()
        indices_w = torch.floor((lidar[:, 0] - self.x_min) / self.step).long()
        indices_d = torch.clamp(
            torch.floor((lidar[:, 2] - self.z_min) / self.z_step),
            0,
            self.z_depth - 1,
        ).long()
        # 3. Remove points out of bound
        valid_mask = ~torch.any(
            torch.stack(
                [
                    indices_h < 0,
                    indices_h >= self.height,
                    indices_w < 0,
                    indices_w >= self.width,
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

    def forward(self, lidars: List[List[torch.Tensor]]):
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


@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
        # self.voxel_layer = Voxelization(**voxel_layer)
        # self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        # self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.voxelizer = Voxelizer(x_min=voxel_layer['point_cloud_range'][0], y_min=voxel_layer['point_cloud_range'][1], z_min=-2, x_max=voxel_layer['point_cloud_range'][3], y_max=voxel_layer['point_cloud_range'][4], z_max=4, step=voxel_layer['voxel_size'][0], z_step=0.15)

        # # self.preprocessor = LidarVQGAN()
        # self.preprocessor = LidarVQViT()
        # print(
        #     self.preprocessor.load_state_dict(
        #         torch.load(
        #             '/mnt/remote/shared_data/users/yuwen/arch_baselines_oct/vqvit_front_2022-10-23_20-47-24_8x_pandaset_front/checkpoint/model_00140e.pth.tar',
        #             map_location="cpu",
        #         )["model"],
        #         strict=False,
        #     )
        # )
        # self.preprocessor.eval()
        # for p in self.preprocessor.parameters():
        #     p.requires_grad = False
        self.preprocessor = None

    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        for p in points:
            p[:, 2] += z_offset
        # voxels, num_points, coors = self.voxelize(points)
        # voxel_features = self.voxel_encoder(voxels, num_points, coors)
        # batch_size = coors[-1, 0].item() + 1
        # x = self.middle_encoder(voxel_features, coors, batch_size)

        x = self.voxelizer([[_] for _ in points])
        if self.preprocessor is not None:
            with torch.no_grad():
                pad_x = x.new_zeros((x.shape[0], x.shape[1], 512, 512))
                pad_x[:, :, :x.shape[2], :x.shape[3]] = x
                residual, _ = self.preprocessor.forward(pad_x)
                pad_x = (pad_x * 20 + residual).sigmoid()
                pad_x[pad_x < 0.1] = 0
                x = pad_x[:, :, :x.shape[2], :x.shape[3]]
                # x = (x > 0.5).float().detach()

        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        for box in gt_bboxes_3d:
            box.tensor[:, 2] += z_offset

        gt_bboxes_ignore = []
        for i in range(len(gt_bboxes_3d)):
            gt_bboxes_ignore.append(gt_bboxes_3d[i][gt_labels_3d[i] == -1].tensor)
            gt_bboxes_3d[i] = gt_bboxes_3d[i][gt_labels_3d[i] != -1]
            gt_labels_3d[i] = gt_labels_3d[i][gt_labels_3d[i] != -1]
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        for i in range(len(bbox_list)):
            bbox_list[i][0].tensor[:, 2] -= z_offset
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
