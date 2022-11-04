# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from ..builder import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector
from typing import List

z_offset = 0.0

class Voxelizer(torch.nn.Module):
    """Voxelizer for converting Lidar point cloud to image"""

    def __init__(self, x_min, x_max, y_min, y_max, step, z_min, z_max, z_step):
        super().__init__()

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.step = step
        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step

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
class CenterPoint(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterPoint,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)

        self.voxelizer = Voxelizer(x_min=pts_voxel_layer['point_cloud_range'][0], y_min=pts_voxel_layer['point_cloud_range'][1], z_min=-2, x_max=pts_voxel_layer['point_cloud_range'][3], y_max=pts_voxel_layer['point_cloud_range'][4], z_max=4, step=pts_voxel_layer['voxel_size'][0], z_step=0.15)


    @property
    def with_velocity(self):
        """bool: Whether the head predicts velocity"""
        return self.pts_bbox_head is not None and \
            self.pts_bbox_head.with_velocity

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        for p in pts:
            p[:, 2] += z_offset

        # voxels, num_points, coors = self.voxelize(pts)

        # voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        # batch_size = coors[-1, 0] + 1
        # x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.voxelizer([[_] for _ in pts])
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        for box in gt_bboxes_3d:
            box.tensor[:, 2] += z_offset

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        for i in range(len(bbox_list)):
            bbox_list[i][0].tensor[:, 2] -= z_offset
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.

        The function implementation process is as follows:

            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.

        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool, optional): Whether to rescale bboxes.
                Default: False.

        Returns:
            dict: Returned bboxes consists of the following keys:

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        outs_list = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            return bbox_list[0]

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]
