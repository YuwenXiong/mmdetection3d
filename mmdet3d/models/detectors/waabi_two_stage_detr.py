# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models import TwoStageDetector

from mmdet3d.core.bbox.transforms import bbox3d2result
from mmdet3d.models.backbones.resnet import PositionEmbeddingSine, conv1x1, conv3x3
from mmdet3d.models.detectors.vqvae import LidarVQGAN
from mmdet3d.models.detectors.waabi_two_stage import Voxelizer, get_roi_coords, get_roi_feats
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
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
import torch.nn.functional as tf
import copy


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def detr_bbox_transform(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        pred_logits: Tensor of dim [batch_size, num_queries, 1] with object score logit
        pred_boxes: Tensor of dim [batch_size, num_queries, 6] with the predicted box coordinates
    Returns:
        the transformed bbox, 6 stands for (score, x, y, length, width, theta) (torch.Tensor N x (H x W) x 6)
    """
    score = torch.sigmoid(pred_logits)
    # assert pred_boxes.shape[-1] == 5
    if pred_boxes.shape[-1] == 7:
        pred_boxes = pred_boxes[..., [0, 1, 3, 4, 6]]
    out = torch.cat([score, pred_boxes], dim=-1)
    mask = out[..., 3:5] < 0
    out[..., 3:5][mask] = 0.000001
    out = out.view(out.shape[0], -1, 6)
    return out


def detr_det_postprocess(
    model_outputs: Dict,
    active_classes: List,
    score_threshold: float,
    nms_threshold: float,
    nms_topk: int,
):
    """
    The det postprocessing function
    det_output: Dict[int, Dict[str, torch.Tensor]]
                Dict[actor_class, Dict[str, torch.Tensor]]
    """
    det_bboxes_per_cls: Dict[int, List[torch.Tensor]] = {}
    pred_logits = model_outputs["pred_logits"]
    pred_boxes = model_outputs["pred_boxes"]
    # all_bboxes = detr_bbox_transform(pred_logits, pred_boxes)
    all_bboxes = torch.cat([pred_logits.sigmoid(), pred_boxes], dim=-1)
    for actor_class in active_classes:
        bboxes = []
        for i in range(all_bboxes.shape[0]):
            score_mask = all_bboxes[i][:, 0] >= score_threshold
            bboxes_pre_nms = all_bboxes[i][score_mask]
            if bboxes_pre_nms.size(0) > 0:
                keep_id = nms_bev(bboxes_pre_nms[:, [1, 2, 4, 5, 7]], bboxes_pre_nms[:, 0], nms_threshold, xywhr=True)
                # use index select to keep score order
                bboxes_post_nms = torch.index_select(bboxes_pre_nms, 0, keep_id)
                post_mask = torch.topk(bboxes_post_nms[:, 0], k=min(bboxes_post_nms.shape[0], nms_topk))[1]
                bboxes_post_topk = bboxes_post_nms[post_mask]
            else:
                bboxes_post_topk = bboxes_pre_nms
            bboxes.append(bboxes_post_topk)

        det_bboxes_per_cls[actor_class] = bboxes

    # convert Dict[int, List[torch.Tensor]] to List[Dict[int, torch.Tensor]]
    det_bboxes = []
    for k, v in det_bboxes_per_cls.items():
        for i, v_i in enumerate(v):
            if i >= len(det_bboxes):
                d: Dict[int, torch.Tensor] = {}
                det_bboxes.append(d)
            # change the order from (score, x, y, l, w, theta) to (x, y, l, w, theta, score)
            # to be aligned with prediction preprocessor
            det_bboxes[i][k] = v_i[:, [1, 2, 3, 4, 5, 6, 7, 0]]

    return det_bboxes, None, None


# def convert_to_xylwsc(out_bbox: torch.Tensor):
#     # x, y, l, w, yaw = out_bbox.unbind(-1)
#     x, y, _, l, w, _, yaw = out_bbox.unbind(-1)
#     sin_yaw, cos_yaw = yaw.sin(), yaw.cos()

#     ret = torch.stack([x, y, l, w, sin_yaw, cos_yaw], dim=-1)
#     mask = ret[:, 2:4] < 0
#     ret[:, 2:4][mask] += -ret[:, 2:4][mask].detach()

#     return ret


# def convert_to_xyzlwhsc(out_bbox: torch.Tensor):
#     # x, y, l, w, yaw = out_bbox.unbind(-1)
#     x, y, z, l, w, h, yaw = out_bbox.unbind(-1)
#     sin_yaw, cos_yaw = yaw.sin(), yaw.cos()

#     ret = torch.stack([x, y, z, l, w, h, sin_yaw, cos_yaw], dim=-1)
#     mask = ret[:, 3:6] < 0
#     ret[:, 3:6][mask] += -ret[:, 3:6][mask].detach()

#     return ret


# class MatchingLoss(nn.Module):
#     def __init__(
#         self,
#         voxel_cfg: VoxelizerConfig,
#         active_classes: List,
#     ):
#         super().__init__()
#         self.voxel_config = voxel_cfg
#         self.active_classes = active_classes

#         self.x_min = voxel_cfg.x_min
#         self.x_max = voxel_cfg.x_max
#         self.y_min = voxel_cfg.y_min
#         self.y_max = voxel_cfg.y_max
#         self.z_min = voxel_cfg.z_min
#         self.z_max = voxel_cfg.z_max

#     def get_raw_bbox_labels(self, label: Sequence[Mapping[ActorClass, LabelData]]):
#         batch_size = len(label)
#         all_class_bboxes = []
#         for i in range(batch_size):
#             batch_i_label = label[i]
#             one_batch_boxes = list()
#             ignores = []
#             for actor_class in self.active_classes:
#                 box_data, ignore = self.process_label_data(batch_i_label[actor_class])
#                 one_batch_boxes.append(box_data)
#                 ignores.append(ignore)
#             # List batch_size x (Tensor num_box x 6)
#             one_frame_boxes = torch.cat(one_batch_boxes, dim=0)
#             ignores = torch.cat(ignores, dim=0)
#             all_class_bboxes.append(dict(boxes=one_frame_boxes, ignores=ignores))
#         return all_class_bboxes

#     def process_label_data(self, label_data: LabelData):
#         x, y = label_data.trajectories[:, 0, 0], label_data.trajectories[:, 0, 1]
#         l, w = label_data.boxes[:, 0], label_data.boxes[:, 1]
#         z = label_data.trajectories[:, 0, 2]
#         h = label_data.boxes[:, 2]

#         yaw_rad = label_data.yaw[:, 0]
#         ignores = label_data.ignores

#         # Remove actors out of ROI (+5m buffer)
#         buffer = 0
#         x_mask = torch.bitwise_and((self.x_min - buffer) < x, x < (self.x_max + buffer))
#         y_mask = torch.bitwise_and((self.y_min - buffer) < y, y < (self.y_max + buffer))
#         mask = torch.bitwise_and(x_mask, y_mask)

#         # z_mask = torch.bitwise_and((self.z_min - buffer) < z, z < (self.z_max + buffer))
#         # mask = torch.bitwise_and(mask, z_mask)

#         x, y, l, w, yaw_rad = x[mask], y[mask], l[mask], w[mask], yaw_rad[mask]

#         z, h = z[mask], h[mask]

#         ignores = ignores[mask]
#         bbox_label = torch.stack(
#             (
#                 x,
#                 y,
#                 z,
#                 l,
#                 w,
#                 h,
#                 yaw_rad,
#             ),
#             dim=1,
#         )
#         return bbox_label, ignores

#     def loss_labels(
#         self, outputs, targets, indices, num_boxes, dontcare_thresh=None, pos_thresh=None
#     ):  # pylint: disable=unused-argument
#         """Classification loss
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert "pred_logits" in outputs
#         src_logits = outputs["pred_logits"]

#         idx = self._get_src_permutation_idx(indices)
#         # src_logits: [batch_size, num_queries, 1]
#         target_scores = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
#         target_scores[idx] = 1

#         target_boxes = [t["boxes"][i] for t, (_, i) in zip(targets, indices)]
#         target_ignores = [t["ignores"][i] for t, (_, i) in zip(targets, indices)]

#         with torch.no_grad():
#             ignores = []
#             for i in range(src_logits.shape[0]):

#                 target_bboxes = detr_bbox_transform(
#                     target_boxes[i].new_zeros((1, target_boxes[i].shape[0], 1)),
#                     target_boxes[i].unsqueeze(0),
#                 )
#                 if dontcare_thresh is not None:
#                     all_bboxes = detr_bbox_transform(outputs["pred_logits"], outputs["pred_boxes"])
#                     riou = rbox_iou_2sets(all_bboxes[i][:, 1:], all_bboxes[i][indices[i][0]][:, 1:])
#                     ignore = riou.new_zeros((riou.shape[0],), dtype=torch.bool)
#                     ignore[((riou > dontcare_thresh).sum(dim=1) > 0) & (target_scores[i] == 0)] = True
#                 else:
#                     all_bboxes = detr_bbox_transform(outputs["pred_logits"], outputs["rois"])
#                     riou = rbox_iou_2sets(all_bboxes[i][:, 1:], target_bboxes[0, :, 1:])
#                     ignore = riou.new_zeros((riou.shape[0],), dtype=torch.bool)
#                     ignore[((riou > pos_thresh).sum(dim=1) == 0) & (target_scores[i] == 1)] = True

#                 ignore[indices[i][0][target_ignores[i]]] = True
#                 ignores.append(ignore)

#         loss_ce = sigmoid_focal_loss(
#             src_logits.squeeze(-1),
#             target_scores.float(),
#             alpha=0.5,
#             gamma=2,
#             reduction="none",
#         )
#         loss_ce = torch.cat([loss_ce[i][~ignores[i]] for i in range(len(ignores))])
#         loss_ce = loss_ce.sum() / num_boxes

#         return loss_ce

#     def loss_boxes(self, outputs, targets, indices, num_boxes, pos_thresh=None):
#         """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
#         targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
#         The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
#         """
#         assert "pred_boxes" in outputs
#         idx = self._get_src_permutation_idx(indices)
#         src_boxes = outputs["pred_boxes"][idx]
#         target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
#         target_ignores = torch.cat([t["ignores"][i] for t, (_, i) in zip(targets, indices)])

#         loss_l1 = nn.functional.l1_loss(
#             convert_to_xyzlwhsc(src_boxes), convert_to_xyzlwhsc(target_boxes), reduction="none"
#         )

#         loss_giou = 1 - rotation_robust_generalized_iou(
#             convert_to_xylwsc(src_boxes).transpose(0, 1),
#             convert_to_xylwsc(target_boxes).transpose(0, 1),
#         )

#         src_boxes = convert_to_xylwsc(src_boxes)
#         target_boxes = convert_to_xylwsc(target_boxes)

#         with torch.no_grad():
#             log_dict = dict()
#             riou = torch.diag(
#                 rbox_iou_2sets(
#                     torch.cat([src_boxes[:, :4], torch.atan2(src_boxes[:, [4]], src_boxes[:, [5]])], dim=-1),
#                     torch.cat([target_boxes[:, :4], torch.atan2(target_boxes[:, [4]], target_boxes[:, [5]])], dim=-1),
#                 )
#             )

#             log_dict["loss_recall_0.1_error"] = riou[riou > 0.1].numel() / (riou.numel() + 1e-5)
#             log_dict["loss_recall_0.5_error"] = riou[riou > 0.5].numel() / (riou.numel() + 1e-5)
#             log_dict["loss_recall_0.7_error"] = riou[riou > 0.7].numel() / (riou.numel() + 1e-5)

#         if pos_thresh is None:
#             loss_l1 = loss_l1[~target_ignores, :]
#             loss_l1 = loss_l1.sum() / num_boxes
#             loss_giou = loss_giou[:, ~target_ignores]
#             loss_giou = loss_giou.sum() / num_boxes
#         else:
#             with torch.no_grad():
#                 src_boxes = convert_to_xylwsc(outputs["rois"][idx])
#                 riou = torch.diag(
#                     rbox_iou_2sets(
#                         torch.cat([src_boxes[:, :4], torch.atan2(src_boxes[:, [4]], src_boxes[:, [5]])], dim=-1),
#                         torch.cat(
#                             [target_boxes[:, :4], torch.atan2(target_boxes[:, [4]], target_boxes[:, [5]])], dim=-1
#                         ),
#                     )
#                 )
#             loss_giou = loss_giou[0, (riou >= pos_thresh) & (~target_ignores)].sum() / num_boxes
#             loss_l1 = loss_l1[(riou >= pos_thresh) & (~target_ignores)].sum() / num_boxes

#         return loss_l1, loss_giou, log_dict

#     def _get_src_permutation_idx(self, indices):
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx

#     @torch.no_grad()
#     def matcher(self, outputs, targets, matching_cost_class, matching_cost_giou):
#         """Performs the matching
#         Params:
#             outputs: This is a dict that contains at least these entries:
#                 "pred_logit": Tensor of dim [batch_size, num_queries, 1] with object score logit
#                 "pred_boxes": Tensor of dim [batch_size, num_queries, 6] with the predicted box coordinates
#             targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
#                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
#                             objects in the target) containing the class labels
#                 "boxes": Tensor of dim [num_target_boxes, 6] containing the target box coordinates
#         Returns:
#             A list of size batch_size, containing tuples of (index_i, index_j) where:
#                 - index_i is the indices of the selected predictions (in order)
#                 - index_j is the indices of the corresponding selected targets (in order)
#             For each batch element, it holds:
#                 len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
#         """
#         # Compute the classification cost
#         alpha = 0.5
#         gamma = 2.0
#         pos_cost_class = sigmoid_focal_loss(
#             outputs["pred_logits"], torch.ones_like(outputs["pred_logits"]), alpha, gamma, "none"
#         )
#         neg_cost_class = sigmoid_focal_loss(
#             outputs["pred_logits"], torch.zeros_like(outputs["pred_logits"]), alpha, gamma, "none"
#         )

#         cost_class = pos_cost_class - neg_cost_class

#         cost_list = []

#         for i in range(outputs["pred_boxes"].shape[0]):
#             out_bbox = outputs["pred_boxes"][i]
#             tgt_bbox = targets[i]["boxes"]
#             cost_giou = (
#                 1
#                 - rotation_robust_generalized_iou(
#                     convert_to_xylwsc(out_bbox)
#                     .transpose(0, 1)[:, :, None]
#                     .repeat(1, 1, tgt_bbox.shape[0])
#                     .flatten(1, 2),
#                     convert_to_xylwsc(tgt_bbox)
#                     .transpose(0, 1)[:, None, :]
#                     .repeat(1, out_bbox.shape[0], 1)
#                     .flatten(1, 2),
#                 ).view(out_bbox.shape[0], tgt_bbox.shape[0])
#             )

#             cost_l1 = torch.cdist(convert_to_xyzlwhsc(out_bbox), convert_to_xyzlwhsc(tgt_bbox), p=1)

#             cost = matching_cost_class * cost_class[i] + cost_l1 * 0.1 + matching_cost_giou * cost_giou
#             cost_list.append(cost.cpu())

#         indices = [linear_sum_assignment(c) for c in cost_list]
#         return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

#     def forward(
#         self,
#         outputs,
#         targets,
#         dontcare_thresh=None,
#         pos_thresh=None,
#         cost_class=1.0,
#         cost_giou=4.0,
#         matching_cost_class=1.0,
#         matching_cost_giou=4.0,
#         indices=None,
#     ):
#         """This performs the loss computation.
#         Parameters:
#             outputs: dict of tensors, see the output specification of the model for the format
#             targets: list of dicts, such that len(targets) == batch_size.
#                     The expected keys in each dict depends on the losses applied, see each loss' doc
#         """
#         assert "pred_logits" in outputs and "pred_boxes" in outputs

#         # Retrieve the matching between the outputs of the last layer and the targets
#         if indices is None:
#             indices = self.matcher(outputs, targets, matching_cost_class, matching_cost_giou)

#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum((~t["ignores"]).sum() for t in targets).float()
#         num_boxes = dist.allreduce(num_boxes, op=dist.Op.AVERAGE)
#         num_boxes = torch.clamp(num_boxes, min=1).item()

#         # Compute all the requested losses
#         loss_ce = self.loss_labels(outputs, targets, indices, num_boxes, dontcare_thresh, pos_thresh)
#         loss_l1, loss_giou, log_dict = self.loss_boxes(outputs, targets, indices, num_boxes, pos_thresh)

#         losses = {
#             "loss_ce": loss_ce.item(),
#             "loss_l1": loss_l1.item(),
#             "loss_giou": loss_giou.item(),
#         }
#         losses.update(log_dict)

#         total_loss = cost_class * loss_ce + 0.1 * loss_l1 + cost_giou * loss_giou

#         return total_loss, losses


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return tf.relu
    if activation == "gelu":
        return tf.gelu
    if activation == "glu":
        return tf.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        # self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

        self.cross_attn = MultiScaleDeformableAttention(
            d_model, num_levels=n_levels, num_heads=n_heads, num_points=n_points, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        # tgt2 = self.cross_attn(
        #     self.with_pos_embed(tgt, query_pos),
        #     reference_points,
        #     src,
        #     src_spatial_shapes,
        #     level_start_index,
        #     src_padding_mask,
        # )
        tgt2 = self.cross_attn(
            tgt,
            src,
            src,
            query_pos=query_pos,
            key_padding_mask=src_padding_mask,
            reference_points=reference_points,
            spatial_shapes=src_spatial_shapes,
            level_start_index=level_start_index,
        )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class TwoStageTransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer: DeformableTransformerDecoderLayer, num_layers=6, voxel_cfg=None, with_sensor_fusion=False
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.layers = _get_clones(decoder_layer, num_layers)
        if with_sensor_fusion:
            self.sensor_fusion_ffn = _get_clones(
                nn.Sequential(nn.Linear(128 + 128, 128), nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Linear(128, 128)),
                num_layers,
            )
            sensor_fusion_ffn_after = nn.Sequential(
                nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Linear(128, 128), nn.LayerNorm(128)
            )
            nn.init.constant_(sensor_fusion_ffn_after[-1].weight, 0)
            nn.init.constant_(sensor_fusion_ffn_after[-1].bias, 0)
            self.sensor_fusion_ffn_after = _get_clones(sensor_fusion_ffn_after, num_layers)
            self.sensor_fusion_layer = _get_clones(
                DeformableTransformerDecoderLayer(128, 128 * 4, n_levels=4, n_points=4), num_layers
            )

        self.register_buffer(
            "bbox_reg_scale",
            torch.tensor(
                [
                    [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1],
                    [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05],
                    [0.033, 0.033, 0.033, 0.067, 0.067, 0.067, 0.033],
                ]
            ),
        )
        self.reg_branches = None
        self.voxel_cfg = voxel_cfg

    def sensor_fusion_preparation(self, bboxes, image_transformer_input, image_mat, num_level):

        bs, _, _ = bboxes.shape
        reference_points = []
        pix_feats = []

        bev_pts = get_vertices3d(bboxes).transpose(-1, -2)

        im_feat = image_transformer_input["src_ori"]

        for cam_id in image_mat.keys():
            all_pix_pts = []
            all_mask = []
            for i in range(bs):

                pix_pts, _, mask = lidar_to_cam(
                    bev_pts[i],
                    image_mat[cam_id]["sensor2cam"][i],
                    image_mat[cam_id]["intrinsics"][i],
                    image_mat[cam_id]["aug_transform"][i],
                    filter_outlier=False,
                )
                all_pix_pts.append(pix_pts)
                all_mask.append(mask)

            pix_pts = torch.stack(all_pix_pts)
            mask = torch.stack(all_mask)
            mask = mask.sum(dim=-1) >= 4

            x_min = pix_pts[..., 0].min(dim=-1)[0]
            y_min = pix_pts[..., 1].min(dim=-1)[0]
            x_max = pix_pts[..., 0].max(dim=-1)[0]
            y_max = pix_pts[..., 1].max(dim=-1)[0]

            pix_pts = torch.stack([(x_min + x_max) / 2, (y_min + y_max) / 2], dim=-1)
            box = torch.stack([x_max - x_min, y_max - y_min], dim=-1)

            pix_pts = pix_pts / torch.tensor([1920, 1056], device=pix_pts.device)  # * 2 - 1
            pix_feat_pts = pix_pts * 2 - 1
            pix_pts[~mask] = -100
            pix_feat_pts[~mask] = -100
            pix_feats.append(F.grid_sample(im_feat[cam_id][0], pix_feat_pts.unsqueeze(2)).squeeze(-1).permute(0, 2, 1))
            box = box / torch.tensor([1920, 1056], device=box.device)
            box[~mask] = 0
            reference_points.append(torch.cat([pix_pts, box], dim=-1).unsqueeze(2).repeat(1, 1, num_level, 1))
        reference_points = torch.cat(reference_points, dim=2)
        pix_feats = torch.cat(pix_feats, dim=-1)

        return reference_points, pix_feats

    def forward(
        self,
        query,
        pos_embed_ori,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        reg_branches,
        image_transformer_input=None,
        image_mat=None,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points.clone()
            if reference_points_input.shape[-1] == 7:
                reference_points_input[..., 0:1] = (reference_points_input[..., 0:1] - self.voxel_cfg[0]) / (
                    self.voxel_cfg[1] - self.voxel_cfg[0]
                )
                reference_points_input[..., 1:2] = (self.voxel_cfg[3] - reference_points_input[..., 1:2]) / (
                    self.voxel_cfg[3] - self.voxel_cfg[2]
                )
                reference_points_input[..., 3:4] = reference_points_input[..., 3:4] / (
                    self.voxel_cfg[1] - self.voxel_cfg[0]
                )
                reference_points_input[..., 4:5] = reference_points_input[..., 4:5] / (
                    self.voxel_cfg[3] - self.voxel_cfg[2]
                )
            else:
                reference_points_input[..., 0:1] = (reference_points_input[..., 0:1] - self.voxel_cfg[0]) / (
                    self.voxel_cfg[1] - self.voxel_cfg[0]
                )
                reference_points_input[..., 1:2] = (self.voxel_cfg[3] - reference_points_input[..., 1:2]) / (
                    self.voxel_cfg[3] - self.voxel_cfg[2]
                )
                reference_points_input[..., 2:3] = reference_points_input[..., 2:3] / (
                    self.voxel_cfg[1] - self.voxel_cfg[0]
                )
                reference_points_input[..., 3:4] = reference_points_input[..., 3:4] / (
                    self.voxel_cfg[3] - self.voxel_cfg[2]
                )
            query_pos = (
                F.grid_sample(pos_embed_ori, reference_points_input[..., :2].unsqueeze(-2) * 2 - 1)
                .squeeze(-1)
                .permute(0, 2, 1)
            )

            if image_transformer_input is not None:
                reference_points_image, pix_feats = self.sensor_fusion_preparation(
                    reference_points, image_transformer_input, image_mat, 4
                )
                output = output + self.sensor_fusion_ffn_after[lid](
                    self.sensor_fusion_layer[lid](
                        self.sensor_fusion_ffn[lid](torch.cat([output, pix_feats], dim=-1)),
                        query_pos,
                        reference_points_image,
                        image_transformer_input["src"],
                        image_transformer_input["src_spatial_shapes"],
                        image_transformer_input["level_start_index"],
                    )
                )

            reference_points_input = reference_points_input[:, :, None].repeat((1, 1, len(src_spatial_shapes), 1))

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, level_start_index)

            bbox_reg = reg_branches[lid](output)
            assert reference_points.shape[-1] == 5 or reference_points.shape[-1] == 7
            new_reference_points = reference_points + bbox_reg * self.bbox_reg_scale[lid]
            reference_points = new_reference_points.detach()

            intermediate.append(output)
            intermediate_reference_points.append(reference_points)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points)


class TwoStageTransformer(nn.Module):
    def __init__(
        self, embed_dims=128, num_decoder_layers=3, n_levels=3, voxel_cfg=None, with_sensor_fusion=False
    ) -> None:
        super().__init__()

        self.embed_dims = embed_dims
        self.voxel_cfg = voxel_cfg
        self.roi_size = 3

        self.decoder = TwoStageTransformerDecoder(
            DeformableTransformerDecoderLayer(self.embed_dims, self.embed_dims * 4, n_levels=n_levels, n_points=4),
            num_decoder_layers,
            voxel_cfg=voxel_cfg,
            with_sensor_fusion=with_sensor_fusion,
        )
        self.pos_embed = PositionEmbeddingSine()
        self.roi_align_rotated = RoIAlignRotated(
            output_size=[self.roi_size, self.roi_size], spatial_scale=1.0, sampling_ratio=1
        )
        self.roi_attn = nn.MultiheadAttention(embed_dims, 8, dropout=0.1, add_bias_kv=True)
        self.roi_ffn = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        self.roi_pos_embed = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
        )

        self.roi_proj = nn.Sequential(
            conv3x3(self.embed_dims, self.embed_dims),
            nn.BatchNorm2d(self.embed_dims),
        )
        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # for m in self.modules():
        #     if isinstance(m, MSDeformAttn):
        #         m.reset_parameters()

    @torch.no_grad()
    def gen_encoder_output_proposals(self, memory, spatial_shapes):
        batch, _, _ = memory.shape
        proposals = []
        for _, (height, width) in enumerate(spatial_shapes):
            mask_flatten_ = torch.zeros((batch, height, width, 1), device=memory.device, dtype=torch.bool)
            valid_h = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_w = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=memory.device),
                torch.arange(width, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1).view(batch, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(batch, -1, -1, -1) + 0.5) / scale
            proposal = grid.view(batch, -1, 2)
            proposals.append(proposal)

        output_proposals = torch.cat(proposals, 1)
        output_proposals = torch.cat(
            [
                output_proposals,  # x, y
                torch.zeros_like(output_proposals[..., :2]),  # z, l
                torch.zeros_like(output_proposals[..., :2]),  # w, h
                torch.zeros_like(output_proposals[..., :2]),  # s, c
            ],
            dim=-1,
        )
        return output_proposals

    def forward(
        self,
        mlvl_feats,
        image_transformer_input=None,
        image_mat=None,
        cls_branches=None,
        reg_branches=None,
        num_query=1000,
    ):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        src = []
        spatial_shapes = []
        for i in range(len(mlvl_feats)):
            spatial_shape = mlvl_feats[i].shape[2:]
            spatial_shapes.append(spatial_shape)
            src.append(mlvl_feats[i].flatten(2, 3).permute(0, 2, 1))
        src_flatten = torch.cat(src, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        enc_outputs_class = cls_branches[-1](mlvl_feats[0]).flatten(2, 3).permute(0, 2, 1)

        output_proposals = self.gen_encoder_output_proposals(src_flatten, [spatial_shapes[0]])
        output_proposals[..., 0:1] = self.voxel_cfg[0] + output_proposals[..., 0:1] * (
            self.voxel_cfg[1] - self.voxel_cfg[0]
        )
        output_proposals[..., 1:2] = self.voxel_cfg[3] - output_proposals[..., 1:2] * (
            self.voxel_cfg[3] - self.voxel_cfg[2]
        )
        output_proposals[..., 2:3] = self.voxel_cfg[4] + output_proposals[..., 2:3] * (
            self.voxel_cfg[5] - self.voxel_cfg[4]
        )
        # real coord
        enc_outputs_coord = reg_branches[-1](mlvl_feats[0]).flatten(2, 3).permute(0, 2, 1) + output_proposals

        topk = num_query
        pre_topk = topk * 2
        topk_proposals = torch.topk(enc_outputs_class[..., 0], pre_topk + topk, dim=1)[1]
        topk_coords = torch.gather(
            enc_outputs_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, enc_outputs_coord.shape[-1])
        )
        pos_embed = self.pos_embed(mlvl_feats[0])
        pos_embed_ori = pos_embed.clone()

        # nms
        bboxes_list = []
        bboxes_full_list = []
        topk_score = torch.gather(enc_outputs_class, 1, topk_proposals.unsqueeze(-1))
        topk_score = topk_score.detach()

        bboxes = torch.stack(
            [
                topk_coords[..., 0],
                topk_coords[..., 1],
                topk_coords[..., 3],
                topk_coords[..., 4],
                torch.atan2(topk_coords[..., 6], topk_coords[..., 7]),
            ],
            dim=-1,
        )
        bboxes_full = torch.stack(
            [
                topk_coords[..., 0],
                topk_coords[..., 1],
                topk_coords[..., 2],
                topk_coords[..., 3],
                topk_coords[..., 4],
                topk_coords[..., 5],
                torch.atan2(topk_coords[..., 6], topk_coords[..., 7]),
            ],
            dim=-1,
        )

        for i in range(topk_score.shape[0]):
            # keep_id = nms(torch.cat((topk_score[i][:pre_topk], bboxes[i][:pre_topk]), 1), 0.7)
            keep_id = nms_bev(bboxes[i][:pre_topk], topk_score[i][:pre_topk].squeeze(-1), 0.7, xywhr=True)
            if len(keep_id) >= topk:
                keep_id = keep_id[:topk]
            else:
                padding = torch.arange(pre_topk + len(keep_id), pre_topk + topk, device=keep_id.device)
                keep_id = torch.cat((keep_id, padding))
            bboxes_list.append(bboxes[i][keep_id])
            bboxes_full_list.append(bboxes_full[i][keep_id])
        bboxes = torch.stack(bboxes_list, dim=0)
        bboxes_full = torch.stack(bboxes_full_list, dim=0)

        bboxes_coords = get_roi_coords(bboxes, self.roi_size)

        # prepare for grid sample
        bboxes_coords[..., 0] = (bboxes_coords[..., 0] - self.voxel_cfg[0]) / (
            self.voxel_cfg[1] - self.voxel_cfg[0]
        ) * 2 - 1
        bboxes_coords[..., 1] = (self.voxel_cfg[3] - bboxes_coords[..., 1]) / (
            self.voxel_cfg[3] - self.voxel_cfg[2]
        ) * 2 - 1
        roi_pos_embed = F.grid_sample(pos_embed, bboxes_coords).permute(0, 2, 3, 1)

        roi_feats = get_roi_feats(
            self.roi_proj(mlvl_feats[0]),
            bboxes,
            self.voxel_cfg,
            self.roi_align_rotated,
        )

        query = roi_feats[:, :, self.roi_size * self.roi_size // 2]
        query_pos_embed = roi_pos_embed[:, :, self.roi_size * self.roi_size // 2]

        query = (
            query
            + self.roi_ffn(
                self.roi_attn(
                    query=query.view(1, -1, query.shape[-1]),
                    key=(roi_feats + self.roi_pos_embed(roi_pos_embed - query_pos_embed.unsqueeze(-2)))
                    .flatten(0, 1)
                    .permute(1, 0, 2),
                    value=roi_feats.flatten(0, 1).permute(1, 0, 2),
                )[0]
            ).view(query.shape)
        )

        query = F.relu(query, inplace=True)
        reference_points = bboxes_full
        init_reference_out = reference_points
        inter_states, inter_references = self.decoder(
            query=query,
            pos_embed_ori=pos_embed_ori,
            reference_points=reference_points,
            src=src_flatten,
            src_spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reg_branches=reg_branches,
            image_transformer_input=image_transformer_input,
            image_mat=image_mat,
        )

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord


class TwoStageDETRHead(nn.Module):
    def __init__(self, voxel_cfg, num_query, embed_dims, reg_dim=5, num_reg_fcs=2, with_sensor_fusion=False) -> None:
        super().__init__()
        self.voxel_cfg = voxel_cfg
        self.num_query = num_query
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.cls_out_channels = 1
        self.reg_dim = reg_dim
        self.with_sensor_fusion = with_sensor_fusion
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        hidden_dim = self.embed_dims
        propose_cls = nn.Sequential(
            conv3x3(self.embed_dims, hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            conv1x1(hidden_dim, 1, bias=True),
        )
        nn.init.constant_(propose_cls[-1].bias, -3.7)
        propose_reg = nn.Sequential(
            conv3x3(self.embed_dims, hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            conv3x3(hidden_dim, hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            conv1x1(hidden_dim, self.reg_dim + 1, bias=True),
        )

        cls_branch = []
        input_dim = self.embed_dims
        hidden_dim = self.embed_dims
        for _ in range(self.num_reg_fcs - 1):
            cls_branch.append(nn.Linear(input_dim, hidden_dim))
            cls_branch.append(nn.LayerNorm(hidden_dim))
            cls_branch.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
        cls_branch.append(nn.Linear(hidden_dim, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        input_dim = self.embed_dims
        hidden_dim = self.embed_dims
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(input_dim, hidden_dim))
            reg_branch.append(nn.LayerNorm(hidden_dim))
            reg_branch.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
        reg_branch.append(nn.Linear(hidden_dim, self.reg_dim))
        reg_branch = nn.Sequential(*reg_branch)

        self.transformer = TwoStageTransformer(
            n_levels=3, voxel_cfg=self.voxel_cfg, with_sensor_fusion=self.with_sensor_fusion
        )

        nn.init.normal_(reg_branch[-1].weight, 0, 0.001)
        nn.init.constant_(reg_branch[-1].bias, 0)

        num_pred = self.transformer.decoder.num_layers

        self.cls_branches = _get_clones(cls_branch, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)
        self.cls_branches.append(propose_cls)
        self.reg_branches.append(propose_reg)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    def forward(self, mlvl_feats, image_transformer_input=None, image_mat=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            mlvl_feats,
            image_transformer_input,
            image_mat,
            cls_branches=self.cls_branches,
            reg_branches=self.reg_branches,
            num_query=self.num_query,
        )
        outputs_classes = []
        outputs_coords = []
        outputs_rois = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            outputs_class = self.cls_branches[lvl](hs[lvl])
            bbox_reg = self.reg_branches[lvl](hs[lvl])
            outputs_coord = reference + bbox_reg * self.transformer.decoder.bbox_reg_scale[lvl]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_rois.append(reference)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_rois = torch.stack(outputs_rois)
        enc_outputs_coord = torch.cat(
            [
                enc_outputs_coord_unact[..., :-2],
                torch.atan2(enc_outputs_coord_unact[..., [-2]], enc_outputs_coord_unact[..., [-1]]),
            ],
            dim=-1,
        )

        outs = {
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "all_rois": outputs_rois,
            "enc_cls_scores": enc_outputs_class,
            "enc_bbox_preds": enc_outputs_coord,
        }
        return outs


def plot_rect3d_on_img(img, num_rects, rect_corners, color=(0, 255, 0), thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.
    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(
                img,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )

    return img.astype(np.uint8)


@DETECTORS.register_module()
class WaabiTwoStageDETRDetector(Base3DDetector):
    def __init__(
        self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None, init_cfg=None, pretrained=None
    ):
        super(WaabiTwoStageDETRDetector, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        # we reuse PerceptionModelConfig, but only detection related sub-configs will be used
        # self.voxel_cfg: VoxelizerConfig = cfg.voxel_cfg
        # self.class_cfg: MulticlassConfig = cfg.class_cfg
        # self.model_cfg: DetectorConfig = cfg.det_cfg
        self.det_has_pred = False

        # # Dimension of current frame detection box in refine header reg dim (default is 5, no need to change)
        # self.box_reg_dim = cfg.det_loss_cfg.box_reg_dim

        # # Regression dimension for refine header
        # # Reg dim = box_reg_dim (5) + [optional - for tracking] past_track_reg_dim (3) * past sweeps(4)
        # self.reg_dim = cfg.det_cfg.header.reg_dim

        # # Length of past frames track output
        # self.past_track_len = (self.reg_dim - self.box_reg_dim) // 3

        # # If true, then the refine header will predict both current box and past tracks. False by default
        # self.has_track = cfg.has_track

        # # If true, we compute track loss on the past tracks output of the refine header. False by default
        # self.has_track_loss = cfg.det_loss_cfg.has_track_loss

        # # build voxelizer
        # if "sparse" in self.model_cfg.backbone.name.lower():
        #     self.voxelizer = DynamicVoxelizer(self.voxel_cfg)
        # else:
        #     self.voxelizer: Union[IntensityVoxelizer, Voxelizer] = (
        #         IntensityVoxelizer(self.voxel_cfg)
        #         if self.voxel_cfg.add_intensity
        #         or (self.voxel_cfg.add_ground and (self.voxel_cfg.z_max - self.voxel_cfg.z_min) > 0)
        #         else Voxelizer(self.voxel_cfg)
        #     )
        self.bev_range = [0, 70, -40, 40, -3, 1]

        self.voxelizer = Voxelizer(
            self.bev_range[0], self.bev_range[1], self.bev_range[2], self.bev_range[3], 0.15625, -3.5, 3.5, 0.2
        )

        # Bring raster cfg out of cfg so we can easily call it from jit models
        # if isinstance(cfg.raster_cfg, RasterizerConfig):
        #     self.raster_cfg = cfg.raster_cfg
        # else:
        #     self.raster_cfg = OmegaConf.to_object(cfg.raster_cfg)
        # # Raster config for jit to store
        # self.jit_raster_cfg = self.raster_cfg.to_jit_object()

        # build detection model
        self.num_dets = 1000
        # self.backbone = backbones.build_backbone(self.model_cfg.backbone)
        self.backbone = build_backbone(backbone)
        # self.neck = necks.build_neck(self.model_cfg.neck)

        # if self.model_cfg.image_cfg is not None:
        #     depth = int(self.model_cfg.image_cfg.backbone.name.split("_")[-1])
        #     with TemporaryDirectory() as tmpdirname:
        #         pretrained_weights_dst_path = os.path.join(tmpdirname, "weights.pth.tar")
        #         aws_s3_download_file(
        #             f"s3://waabi-model-zoo/resnet{depth}_imagenet/checkpoint.pth.tar", pretrained_weights_dst_path
        #         )
        #         backbone_weights = torch.load(pretrained_weights_dst_path, map_location="cpu")
        #     self.image_backbone = mmcv.cnn.resnet.ResNet(
        #         depth,
        #         frozen_stages=1,
        #         bn_eval=False,
        #         bn_frozen=False,
        #     )
        #     self.image_backbone.load_state_dict(backbone_weights, strict=False)

        #     self.image_neck = necks.build_neck(self.model_cfg.image_cfg.neck)
        #     self.cam_to_lidar_align = MSDeformAttn(d_model=128, n_levels=4 * 3)

        #     self.pos_embed = nn.ModuleDict()
        #     for k in self.model_cfg.image_cfg.camera_list:
        #         self.pos_embed[k] = PositionEmbeddingLearned(500, 500)
        #     self.pos_embed["lidar"] = PositionEmbeddingLearned(500, 500)
        #     self.sensor_fusion_proj = nn.Sequential(
        #         conv3x3(128 * len(self.model_cfg.image_cfg.camera_list) + 128, 128), nn.BatchNorm2d(128), nn.ReLU()
        #     )
        #     nn.init.constant_(self.sensor_fusion_proj[1].weight.data, 0)
        #     nn.init.constant_(self.sensor_fusion_proj[1].bias.data, 0)

        self.head = TwoStageDETRHead(
            self.bev_range, self.num_dets, 128, 7, with_sensor_fusion=False  # self.model_cfg.image_cfg is not None
        )  # xyzlwht
        # self.det_loss = MatchingLoss(
        #     self.voxel_cfg,
        #     self.class_cfg.active_classes,
        # )

        self.nms_threshold = 0.5  # self.model_cfg.nms_threshold
        self.score_threshold = 0.0  # self.model_cfg.score_threshold
        self.pre_nms_topk = 200  # self.model_cfg.pre_nms_topk
        self.nms_topk = 100  # self.model_cfg.nms_topk

        # # pnp output converter
        # self.num_pred_modes = cfg.pred_cfg.model_cfg.num_modes
        # self.pred_len = cfg.pred_cfg.model_cfg.pred_len
        # self.pred_target_dim = cfg.pred_cfg.model_cfg.pred_target
        # self.pnp_converter = PredictionPostProcessor(
        #     cfg.pred_cfg.postprocess_cfg,
        #     self.class_cfg,
        #     cfg.pred_label_type,
        #     cfg.pred_cfg.sweep_duration_secs,
        #     cfg.pred_cfg.pred_delta_t_secs,
        # )

        # self.active_classes = [ActorClass.VEHICLE]

        self.preprocessor = LidarVQGAN()
        self.preprocessor.load_state_dict(
            torch.load(
                # "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/vqvae_w_two_stage_detr.pth",
                "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/det_front_2022-09-21_03-41-58_vqvae_decoder_frozen/checkpoint/vqvae.pth",
                # "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/vqvae.pth",
                # "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/ae_baseline.pth",
                map_location="cpu",
            ),
            strict=False,
        )
        # self.preprocessor = None

    def forward_model(self, points, preprocessing_model=None):
        # bev = self.voxelizer(model_input.batched_lidar)
        # if self.training:
        #     bev = self.voxelizer(subsample_lidar(model_input.batched_lidar)[0])
        # else:
        bev = self.voxelizer(points)
        if self.preprocessor is not None:
            residual, _ = self.preprocessor.forward(bev)
            bev = (bev * 20 + residual).sigmoid()

            bev = (bev > 0.5).float()

        # assert not hasattr(self.backbone, "map_channels")
        # if preprocessing_model is not None:
        #     updated_bev, _ = preprocessing_model(preprocessing_model.patchify(bev.voxel_features))
        #     updated_bev = gumbel_sigmoid(preprocessing_model.unpatchify(updated_bev), hard=True)
        #     # updated_bev, _ = preprocessing_model(bev.voxel_features)
        #     # updated_bev = gumbel_sigmoid(updated_bev, hard=True)
        #     fm = self.neck(self.backbone(VoxelizerOutput(updated_bev)))
        # else:
        fm = self.backbone(bev)
        # if model_input.batched_image is not None:
        #     image_fm = {}
        #     for cam_id in model_input.batched_image.keys():
        #         image_fm[cam_id] = self.image_neck(self.image_backbone(model_input.batched_image[cam_id]))
        #     fm, image_transformer_input = self.sensor_fusion(fm, image_fm, model_input.batched_image_mat)
        #     return self.head(fm, image_transformer_input, model_input.batched_image_mat)
        # else:
        return self.head(fm)

    def sensor_fusion(self, fm: List[torch.Tensor], image_fm: Dict[int, torch.Tensor], image_mat):
        fm_pos_embed = self.pos_embed["lidar"](fm[0])
        bs, channel, height, width = fm[0].shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=fm[0].device),
            torch.arange(width, dtype=torch.float32, device=fm[0].device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        grid[..., 0:1] = self.voxel_cfg[0] + grid[..., 0:1] / width * (self.voxel_cfg[1] - self.voxel_cfg[0])
        grid[..., 1:2] = self.voxel_cfg[3] - grid[..., 1:2] / height * (self.voxel_cfg[3] - self.voxel_cfg[2])
        voxel_size = torch.tensor(
            [self.voxel_cfg.step * 4, self.voxel_cfg.step * 4, self.voxel_cfg.z_max - self.voxel_cfg.z_min],
            device=fm[0].device,
        )
        grid = torch.cat(
            [
                grid,
                grid.new_ones(grid.shape[:-1]).unsqueeze(-1) * (self.voxel_cfg.z_max + self.voxel_cfg.z_min) / 2,
            ],
            dim=-1,
        )
        num_sample_points = 4
        bev_pts = sample_points_from_bev(grid, num_sample_points, voxel_size)  # B x H x W x 10 x 3
        srcs = []
        all_attn_feat = []
        for cam_id in image_fm.keys():
            all_pix_pts = []
            all_mask = []
            for i in range(bs):
                pix_pts, _, mask = lidar_to_cam(
                    bev_pts,
                    image_mat[cam_id]["sensor2cam"][i],
                    image_mat[cam_id]["intrinsics"][i],
                    image_mat[cam_id]["aug_transform"][i],
                    filter_outlier=False,
                )

                all_pix_pts.append(pix_pts.flatten(0, 1))
                all_mask.append(mask.flatten(0, 1))

            pix_pts = torch.stack(all_pix_pts)
            mask = torch.stack(all_mask)
            pix_pts = pix_pts / torch.tensor([1920, 1056], device=pix_pts.device)
            pix_pts[~mask] = -100

            reference_points = pix_pts.repeat(1, 1, 3, 1)
            input_flatten = SpatialFusion.get_src_flatten(image_fm[cam_id] * 3)
            input_spatial_shapes = SpatialFusion.get_spatial_shapes(image_fm[cam_id] * 3)
            input_level_start_index = SpatialFusion.get_level_start_index(input_spatial_shapes)
            attn_feat = self.cam_to_lidar_align(
                (fm[0] + fm_pos_embed).flatten(2, 3).permute(0, 2, 1),  # B x #pts x C
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
            )
            srcs.extend(image_fm[cam_id])
            all_attn_feat.append(attn_feat.reshape(bs, height, width, channel))

        src_flatten = SpatialFusion.get_src_flatten(srcs)
        spatial_shapes = SpatialFusion.get_spatial_shapes(srcs)
        level_start_index = SpatialFusion.get_level_start_index(spatial_shapes)
        all_attn_feat = torch.cat(all_attn_feat, dim=-1).permute(0, 3, 1, 2)
        attn_feat = torch.cat([all_attn_feat, fm[0]], dim=1)
        fm[0] = fm[0] + self.sensor_fusion_proj(attn_feat)
        return fm, {
            "src": src_flatten,
            "src_ori": image_fm,
            "src_spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
        }

    def extract_feat(self, points, img_metas=None):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        x = self.backbone(points)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feats(self, points, img_metas):
        """Extract features of multiple samples."""
        return [self.extract_feat(pts, img_meta) for pts, img_meta in zip(points, img_metas)]

    def simple_test(self, points, img_metas, imgs=None, rescale=False):

        header_out = self.forward_model([points])

        # bbox_out = self.det_post_process(header_out)

        pred_scores = header_out["all_cls_scores"].sigmoid().mean(dim=0)
        bbox_out, _, _ = detr_det_postprocess(
            {
                "pred_logits": inverse_sigmoid(pred_scores),
                "pred_boxes": header_out["all_bbox_preds"][-1],
            },
            [0],
            0.0,
            nms_threshold=self.nms_threshold,
            nms_topk=self.nms_topk,
        )

        bbox_list = []

        for i in range(len(bbox_out)):
            # bboxes = torch.zeros((bbox_out[i][0].shape[0], 7))
            bboxes = bbox_out[i][0][:, :-1]
            # bboxes[:, 0:2] = bbox_out[i][0][:, 0:2]
            # bboxes[:, 2] -= 2
            # bboxes[:, 3:5] = bbox_out[i][0][:, 2:4]
            # bboxes[:, 5] = 2.1
            # bboxes[:, 6:7] = bbox_out[i][0][:, 4:5]
            bboxes = img_metas[i]["box_type_3d"](bboxes, box_dim=7)
            scores = bbox_out[i][0][:, -1]
            labels = torch.zeros_like(scores).long()

            bbox_list.append((bboxes, scores, labels))

        # import ipdb

        # ipdb.set_trace()

        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        pass

    # def convert_to_pnp_output(
    #     self, det_output: DetectionModelOutput, sweep_end_ns: List[torch.Tensor]
    # ) -> List[PnPTrajectoryOutput]:
    #     if self.det_has_pred:  # one-stage pnp
    #         trajs: Optional[List[Dict[int, torch.Tensor]]] = det_output.preds
    #         assert trajs is not None
    #         # one-stage pnp has uni-modal predictions only
    #         mode_scores: List[Dict[int, torch.Tensor]] = []
    #         for i in range(len(trajs)):
    #             mode_score: Dict[int, torch.Tensor] = {}
    #             for c, v in trajs[i].items():
    #                 n_det = v.size(0)
    #                 mode_score[c] = torch.zeros(n_det, 0, dtype=v.dtype, device=v.device)
    #             mode_scores.append(mode_score)
    #         pred_output = PredictionModelOutput(trajs=trajs, mode_scores=mode_scores, anchors=None)
    #     else:  # generate dummy pred_output for det_only model
    #         bboxes = det_output.bboxes
    #         assert bboxes is not None
    #         pred_output = generate_dummy_predictions(bboxes, self.num_pred_modes, self.pred_len, self.pred_target_dim)

    #     return self.pnp_converter(
    #         [int(lidar_timestamps_example[-1].item()) for lidar_timestamps_example in sweep_end_ns],
    #         det_output,
    #         pred_output,
    #     )

    # @torch.jit.unused
    # def loss(self, det_out, labels) -> Tuple[torch.Tensor, Dict[str, float]]:
    #     metas = {}

    #     gt = self.det_loss.get_raw_bbox_labels(labels)

    #     all_det_loss_list: List[torch.Tensor] = []

    #     det_loss, det_metas = self.det_loss(
    #         {
    #             "pred_logits": det_out["enc_cls_scores"],
    #             "pred_boxes": det_out["enc_bbox_preds"],
    #         },
    #         gt,
    #         dontcare_thresh=0.8,
    #         cost_class=3.0,
    #     )
    #     det_metas = {"det/0/enc_" + k: v for k, v in det_metas.items()}
    #     metas.update(det_metas)
    #     all_det_loss_list.append(det_loss)

    #     pos_thresh = [0.3, 0.4, 0.5]
    #     for i in range(det_out["all_cls_scores"].shape[0]):
    #         det_loss, det_metas = self.det_loss(
    #             {
    #                 "pred_logits": det_out["all_cls_scores"][i],
    #                 "pred_boxes": det_out["all_bbox_preds"][i],
    #                 "rois": det_out["all_rois"][i],
    #             },
    #             gt,
    #             pos_thresh=pos_thresh[i],
    #         )
    #         all_det_loss_list.append(det_loss)
    #         det_metas = {"det/0/layer{}_".format(i) + k: v for k, v in det_metas.items()}
    #         metas.update(det_metas)

    #     all_det_loss: torch.Tensor = cast(torch.Tensor, sum(all_det_loss_list))
    #     metas["detection_loss"] = all_det_loss.item()
    #     metas["total_loss"] = all_det_loss.item()
    #     return all_det_loss, metas

    # @torch.jit.unused
    # def train_iter(self, batched_frames: BatchedPnPInput) -> Tuple[torch.Tensor, Dict[str, float]]:
    #     self.train()
    #     model_input = PnPModelInput.from_batched_frames(batched_frames, self.raster_cfg)
    #     det_out = self.forward(model_input)

    #     total_loss, metas = self.loss(det_out, batched_frames.labels)
    #     return total_loss, metas

    # @staticmethod
    # @torch.jit.unused
    # def eval_iter(
    #     model: Union[nn.Module, torch.jit.ScriptModule],
    #     batched_frames: BatchedPnPInput,
    #     compute_loss: bool,
    #     **kwargs,  # pylint: disable=unused-argument
    # ):
    #     """Returns (model_output, model_output_label, model_output_loss) per batch.
    #     Outputs are aggregated in ../evaluator.py, evaluation metrics are then computed by self.compute_metrics().
    #     """
    #     model.eval()

    #     if hasattr(model, "raster_cfg"):
    #         model_input = PnPModelInput.from_batched_frames(batched_frames, model.raster_cfg)
    #     else:
    #         model_input = PnPModelInput.from_batched_frames(batched_frames)

    #     det_out = model.forward(model_input, kwargs.get("preprocessing_model", None))

    #     pred_scores = det_out["all_cls_scores"].sigmoid().mean(dim=0)
    #     bboxes, _, _ = detr_det_postprocess(
    #         {
    #             "pred_logits": inverse_sigmoid(pred_scores),
    #             "pred_boxes": det_out["all_bbox_preds"][-1],
    #         },
    #         model.active_classes,
    #         model.score_threshold,
    #         nms_threshold=model.nms_threshold,
    #         nms_topk=model.nms_topk,
    #     )

    #     det_output = DetectionModelOutput(
    #         bboxes=bboxes,
    #         tracks=None,
    #         preds=None,
    #         det_outs={1: {"a": 1}},  # to pass lint
    #         score_threshold=0.0,
    #         nms_threshold=0.3,
    #         pre_nms_topk=2000,
    #         nms_topk=200,
    #         det_feat=None,
    #         det_pair_feat=None,
    #         adv_loss=None,
    #     )

    #     pnp_traj_output = model.convert_to_pnp_output(det_output, batched_frames.sweep_end_ns)

    #     loss_metas: Dict[str, float] = {}
    #     if compute_loss and batched_frames.dense_labels is not None:
    #         assert not isinstance(model, torch.jit.ScriptModule)
    #         _, loss_metas = model.loss(det_out, batched_frames.labels)

    #     return {"pnp_traj": pnp_traj_output}, batched_frames.labels, loss_metas

    # @staticmethod
    # @torch.jit.unused
    # def compute_metrics(
    #     config: EvaluatorConfig,
    #     model_outputs: Dict[str, List],
    #     labels: List,
    #     logger: ExperimentLogger,
    #     metric_metadata: Optional[Sequence[PnPMetricMetadata]] = None,
    # ):
    #     del metric_metadata  # Unused
    #     metrics = {}
    #     pnp_outputs = model_outputs["pnp_traj"]
    #     if config.detection_metrics_config is not None:
    #         start = time.time()
    #         detection_metrics_runner = DetectionSequentialMetricsRunner.build(config.detection_metrics_config)
    #         detection_metrics = detection_metrics_runner.run(pnp_outputs, labels)

    #         logger.log_val_detection(detection_metrics)
    #         logger.print(f"det_metrics_time: {(time.time() - start):.3f}")
    #         metrics.update(detection_metrics)

    #     return metrics


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
