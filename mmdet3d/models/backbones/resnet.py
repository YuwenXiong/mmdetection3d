# Copyright (c) 2021-2022 Waabi Innovation. All rights reserved.

from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from mmdet3d.models.builder import BACKBONES

# import waabi.common.profiling as profile_wrapper
# from waabi.autonomy.pnp.nn import build_norm_layer, conv3x3, trunc_normal_
# from waabi.autonomy.pnp.perception.config import BackboneConfig
# from waabi.autonomy.pnp.perception.modules.backbones.blocks import BasicBlock, Bottleneck, ConvNeXt, all_block_type
# from waabi.autonomy.pnp.perception.ops.conv_wrapper.conv1x1 import CustomConv1x1Stride2, get_conv1x1
# from waabi.autonomy.pnp.perception.ops.voxelizer.voxelizer import VoxelizerOutput
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
import math


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = False
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def build_norm_layer(
    norm_type: str, num_features: int, channel_dim: Optional[int] = None, affine: bool = True, dim: int = 2
) -> nn.Module:
    """Return norm layer according to the given norm type"""
    assert norm_type in ["BN", "GN", "SyncBN", "None"]
    assert dim in [1, 2, 3], "Norm layer dim must be one of [1, 2, 3]"
    if norm_type == "BN":
        if dim == 1:
            return nn.BatchNorm1d(num_features, affine=affine)
        elif dim == 2:
            return nn.BatchNorm2d(num_features, affine=affine)
        elif dim == 3:
            return nn.BatchNorm3d(num_features, affine=affine)
    elif norm_type == "GN":
        return nn.GroupNorm(4, num_features, affine=affine)
    else:
        return nn.Identity()


class BasicBlock(nn.Module):
    """The BasicBlock module used in ResNet, it consists of two 3x3 conv layers"""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_type: str = "BN",
    ):

        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.norm1 = build_norm_layer(norm_type, planes)

        self.conv2 = conv3x3(planes, planes)
        self.norm2 = build_norm_layer(norm_type, planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    The Bottleneck module used in ResNet, it consists of three conv layers with 1x1 - 3x3 - 1x1,
    the expansion is fixed to be 4 to balance the computation cost of 1x1 and 3x3 conv in the module
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_type: str = "BN",
    ):
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation

        self.conv1 = conv1x1(inplanes, planes)
        self.norm1 = build_norm_layer(norm_type, planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        self.norm2 = build_norm_layer(norm_type, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.norm3 = build_norm_layer(norm_type, planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L135
    Apache-2.0 License
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


all_block_type: Dict[str, Type[Union[BasicBlock, Bottleneck]]] = {
    "BasicBlock": BasicBlock,
    "Bottleneck": Bottleneck,
}


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if mask is None:
            not_mask = x.new_ones((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool)
        else:
            not_mask = ~mask  # pylint: disable=invalid-unary-operand-type
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class SpatialFusionLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_levels, n_heads, n_points, dropout=0.1):
        # self attention over different scales
        super().__init__()
        # self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.self_attn = MultiScaleDeformableAttention(
            d_model, num_levels=n_levels, num_heads=n_heads, num_points=n_points, batch_first=True, dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=False)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        # src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index, padding_mask)
        # import ipdb

        # ipdb.set_trace()
        src2 = self.self_attn(
            src,
            src,
            src,
            query_pos=pos,
            key_padding_mask=padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        # src = src + self.dropout1(src2)
        src = self.norm1(src2)
        # ffn
        src = self.forward_ffn(src)

        return src


class SpatialFusion(nn.Module):
    def __init__(
        self,
        n_levels,
        num_spatial_fusion_layers,
        d_model=256,
        d_ffn=1024,
        n_heads=8,
        n_points=4,
        dropout=0.1,
    ):
        super().__init__()
        self.pos_embed_func = PositionEmbeddingSine(d_model // 2)
        self.spatial_level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        self.spatial_fusion = nn.ModuleList(
            [
                SpatialFusionLayer(d_model, d_ffn, n_levels, n_heads, n_points, dropout)
                for _ in range(num_spatial_fusion_layers)
            ]
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.spatial_level_embed)

    def get_reference_points(self, shapes):
        return self.calc_reference_points(shapes)

    @staticmethod
    def calc_reference_points(spatial_shapes):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):  # pylint: disable=invalid-name,unused-variable
            # (H, W)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=spatial_shapes.device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=spatial_shapes.device),
            )
            # (B, H*W)
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_

            # (B, H*W, 2)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # (B, sum H*W, 2)
        reference_points = torch.cat(reference_points_list, 1)
        # (B, sum H*W, 1, 2) -> (B, sum H*W, #lvl, 2)
        reference_points = reference_points[:, :, None].repeat((1, 1, len(spatial_shapes), 1))
        return reference_points

    def forward(self, srcs):
        """[summary]
        Args:
            srcs: List[Tensor] [B x C_i x H_i x W_i]
        """
        batch_size, _, _, _ = srcs[-1].shape

        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, src in enumerate(srcs):
            pos_embed = self.pos_embed_func(src)
            _, c, h, w = pos_embed.shape

            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            pos_embed = pos_embed.view(batch_size, c, h * w).transpose(1, 2).contiguous()
            pos_embed = pos_embed + self.spatial_level_embed[lvl].view(1, 1, -1)
            src = src.view(batch_size, -1, h * w).transpose(1, 2).contiguous()
            lvl_pos_embed_flatten.append(pos_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        reference_points = self.get_reference_points(spatial_shapes).repeat(batch_size, 1, 1, 1)
        for i in range(len(self.spatial_fusion)):
            src_flatten = self.spatial_fusion[i](
                src_flatten, lvl_pos_embed_flatten, reference_points, spatial_shapes, level_start_index
            )  # (B*T, #ele, C)

        # unflatten src
        unflatten_src = []
        for i in range(len(level_start_index)):
            st = level_start_index[i]
            ed = level_start_index[i + 1] if i + 1 != len(level_start_index) else src_flatten.shape[1]
            unflatten_src_i = (
                src_flatten[:, st:ed]
                .transpose(1, 2)
                .reshape(batch_size, -1, spatial_shapes[i][0], spatial_shapes[i][1])
            )
            unflatten_src.append(unflatten_src_i)

        return unflatten_src


@BACKBONES.register_module()
class ResNet(nn.Module):
    """The conventional ResNet module"""

    def __init__(
        self,
        in_channels: int,
        groups_in_stem: int,
        stride_in_stem: int,
        channels_in_stem: Tuple[int, ...],
        blocks_per_stage: Tuple[int, ...],
        block_type: str,
        channels_per_stage: Tuple[int, ...],
        strides_per_stage: Tuple[int, ...],
        dilations_per_stage: Tuple[int, ...],
        out_indices: Tuple[int, ...],
        norm_type: str = "BN",
        use_custom_1x1_conv: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = groups_in_stem
        self.channels_in_stem = channels_in_stem
        self.stride_in_stem = stride_in_stem

        self.num_stages = len(blocks_per_stage)
        assert self.num_stages >= 1
        self.blocks_per_stage = blocks_per_stage

        self.strides = strides_per_stage
        self.dilations = dilations_per_stage
        self.channels = channels_per_stage
        assert len(self.strides) == len(self.dilations) == len(self.channels) == self.num_stages
        assert block_type in all_block_type
        self.block = all_block_type[block_type]
        self.out_indices = [int(out_index) for out_index in out_indices]
        assert max(out_indices) < self.num_stages + 1
        self.norm_type = norm_type

        self.stem, self.stem_stride = self._make_stem_layer()
        res_layers = []
        inplanes = self.channels_in_stem[-1]
        for i, num_blocks in enumerate(blocks_per_stage):
            stride = self.strides[i]
            dilation = self.dilations[i]
            planes = self.channels[i]
            if block_type == "ConvNeXt":
                res_layer = self._make_convnext_layer(
                    inplanes,
                    planes,
                    num_blocks,
                    stride=stride,
                )
            else:
                res_layer = self._make_res_layer(
                    self.block,
                    inplanes,
                    planes,
                    num_blocks,
                    stride=stride,
                    dilation=dilation,
                    norm_type=self.norm_type,
                    use_custom_1x1_conv=use_custom_1x1_conv,
                )
            inplanes = planes * self.block.expansion
            res_layers.append(res_layer)
        self.res_layers = torch.nn.ModuleList(res_layers)
        self._init_weights()

        self.spatial_fusion = nn.ModuleList()
        self.temporal_fusion = nn.ModuleList()
        self.input_proj = nn.ModuleList()
        self.output_proj = nn.ModuleList()
        for i in range(self.num_stages):
            self.input_proj.append(
                nn.Sequential(
                    conv1x1(self.channels[i] * self.block.expansion, 128),
                    nn.GroupNorm(32, 128),
                    nn.ReLU(),
                )
            )
            self.output_proj.append(
                nn.Sequential(
                    conv1x1(128, self.channels[i] * self.block.expansion),
                    nn.GroupNorm(32, self.channels[i] * self.block.expansion),
                    nn.ReLU(),
                )
            )
            self.spatial_fusion.append(
                SpatialFusion(i + 1, num_spatial_fusion_layers=1, d_model=128, d_ffn=512, dropout=0.1)
            )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.norm3.weight, 0)
                nn.init.constant_(m.norm3.bias, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.norm2.weight, 0)
                nn.init.constant_(m.norm2.bias, 0)

    def _make_stem_layer(self):
        """In conventional ResNet, the stem layer has output stride = 4
        here we remove the max pooling layer since BEV images may require higher resolution
        """
        layers = []
        out_channels = 0
        for i, channels in enumerate(self.channels_in_stem):
            num_groups = self.num_groups if i == 0 else 1
            in_channels = self.in_channels if i == 0 else out_channels
            out_channels = channels * num_groups if i == 0 else channels
            stride = self.stride_in_stem if i == 0 else 1

            layers.append(conv3x3(in_channels, out_channels, stride=stride, groups=num_groups))
            layers.append(build_norm_layer(self.norm_type, out_channels))
            layers.append(nn.ReLU(inplace=True))

        stem = nn.Sequential(*layers)

        return stem, self.stride_in_stem

    def _make_res_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        norm_type="BN",
        use_custom_1x1_conv=True,
    ):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride=stride),
                build_norm_layer(norm_type, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                norm_type=norm_type,
            )
        )
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=inplanes, planes=planes, stride=1, dilation=dilation, norm_type=norm_type))

        return nn.Sequential(*layers)

    def forward(self, x) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor batch_size x in_channels x input_height x input_width): the input images

        Returns:
            List[torch.Tensor]: for each stages in out_indices, there will be an output feature map
                tensor (torch.Tensor batch_size x hidden_dim_i x output_height_i x output_width_i) in outs
        """
        outs = []
        attn_feat = []
        x = self.stem(x)
        if 0 in self.out_indices:
            outs.append(x)
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            x = self.input_proj[i](x)
            attn_feat.append(x)
            attn_feat = self.spatial_fusion[i](attn_feat)
            x = self.output_proj[i](attn_feat[-1])
        outs = [attn_feat[i - 1] for i in self.out_indices]
        return outs
