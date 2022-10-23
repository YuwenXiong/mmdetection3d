import copy
import functools
import math
import time
from os import sync
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pykeops.torch import Genred, generic_argkmin, generic_argmin
from pykeops.torch import LazyTensor
from scipy.cluster.vq import kmeans2
from sklearn.cluster import kmeans_plusplus
from torchvision.ops import sigmoid_focal_loss
# from waabi.autonomy.pnp.openset.detr.deformable_transformer import inverse_sigmoid
# from waabi.autonomy.pnp.perception.two_stage_detr import TwoStageDETR, detr_det_postprocess
# from waabi.autonomy.pnp.type.internal import DetectionModelOutput

# import waabi.common.distributed as dist
# from waabi.autonomy.data.dataloader import BatchedPnPInput
# from waabi.autonomy.pnp.config import EvaluatorConfig, setup_config
# from waabi.autonomy.pnp.perception.detection_net import PnPModelInput
# from waabi.autonomy.pnp.perception.ops.voxelizer.voxelizer import Voxelizer, VoxelizerOutput
# from waabi.autonomy.pnp.perception.two_stage import TwoStage, gt_from_label
# from waabi.autonomy.pnp.type.metadata.metric_metadata import PnPMetricMetadata
# from waabi.common.training.experiments import ExperimentLogger
# from waabi.metrics.detection.detection_runner import DetectionSequentialMetricsRunner
from pykeops.torch import Genred, generic_argmin, generic_argkmin

train_sparse = False
gumbel_sigmoid_coeff = 10
use_vq = True
novq_in_first2000 = True
curriculum = False
subsample = False
use_pair_label = True
enable_gan = False


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


class GaussianNoise(nn.Module):  # Try noise just for real or just for fake images.
    def __init__(self, std=0.1, decay_rate=0.0005):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).uniform_(0, self.std)
        else:
            return x


def inverse_sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * (((1 + p_t) ** gamma))

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


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
    y_soft = _sigmoid_sample(logits * gumbel_sigmoid_coeff, tau=tau)
    if hard:
        y_hard = torch.where(y_soft > 0.5, torch.ones_like(y_soft), torch.zeros_like(y_soft))
        y = y_hard.data - y_soft.data + y_soft
    else:
        y = y_soft
    return y


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        # self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        # h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # self.ffn = nn.Sequential(
        #     torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True),
        #     torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        # )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        # x = x + h_
        # h_ = self.ffn(x)

        return x + h_


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.norm = Normalize(in_channels)
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.with_conv:
            # pad = (0, 1, 0, 1)
            # x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.norm(x)
            x = nonlinearity(x)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class SimpleEncoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            down = nn.Module()
            down.downsample = Downsample(block_in, resamp_with_conv)
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down.block = block
            down.attn = attn
            curr_res = curr_res // 2
            self.down.append(down)

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            hs.append(self.down[i_level].downsample(hs[-1]))
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

        # middle
        h = hs[-1]
        return h


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # # downsampling
        # self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (ch_mult[0],) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            down = nn.Module()
            down.downsample = Downsample(block_in, resamp_with_conv)
            curr_res = curr_res // 2
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down.block = block
            down.attn = attn
            # if i_level != self.num_resolutions - 1:
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        # hs = [self.conv_in(x)]
        hs = [x]
        for i_level in range(self.num_resolutions):
            # if i_level != self.num_resolutions - 1:
            hs.append(self.down[i_level].downsample(hs[-1]))
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        ch_in=None,
        decoder_t=None,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1] if ch_in is None else ch_in
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=1, stride=1, padding=0)

        # # middle
        # self.mid = nn.Module()
        # self.mid.block_1 = ResnetBlock(
        #     in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        # )
        # # self.mid.attn_1 = AttnBlock(block_in)
        # self.mid.block_2 = ResnetBlock(
        #     in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        # )
        # self.mid.block_3 = ResnetBlock(
        #     in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        # )
        # self.norm_out_mid = Normalize(block_in)
        # self.conv_out_mid = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            curr_res = curr_res * 2
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            upsample = Upsample(block_in, resamp_with_conv)
            up.block = block
            up.attn = attn
            up.upsample = upsample
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        self.decoder_t = decoder_t

        # nn.init.constant_(self.conv_out_mid.bias, -5)
        nn.init.constant_(self.conv_out.bias, -5.0)
        # nn.init.normal_(self.conv_out.weight, 0, 0.1)
        # nn.init.constant_(self.conv_out.bias, 0)

    def forward(self, z, z_t, give_pre_end=False):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        h_t = self.decoder_t(z_t)
        h = self.conv_in(torch.cat([z, h_t], dim=1))

        # h = self.mid.block_1(h, temb)
        # h = self.mid.block_2(h, temb)
        # pre_end_h = self.mid.block_3(h, temb)
        # pre_end_h = self.norm_out_mid(pre_end_h)
        # pre_end_h = nonlinearity(pre_end_h)
        # pre_end_h = self.conv_out_mid(pre_end_h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            # if i_level != 0:
            h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end or give_pre_end:
            return h
        pre_end_h = h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, pre_end_h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (ch_mult[0],) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # curr_res = resolution // 2 ** (self.num_resolutions - 1)
        curr_res = 32
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            # if i_level != 0:
            up.upsample = Upsample(block_in, resamp_with_conv)
            curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        # nn.init.constant_(self.conv_out.bias, -0.49)
        # nn.init.normal_(self.conv_out.weight, 0, 0.1)
        # nn.init.constant_(self.conv_out.bias, 0)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True, dead_limit=256
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.dead_limit = dead_limit

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        # self.embedding.weight.data.uniform_(-0.22, 0.22)
        # self.embedding.weight.data.normal_(0, 0.4)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        self.register_buffer("code_age", torch.zeros(self.n_e) * 10000)
        self.register_buffer("code_usage", torch.zeros(self.n_e))
        self.register_buffer("data_initialized", torch.ones(1))
        # self.register_buffer("reservoir", torch.zeros(self.n_e * 10, e_dim))
        self.reservoir = torch.zeros(self.n_e * 10, e_dim)

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # if self.embedding.weight.requires_grad and use_vq and self.training:
        #     self.update_reservoir(z_flattened.detach())

        # if self.embedding.weight.requires_grad and use_vq:
        #     self.update_codebook(z_flattened, force_update=self.data_initialized.item() == 0)

        # z_flattened = LazyTensor(z_flattened[:, None, :])
        # all_zq = LazyTensor(self.embedding.weight / self.embedding.weight.norm(dim=-1, keepdim=True)[None, :, :].clamp(1e-7))
        # min_encoding_indices = (z_flattened | all_zq).argmax(dim=1).long().view(-1)
        # if z_flattened.shape[-1] == 1024:
        # else:
        topk_nn = generic_argkmin("SqDist(X,Y)", "a = Vi(1)", f"X = Vi({self.e_dim})", f"Y = Vj({self.e_dim})")
        min_encoding_indices = topk_nn(z_flattened, self.embedding.weight).squeeze()

        # min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        # z_q = z_q / z_q.norm(dim=-1, keepdim=True).clamp(1e-7)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # if not self.legacy:
        #     loss = -(self.beta * torch.mean((z_q.detach() * z).sum(dim=-1)) + torch.mean((z_q * z.detach()).sum(dim=-1)))
        # else:
        #     loss = -(torch.mean((z_q.detach() * z).sum(dim=-1)) + self.beta * torch.mean((z_q * z.detach()).sum(dim=-1)))

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        # code_idx = dist.allgather(min_encoding_indices).to(min_encoding_indices.device)

        # self.code_age += 1
        # self.code_age[code_idx] = 0
        # self.code_usage.index_add_(0, code_idx, torch.ones_like(code_idx).float())

        code_util = (self.code_age < self.dead_limit // 2).sum() / self.code_age.numel()
        code_age = self.code_age.mean()

        # self.update_codebook()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices, code_util, code_age)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

    def update_reservoir(self, z_flattened, sample=True):

        if sample:
            rp = torch.randperm(z_flattened.size(0))
            num_sample = self.reservoir.shape[0] // 100
            self.reservoir = torch.cat([self.reservoir[num_sample:].cpu(), z_flattened[rp[:num_sample]].cpu()])
        else:
            self.reservoir = torch.cat([self.reservoir[z_flattened.shape[0] :].cpu(), z_flattened.cpu()])

    def update_codebook(self, z_flattened, force_update=False):

        # dead_code_num = (self.code_age >= self.dead_limit).sum()
        dead_code_num = torch.tensor(self.embedding.weight.shape[0])
        if (self.training and (self.code_age >= self.dead_limit).sum() > self.n_e * 0.7) or force_update:
            if dead_code_num > 0:
                all_z = torch.cat([self.reservoir, self.embedding.weight[self.code_age < self.dead_limit].data.cpu()])
                if dist.rank() == 0:
                    print("running kmeans!!", dead_code_num.item())  # data driven initialization for the embeddings
                    best_dist = 1e10
                    best_kd = None
                    for i in range(5):
                        rp = torch.randperm(all_z.size(0))
                        init = torch.cat(
                            [
                                self.embedding.weight[self.code_age < self.dead_limit].data.cpu(),
                                all_z[rp][: (dead_code_num - (self.code_age < self.dead_limit).sum())],
                            ]
                        )
                        kd = kmeans2(
                            all_z[rp].data.cpu().numpy(),
                            init.data.cpu().numpy(),
                            minit="matrix",
                            # dead_code_num.item(),
                            # minit="points",
                            iter=50,
                        )
                        z_dist = (all_z[rp] - torch.from_numpy(kd[0][kd[1]]).to(all_z.device)).norm(dim=1).sum().item()
                        if np.unique(kd[1]).size == dead_code_num.item():
                            best_kd = kd
                            best_dist = z_dist
                            break
                        else:
                            if z_dist < best_dist:
                                best_dist = z_dist
                                best_kd = kd
                            print("empty cluster", z_dist)
                            continue
                    kd = best_kd
                    z_dist = best_dist

                    self.embedding.weight.data = torch.from_numpy(kd[0]).to(self.embedding.weight.device)

                    print("finish kmeans", z_dist)

            if force_update:
                self.data_initialized.fill_(1)

            dist.broadcast(self.embedding.weight, src=0)
            # self.code_age[self.code_age >= self.dead_limit] = 0
            self.code_age.fill_(0)

    # def update_codebook(self, z_flattened, force_update=False):

    #     dead_code_num = (self.code_age >= self.dead_limit).sum()
    #     if (self.training and dead_code_num > self.n_e * 0.3) or force_update:
    #         print("running kmeans!!", dead_code_num.item())  # data driven initialization for the embeddings
    #         if dead_code_num > 0:
    #             all_z = dist.allgather(z_flattened)
    #             if dist.rank() == 0:
    #                 all_z = torch.cat([all_z, self.embedding.weight[self.code_age < self.dead_limit].to(all_z.device)])
    #                 rp = torch.randperm(all_z.size(0))
    #                 kd = kmeans2(all_z[rp[:20000]].data.cpu().numpy(), dead_code_num.item(), minit="points")
    #                 self.embedding.weight.data[self.code_age >= self.dead_limit] = torch.from_numpy(kd[0]).to(
    #                     self.embedding.weight.device
    #                 )
    #                 # print()
    #                 # live_code = self.embedding.weight[self.code_age < self.dead_limit]

    #                 # new_code = live_code[torch.multinomial(self.code_usage[self.code_age < self.dead_limit], dead_code_num, replacement=True)]
    #                 # new_code = new_code + new_code.uniform_(-0.001, 0.001) * new_code
    #                 # self.embedding.weight.data[self.code_age >= self.dead_limit] = new_code

    #                 # # mean, std = live_code.mean(dim=0), live_code.std(dim=0)
    #                 # # a = live_code.min(dim=0)[0]
    #                 # # b = live_code.max(dim=0)[0]
    #                 # # self.embedding.weight.data[self.code_age >= self.dead_limit] = (
    #                 # #     self.embedding.weight.data[self.code_age >= self.dead_limit].uniform_(0, 1) * (b - a) + a
    #                 # # )
    #                 # # self.embedding.weight.data[self.code_age >= self.dead_limit] = self.embedding.weight.data[
    #                 # #     self.code_age >= self.dead_limit
    #                 # # ].mul_(std)
    #                 # # self.embedding.weight.data[self.code_age >= self.dead_limit] = self.embedding.weight.data[
    #                 # #     self.code_age >= self.dead_limit
    #                 # # ].add_(mean)

    #         if force_update:
    #             self.data_initialized.fill_(1)

    #         dist.broadcast(self.embedding.weight, src=0)
    #         self.code_age[self.code_age >= self.dead_limit] = 0


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        cfg = setup_config(Path(__file__).parent.parent / "configs", "two_stage_v1.1_512beam")
        self.model = TwoStage(cfg.perception_model)
        # cfg = setup_config(Path(__file__).parent.parent / "configs", "two_stage_detr_v1.1_512beam")
        # self.model = TwoStageDETR(cfg.perception_model)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = torch.load(
            "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/two_stage_v1.1_2022-09-24_03-26-17_v5_1sweep_zh_bottom_box/checkpoint/model_00025e.pth.tar",
            # "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/two_stage_v1.1_2022-09-23_01-33-59_3d_new_voxel_dense/checkpoint/model_00024e.pth.tar",
            # "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/two_stage_v1.1_2022-09-22_03-40-04_3d_new_voxel/checkpoint/model_00024e.pth.tar",
            # "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/two_stage_detr_v1.1_2022-09-16_16-05-17/checkpoint/model_00024e.pth.tar",
            # "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/two_stage_v1.1_2022-08-26_21-16-35_v5data_1sweep/checkpoint/model_00025e.pth.tar",
            # "/mnt/remote/shared_data/users/yuwen/arch_baselines_july/two_stage_v1.1_2022-08-26_01-05-55_v2data_1sweep/checkpoint/model_0021e.pth.tar",
            # "/mnt/remote/shared_data/users/yuwen/arch_baselines_july/two_stage_v1.1_2022-08-23_22-24-04_v3data/checkpoint/model_00025e.pth.tar",
            # "/mnt/remote/shared_data/users/yuwen/arch_baselines_july/two_stage_v1.1_2022-07-13_23-45-59/checkpoint/model_0025e.pth.tar",
            map_location=torch.device("cpu"),
        )["model"]

        print(self.model.load_state_dict(ckpt, strict=True))

    def forward(self, input, batched_frames, bev_range, input_ori=None):
        self.eval()
        fm = self.model.neck(self.model.backbone(VoxelizerOutput(input)))
        header_out = self.model.forward_header(fm.float(), bev_range)
        det_output = self.model.det_post_process(fm, header_out, postprocess=False)
        # det_out = self.model.head(fm, bev_range=bev_range)

        gt = gt_from_label(
            batched_frames.labels if not use_pair_label else batched_frames.pair_labels,
            self.model.bev_range if bev_range is None else bev_range,
            classes=self.model.active_classes,
        )

        total_loss, metas = self.model.det_loss(det_output.det_outs, gt)
        # total_loss, metas = self.model.loss(det_out, batched_frames.labels if not use_pair_label else batched_frames.pair_labels, bev_range=bev_range)

        metas["detection_loss"] = total_loss.item()
        metas["total_loss"] = total_loss.item()

        if input_ori is not None:
            with torch.no_grad():
                target_fm = self.model.neck(self.model.backbone(VoxelizerOutput(input_ori)))
            feat_loss = F.smooth_l1_loss(fm, target_fm) * 5
            # feat_loss = sum([F.smooth_l1_loss(x, y) * 2 for x, y in zip(fm, target_fm)])
            metas["det/0/feat_loss"] = feat_loss.item()
            total_loss = (total_loss, feat_loss)

        return total_loss, metas


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # norm_layer = nn.BatchNorm2d
        norm_layer = nn.Identity
        use_bias = True
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func != nn.BatchNorm2d
        # else:
        #     use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Dropout(0.2),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
            nn.AdaptiveAvgPool2d((1, 1)),
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        """Standard forward."""
        # input = input.unflatten(1, (1, -1))
        return self.main(input)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def bce_loss(logits_real, logits_fake, d_weight=None):
    # logits_real = torch.mean(torch.flatten(logits_real, start_dim=1), dim=-1)
    # logits_fake = torch.mean(torch.flatten(logits_fake, start_dim=1), dim=-1)
    loss_real = F.binary_cross_entropy_with_logits(
        logits_real, torch.ones_like(logits_real) * (d_weight if d_weight is not None else 1.0)
    )
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    return (loss_real + loss_fake) * 0.5


def ce_loss(logits_real, logits_fake):
    logits_real = torch.mean(torch.flatten(logits_real, start_dim=1), dim=-1, keepdim=True)
    logits_fake = torch.mean(torch.flatten(logits_fake, start_dim=1), dim=-1, keepdim=True)
    # loss_real = F.binary_cross_entropy(logits_real, torch.ones_like(logits_real)).mean()
    # loss_fake = F.binary_cross_entropy(logits_fake, torch.zeros_like(logits_real)).mean()
    return F.cross_entropy(
        torch.cat([logits_real, logits_fake], dim=-1), logits_real.new_zeros([logits_real.shape[0]], dtype=torch.long)
    )


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def subsample_lidar(batched_lidar, global_step):

    # step_sizes = [2000, 19500, 37000, 55500]
    # step_sizes = [2000, 19500, 37000, 74000]
    # step_sizes = [0, 1000, 2000, 3000]
    step_sizes = [0, 5000, 10000, 15000]
    stride = 1
    if global_step >= step_sizes[-1]:
        num_beam = 64
        stride = 8
    elif global_step < step_sizes[0]:
        num_beam = 512
        stride = 1
    else:
        for idx in range(len(step_sizes)):
            if global_step >= step_sizes[idx] and global_step < step_sizes[idx + 1]:
                num_beam = 512 // stride - (global_step - step_sizes[idx]) / (step_sizes[idx + 1] - step_sizes[idx]) * (
                    512 // stride - 512 // stride / 2
                )
                num_beam = int(np.clip(num_beam, 64, 511))
                break
            else:
                stride *= 2

    # num_beam = 512 - global_step * step_sizes / (512 / stride)
    # num_beam = ((global_step - step) - 25000) * 512 // stride
    # num_beam = int(np.clip(512 - global_step / 224, 64, 511))
    # stride = 1
    start_idx = 0

    preserved_beam = np.arange(start_idx, 512, stride)
    num_preserved_beam = 512 // stride

    while 512 // stride > num_beam:
        all_beam = np.arange(start_idx, 512, stride)
        stride *= 2
        preserved_beam = np.arange(start_idx, 512, stride)
        num_preserved_beam = 512 // stride

    if num_beam - num_preserved_beam > 0:
        # all_beam = np.arange(0, 512)
        assert num_beam - num_preserved_beam <= all_beam.size - preserved_beam.size
        remain_beam = np.random.permutation(np.setdiff1d(all_beam, preserved_beam))[: (num_beam - num_preserved_beam)]
        preserved_beam = np.concatenate([preserved_beam, remain_beam])
    preserved_beam.sort()
    # if len(preserved_beam) == 64:
    #     return batched_lidar, 64
    # else:
    #     preserved_beam = np.setdiff1d(preserved_beam, np.arange(start_idx, 512, 8))
    preserved_beam = preserved_beam.tolist()
    updated_batched_lidar = []
    for i in range(len(batched_lidar)):
        assert len(batched_lidar[i]) == 2, "currently only support single sweep with pair data"
        laser_id = batched_lidar[i][0][:, 4].long()
        masks = torch.any(torch.stack([torch.eq(laser_id, aelem) for aelem in preserved_beam], dim=0), dim=0)
        updated_batched_lidar.append([batched_lidar[i][0], batched_lidar[i][0][masks]])
        # updated_batched_lidar.append([batched_lidar[i][0], batched_lidar[i][0]])

    return updated_batched_lidar, len(preserved_beam)


def update_lidar(batched_lidar):
    all_beam = np.arange(0, 512)
    preserved_beam = np.arange(0, 512, 8)
    remain_beam = np.setdiff1d(all_beam, preserved_beam)
    preserved_beam = remain_beam
    updated_batched_lidar = []
    for i in range(len(batched_lidar)):
        assert len(batched_lidar[i]) >= 2, "currently only support single sweep with pair data"
        laser_id = batched_lidar[i][0][:, 4].long()
        masks = torch.any(torch.stack([torch.eq(laser_id, aelem) for aelem in preserved_beam], dim=0), dim=0)
        updated_batched_lidar.append(
            [torch.cat([batched_lidar[i][0][masks], batched_lidar[i][1]])] + batched_lidar[i][1:]
        )
        # updated_batched_lidar.append([batched_lidar[i][0], batched_lidar[i][0]])
    return updated_batched_lidar


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        disc_conditional=False,
        disc_ndf=64,
        disc_loss="hinge",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = LPIPS().eval()

        if enable_gan:
            self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, ndf=disc_ndf)
            self.no_grad_discriminator = copy.deepcopy(self.discriminator)
            self.gaussian_noise = GaussianNoise(std=0.015)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = bce_loss  # hinge_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        batched_frames=None,
        bev_range=None,
        sparse_x=None,
        xrec_t=None,
        x_t=None,
    ):

        meta = {}

        rec_coeff = 1
        reconstructions = reconstructions * rec_coeff
        # rec_loss = inverse_sigmoid_focal_loss(reconstructions, inputs, alpha=-1, reduction="mean") * 10
        # rec_loss = F.binary_cross_entropy_with_logits(reconstructions, inputs, reduction="mean") * 10
        smoothed_inputs = inputs.clone()
        # smoothed_inputs *= 0.995
        # smoothed_inputs[smoothed_inputs < 0.5] += 0.005
        rec_loss = F.binary_cross_entropy_with_logits(reconstructions, smoothed_inputs, reduction="mean") * 10
        # rec_loss_t = F.binary_cross_entropy_with_logits(xrec_t, x_t, reduction="mean") * 10
        # rec_loss = F.binary_cross_entropy_with_logits(reconstructions, inputs, reduction="none").reshape((-1)).topk(int(reconstructions.numel() * 0.9999), largest=False)[0].mean() * 10
        # reconstructions = gumbel_sigmoid(reconstructions + (0 if sparse_x is None else sparse_x * 10), hard=True)
        reconstructions = (reconstructions + (0 if sparse_x is None else sparse_x * 20)).sigmoid()
        # if sparse_x is not None:
        #     reconstructions = torch.where(sparse_x != 1.0, reconstructions, sparse_x)

        meta.update(
            {
                # "det/0/inputs": inputs.sum() / inputs.shape[0],
                "det/0/rec": reconstructions.detach().sum().item() / reconstructions.shape[0],
                "det/0/rec_diff": (reconstructions.detach() - inputs).abs().sum().item() / reconstructions.shape[0],
                "det/0/rec_iou": ((reconstructions >= 0.5) & (inputs == 1)).sum().item()
                / ((reconstructions >= 0.5) | (inputs == 1)).sum().item(),
                "det/0/quant_loss": codebook_loss.detach().mean().item(),
                "det/0/rec_loss": rec_loss.detach().mean().item(),
                # "det/0/rec_loss_t": rec_loss_t.detach().mean().item(),
            }
        )

        if self.perceptual_weight > 0 and (optimizer_idx == 0 or (optimizer_idx == 1 and self.training)):
            p_loss, p_loss_meta = self.perceptual_loss(reconstructions, batched_frames, bev_range, inputs)
            # p_weight = 0.1
            try:
                # p_weight = self.calculate_adaptive_weight(rec_loss, p_loss[0], last_layer=last_layer).item()
                p_weight = torch.tensor(0.03, device=reconstructions.device)
            except RuntimeError:
                assert not self.training
                p_weight = torch.tensor(1.0, device=reconstructions.device).item()
            # p_weight = p_weight * min(global_step / 50000, 1)
            # p_weight = p_weight * 5
            # p_weight = max(p_weight, 2.0)
            nll_loss = rec_loss + self.perceptual_weight * p_weight * p_loss[0] + p_loss[1]
            meta.update({"det/0/p_loss": sum(p_loss).detach().mean().item()})
            meta.update({"det/0/p_weight": p_weight})
            meta.update(p_loss_meta)
        else:
            nll_loss = rec_loss

        meta.update({"det/0/nll_loss": nll_loss.detach().mean().item()})

        loss = nll_loss
        if (global_step >= 2000 or not novq_in_first2000) and use_vq:
            loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        # loss = loss + rec_loss_t

        # if optimizer_idx == 1:
        #     logits_fake = self.discriminator(reconstructions.contiguous())
        #     g_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
        #     try:
        #         d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
        #         # sync_d_weight = dist.allreduce(d_weight).item()
        #         # if sync_d_weight < 0.1 and dist.allreduce(np.random.random()) > sync_d_weight * 10:
        #         #     optimizer_idx = 0
        #     except RuntimeError:
        #         assert not self.training
        #         d_weight = torch.tensor(0.0)

        if enable_gan:
            # disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            disc_factor = 1.0
            # now the GAN part
            if optimizer_idx == 0:
                # generator update
                self.no_grad_discriminator = copy.deepcopy(self.discriminator)
                logits_fake = self.no_grad_discriminator(reconstructions.contiguous())
                g_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))

                try:
                    d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer=last_layer).item()
                    d_weight = min(d_weight, 1.0)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(1.0).item()
                # print(d_weight)

                # with torch.no_grad():
                #     disc_loss = self.disc_loss(logits_real, logits_fake)
                #     d_loss = disc_factor * disc_loss
                # if self.training:
                #     ().backward(retain_graph=True)

                meta.update(
                    {
                        "det/0/d_weight": d_weight,
                        # "det/0/d_loss": d_loss.detach().mean().item(),
                        # "det/0/g_loss": g_loss.detach().mean().item(),
                        # "det/0/logits_real": logits_real.detach().mean().item(),
                        # "det/0/logits_fake": logits_fake.detach().mean().item(),
                        "det/0/disc_factor": torch.tensor(disc_factor).item(),
                    }
                )
                loss = loss + d_weight * disc_factor * g_loss
                # if global_step % 2000 == 0:
                #     self.gaussian_noise.decay_step()
                # logits_real = self.discriminator(
                #     torch.where(
                #         sparse_x != 1.0, self.gaussian_noise(inputs).contiguous().detach().clamp(0, 1), sparse_x
                #     )
                # )
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                disc_loss = self.disc_loss(logits_real, logits_fake, max(d_weight, np.random.random() / 2 + 0.5))
                d_loss = self.discriminator_weight * d_weight * disc_factor * disc_loss
                loss = loss + d_loss

                # if optimizer_idx == 1:
                # pass
                # # second pass for discriminator update

                # # inputs_new = inputs.clone()
                # # inputs_new[inputs != reconstructions] = gumbel_sigmoid(
                # #     inputs_new[inputs != reconstructions] * 12 - 6, hard=True
                # # )
                # # inputs = inputs_new

                # logits_real = self.discriminator(inputs.contiguous().detach())
                # logits_fake = self.discriminator(reconstructions.contiguous().detach())
                # g_loss = F.binary_cross_entropy_with_logits(
                #     self.discriminator(reconstructions.contiguous()), torch.ones_like(logits_fake)
                # )

                # try:
                #     d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer=last_layer).item()
                #     d_weight = min(d_weight, 1.0)
                # except RuntimeError:
                #     assert not self.training
                #     d_weight = torch.tensor(1.0).item()

                # disc_loss = self.disc_loss(logits_real, logits_fake)
                # d_loss = self.discriminator_weight * d_weight * disc_factor * disc_loss
                meta.update(
                    {
                        # "det/0/d_weight": d_weight,
                        "det/0/d_loss": d_loss.detach().mean().item(),
                        "det/0/g_loss": g_loss.detach().mean().item(),
                        "det/0/logits_real": logits_real.detach().mean().item(),
                        "det/0/logits_fake": logits_fake.detach().mean().item(),
                        # "det/0/disc_factor": torch.tensor(disc_factor).item(),
                    }
                )
                # loss = d_loss

        return loss, meta, optimizer_idx


class VQGAN(nn.Module):
    def __init__(
        self,
        n_embed,
        embed_dim,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        **kwargs,
    ):
        super().__init__()
        self.in_chans = 40
        z_channels = embed_dim
        hidden_dim = 128
        self.resolution = 512
        self.encoder_b = SimpleEncoder(
            double_z=False,
            z_channels=z_channels,
            resolution=self.resolution,
            in_channels=self.in_chans,
            out_ch=self.in_chans,
            ch=hidden_dim,
            ch_mult=[1, 2],
            # ch_mult=[1, 2, 2, 4],
            num_res_blocks=[1, 2],
            attn_resolutions=[32],
            # attn_resolutions=[32],
            dropout=0.0,
        )
        self.encoder_t = Encoder(
            double_z=False,
            z_channels=z_channels,
            resolution=self.resolution // 4,
            in_channels=hidden_dim * 2,
            out_ch=hidden_dim * 2,
            ch=hidden_dim,
            ch_mult=[2, 4],
            # ch_mult=[1, 2, 2, 4],
            num_res_blocks=[2, 2],
            attn_resolutions=[32],
            # attn_resolutions=[32],
            dropout=0.0,
        )

        # self.decoder = Decoder(
        #     double_z=False,
        #     z_channels=z_channels,
        #     resolution=256,
        #     in_channels=self.in_chans,
        #     out_ch=self.in_chans,
        #     ch=hidden_dim,
        #     ch_mult=[1, 1, 2, 4, 4],
        #     # ch_mult=[1, 2, 2, 4],
        #     num_res_blocks=3,
        #     attn_resolutions=[32, 16],
        #     # attn_resolutions=[32],
        #     dropout=0.0,
        # )

        decoder_t = Decoder(
            double_z=False,
            z_channels=z_channels,
            resolution=self.resolution // 4,
            in_channels=hidden_dim * 2,
            out_ch=hidden_dim * 2,
            ch=hidden_dim,
            ch_mult=[2, 4],
            # ch_mult=[1, 2, 2, 4],
            num_res_blocks=[2, 2],
            attn_resolutions=[32],
            # attn_resolutions=[32],
            dropout=0.0,
            give_pre_end=True,
        )

        self.decoder = SimpleDecoder(
            double_z=False,
            z_channels=embed_dim + hidden_dim * 2,
            resolution=self.resolution,
            in_channels=self.in_chans,
            out_ch=self.in_chans,
            ch=hidden_dim,
            ch_mult=[1, 2],
            # ch_in=hidden_dim * 2 + embed_dim,
            # ch_mult=[1, 2, 2, 4],
            num_res_blocks=[1, 2],
            attn_resolutions=[32],
            # attn_resolutions=[32],
            dropout=0.0,
            decoder_t=copy.deepcopy(decoder_t),
        )

        # import ipdb; ipdb.set_trace()

        # self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 8, stride=4, padding=2)
        self.upsample_t = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
        )
        # self.decoder_proj = nn.Sequential(
        #     nn.Conv2d(2048, 128, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )

        # self.loss = VQLPIPSWithDiscriminator(
        #     disc_conditional=False,
        #     disc_in_channels=self.in_chans,
        #     disc_start=500000,
        #     disc_weight=1.0,
        #     codebook_weight=10.0,
        #     perceptual_weight=0.0,
        #     disc_num_layers=3,
        #     disc_ndf=64,
        # )

        self.quantize_b = VectorQuantizer(
            n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape, legacy=True
        )
        self.quant_conv_b = torch.nn.Conv2d(hidden_dim * 2 + embed_dim // 4, embed_dim, 1)

        self.post_quant_conv_b = torch.nn.Conv2d(embed_dim, embed_dim, 1)

        self.quantize_t = VectorQuantizer(
            n_embed // 2, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape, legacy=True
        )
        self.quant_conv_t = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv_t = torch.nn.Conv2d(embed_dim, z_channels, 1)

        # self.voxelizer = Voxelizer(kwargs["cfg"].voxel_cfg)
        # voxel_cfg = copy.deepcopy(kwargs["cfg"].voxel_cfg)
        # voxel_cfg.step *= 4
        # self.voxelizer_t = Voxelizer(voxel_cfg)
        # self.aug = kornia.augmentation.RandomCrop((self.resolution, self.resolution), resample=kornia.Resample.NEAREST)
        # if train_sparse:
        #     self.scale = torch.nn.parameter.Parameter(torch.zeros((1,)), requires_grad=True)

        # self.sparse_encoder = copy.deepcopy(self.encoder)

        # self.sparse_quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)

        # self.sparse_code_idx_cls = torch.nn.Sequential(
        #     torch.nn.Linear(z_channels, z_channels),
        #     Normalize(z_channels),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Linear(z_channels, n_embed),
        # )

        # for p in self.quantize_b.parameters():
        #     p.requires_grad = False
        # for p in self.quantize_t.parameters():
        #     p.requires_grad = False

        # for p in self.post_quant_conv_t.parameters():
        #     p.requires_grad = False
        # for p in self.post_quant_conv_b.parameters():
        #     p.requires_grad = False
        # # for p in self.upsample_t.parameters():
        # #     p.requires_grad = False
        # for p in self.decoder.parameters():
        #     p.requires_grad = False
        # for p in self.parameters():
        #     p.requires_grad = False

        # for p in self.sparse_encoder.parameters():
        #     p.requires_grad = True

        # for p in self.sparse_quant_conv.parameters():
        #     p.requires_grad = True

        # for p in self.sparse_code_idx_cls.parameters():
        #     p.requires_grad = True
        print(self)

    def encode(self, x, global_step=100000000):
        enc_b = self.encoder_b(x)
        enc_t = self.encoder_t(enc_b)
        h_t = self.quant_conv_t(enc_t)
        # h_t = h_t / h_t.norm(dim=1, keepdim=True).clamp(1e-7)
        if global_step == 2000 and novq_in_first2000:
            self.quantize_t.data_initialized.fill_(0)
        quant_t, emb_loss_1, info_1 = self.quantize_t(h_t)
        if (global_step < 2000 and novq_in_first2000) or not use_vq:
            quant_t = h_t

        # dec_t = self.decoder_t(quant_t)
        dec_t = self.upsample_t(quant_t)
        if global_step < 2000:
            enc_b = torch.cat([dec_t, torch.zeros_like(enc_b)], dim=1)
        else:
            enc_b = torch.cat([dec_t, enc_b * min(1.0, (global_step - 2000) / 10000)], dim=1)

        h_b = self.quant_conv_b(enc_b)
        # h_b = h_b / h_b.norm(dim=1, keepdim=True).clamp(1e-7)
        if global_step == 2000 and novq_in_first2000:
            self.quantize_b.data_initialized.fill_(0)
        # import ipdb; ipdb.set_trace()
        quant_b, emb_loss_2, info_2 = self.quantize_b(h_b)
        if (global_step < 2000 and novq_in_first2000) or not use_vq:
            quant_b = h_b

        return quant_t, quant_b, emb_loss_1 + emb_loss_2, (info_1, info_2), h_t, h_b

    def encode_sparse(self, x):
        pre_h = self.sparse_encoder(x)
        h = self.sparse_quant_conv(pre_h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, h, pre_h

    def decode(self, quant_t, quant_b, give_pre_end=False):
        quant_t = self.post_quant_conv_t(quant_t)
        quant_b = self.post_quant_conv_b(quant_b)
        # quant_t = self.upsample_t(quant_t)
        # quant = torch.cat([quant_t, quant_b], dim=1)
        dec = self.decoder(quant_b, quant_t, give_pre_end)
        return dec  # , self.decoder_proj(quant)

    def decode_code(self, code_b, shape=None):
        quant_b = self.quantize.get_codebook_entry(code_b, shape)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant_t, quant_b, diff, _, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)
        return dec

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 256
        # x = F.unfold(imgs, p, stride=p).view(imgs.shape[0], -1, 256, 256, 4).permute(0, 4, 1, 2, 3).flatten(0, 1)
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(shape=(imgs.shape[0] * 4, self.in_chans, p, p))
        return x

    def unpatchify(self, imgs):

        p = 256
        imgs = imgs.unflatten(0, (-1, 4)).reshape(-1, 2, 2, self.in_chans, p, p)
        imgs = torch.einsum("nhwcpq->nchpwq", imgs)
        imgs = imgs.reshape(shape=(imgs.shape[0], self.in_chans, 2 * p, 2 * p))
        return imgs

    # @torch.jit.unused
    # def train_iter(
    #     self, batched_frames: BatchedPnPInput, global_step: int, optimizer_idx
    # ) -> Tuple[torch.Tensor, Dict[str, float]]:
    #     self.train()

    #     model_input = PnPModelInput.from_batched_frames(batched_frames, None, False)

    #     if train_sparse:
    #         batched_lidar = update_lidar(model_input.batched_lidar)
    #         if subsample:
    #             batched_lidar, num_beam = subsample_lidar(
    #                 batched_lidar, global_step=global_step if curriculum else 10000000
    #             )
    #         for p in batched_lidar:
    #             p[0][:, 2] += 0.4
    #             p[1][:, 2] += 0.4
    #         bev = self.voxelizer(batched_lidar)
    #     else:
    #         batched_lidar = update_lidar(model_input.batched_lidar)
    #         for p in batched_lidar:
    #             p[0][:, 2] += 0.4
    #             p[1][:, 2] += 0.4
    #         bev = self.voxelizer(batched_lidar)

    #     # x = self.aug(bev.voxel_features)

    #     # crop_range = self.aug._params["src"].clone()
    #     # x_min, y_max = crop_range[:, 0].chunk(2, dim=1)
    #     # x_min = x_min * self.voxelizer.step + self.voxelizer.x_min
    #     # y_max = self.voxelizer.y_max - y_max * self.voxelizer.step
    #     # x_max = x_min + self.resolution * self.voxelizer.step
    #     # y_min = y_max - self.resolution * self.voxelizer.step
    #     # bev_range = torch.cat([x_min, x_max, y_min, y_max], dim=-1).to(x.device)
    #     # bev_range = (0, 70, -35, 35)
    #     bev_range = None
    #     x = bev.voxel_features
    #     # x_t = self.voxelizer_t(batched_lidar).voxel_features.chunk(2, dim=1)[0]

    #     if train_sparse:

    #         x, sparse_x = x.chunk(2, dim=1)

    #         quant_t, quant_b, qloss, qinfo, h_t, h_b = self.encode(sparse_x, global_step)
    #         # (xrec, _), xrec_feat = self.decode(quant_t, quant_b)
    #         xrec, xrec_t = self.decode(quant_t, quant_b)
    #         # xrec_feat_cont = self.decode(h_t, h_b, give_pre_end=True)
    #         # # xrec_feat_diff_loss = (((xrec_feat - xrec_feat_cont) ** 2) * 10).mean()
    #         # xrec_feat_diff_loss = (xrec_feat - xrec_feat_cont).norm(dim=1).mean()

    #         metas = {}
    #         metas.update(
    #             {
    #                 "det/0/q_t_util": qinfo[0][3].detach().item(),
    #                 "det/0/q_t_age": qinfo[0][4].detach().item(),
    #                 "det/0/q_util": qinfo[1][3].detach().item(),
    #                 "det/0/q_age": qinfo[1][4].detach().item(),
    #                 "det/0/num_beam": num_beam if subsample else 64,
    #                 # "det/0/rec_feat_diff": xrec_feat_diff_loss.detach().item(),
    #             }
    #         )

    #         loss, log_dict, optimizer_idx = self.loss(
    #             qloss,
    #             x,
    #             xrec,
    #             optimizer_idx,
    #             global_step,
    #             last_layer=self.get_last_layer(),
    #             batched_frames=batched_frames,
    #             bev_range=bev_range,
    #             sparse_x=sparse_x,
    #             # xrec_feat=xrec_feat,
    #         )
    #         # loss = loss + xrec_feat_diff_loss

    #         metas.update(log_dict)

    #     else:
    #         x, sparse_x = x.chunk(2, dim=1)

    #         quant_t, quant_b, qloss, qinfo, h_t, h_b = self.encode(x, global_step)
    #         xrec, xrec_t = self.decode(quant_t, quant_b)

    #         metas = {}
    #         metas.update(
    #             {
    #                 "det/0/q_t_util": qinfo[0][3].detach().item(),
    #                 "det/0/q_t_age": qinfo[0][4].detach().item(),
    #                 "det/0/q_util": qinfo[1][3].detach().item(),
    #                 "det/0/q_age": qinfo[1][4].detach().item(),
    #             }
    #         )

    #         loss, log_dict, optimizer_idx = self.loss(
    #             qloss,
    #             x,
    #             xrec,
    #             optimizer_idx,
    #             global_step,
    #             last_layer=self.get_last_layer(),
    #             batched_frames=batched_frames,
    #             bev_range=bev_range,
    #             # xrec_t=xrec_t,
    #             # x_t=x_t,
    #         )

    #         metas.update(log_dict)

    #     return loss, metas, optimizer_idx

    # @staticmethod
    # @torch.jit.unused
    # def eval_iter(
    #     model: Union[nn.Module, torch.jit.ScriptModule],
    #     batched_frames: BatchedPnPInput,
    #     compute_loss: bool,
    #     global_step=0,
    # ):
    #     model.eval()

    #     model_input = PnPModelInput.from_batched_frames(batched_frames, None, False)

    #     if train_sparse:
    #         batched_lidar = update_lidar(model_input.batched_lidar)
    #         if subsample:
    #             batched_lidar, num_beam = subsample_lidar(
    #                 batched_lidar, global_step=global_step if curriculum else 10000000
    #             )
    #         for p in batched_lidar:
    #             p[0][:, 2] += 0.4
    #             p[1][:, 2] += 0.4
    #         bev = model.voxelizer(batched_lidar)
    #     else:
    #         batched_lidar = update_lidar(model_input.batched_lidar)
    #         for p in batched_lidar:
    #             p[0][:, 2] += 0.4
    #             p[1][:, 2] += 0.4
    #         bev = model.voxelizer(batched_lidar)

    #     x = bev.voxel_features
    #     # x_t = model.voxelizer_t(batched_lidar).voxel_features.chunk(2, dim=1)[0]

    #     if train_sparse:
    #         x, sparse_x = x.chunk(2, dim=1)

    #         quant_t, quant_b, qloss, qinfo, h_t, h_b = model.encode(sparse_x, global_step=global_step)
    #         # xrec = model.decode(quant_t, quant_b)
    #         # (xrec, _), xrec_feat = model.decode(quant_t, quant_b)
    #         xrec, xrec_t = model.decode(quant_t, quant_b)
    #         # xrec_feat_cont = model.decode(h_t, h_b, give_pre_end=True)
    #         # # xrec_feat_diff_loss = (((xrec_feat - xrec_feat_cont) ** 2) * 10).mean()
    #         # xrec_feat_diff_loss = (xrec_feat - xrec_feat_cont).norm(dim=1).mean()

    #         metas = {}
    #         metas.update(
    #             {
    #                 "det/0/q_util": qinfo[3].detach(),
    #                 "det/0/q_age": qinfo[4].detach(),
    #                 "det/0/num_beam": num_beam if subsample else 64,
    #                 # "det/0/rec_feat_diff": xrec_feat_diff_loss.detach().item(),
    #             }
    #         )
    #     else:
    #         x, sparse_x = x.chunk(2, dim=1)

    #         quant_t, quant_b, qloss, qinfo, h_t, h_b = model.encode(x, global_step=global_step)
    #         xrec, xrec_t = model.decode(quant_t, quant_b)
    #         metas = {}
    #         metas.update(
    #             {
    #                 "det/0/q_t_util": qinfo[0][3].detach().item(),
    #                 "det/0/q_t_age": qinfo[0][4].detach().item(),
    #                 "det/0/q_util": qinfo[1][3].detach().item(),
    #                 "det/0/q_age": qinfo[1][4].detach().item(),
    #             }
    #         )

    #     _, log_dict_ae, _ = model.loss(
    #         qloss,
    #         x,
    #         xrec,
    #         0,
    #         global_step,
    #         last_layer=model.get_last_layer(),
    #         batched_frames=batched_frames,
    #         bev_range=None,
    #         sparse_x=None if not train_sparse else sparse_x,
    #         # xrec_feat=xrec_feat,
    #         # xrec_t=xrec_t,
    #         # x_t=x_t,
    #     )

    #     metas.update(log_dict_ae)
    #     if enable_gan:
    #         _, log_dict_disc, _ = model.loss(qloss, x, xrec, 1, global_step, last_layer=model.get_last_layer())
    #         metas.update(log_dict_disc)

    #     # det_output = model.loss.perceptual_loss.model.forward(model_input, post_process=True)

    #     fm = model.loss.perceptual_loss.model.neck(
    #         model.loss.perceptual_loss.model.backbone(
    #             # VoxelizerOutput(gumbel_sigmoid(xrec + (0 if not train_sparse else sparse_x * 10), hard=True))
    #             VoxelizerOutput((xrec + (0 if not train_sparse else sparse_x * 20)).sigmoid())
    #         )
    #     )
    #     # fm = xrec_feat
    #     header_out = model.loss.perceptual_loss.model.forward_header(fm.float())
    #     det_output = model.loss.perceptual_loss.model.det_post_process(fm, header_out, True)
    #     # det_out = model.loss.perceptual_loss.model.head(fm)

    #     # pred_scores = det_out["all_cls_scores"].sigmoid().mean(dim=0)
    #     # bboxes, _, _ = detr_det_postprocess(
    #     #     {
    #     #         "pred_logits": inverse_sigmoid(pred_scores),
    #     #         "pred_boxes": det_out["all_bbox_preds"][-1],
    #     #     },
    #     #     model.loss.perceptual_loss.model.active_classes,
    #     #     model.loss.perceptual_loss.model.score_threshold,
    #     #     nms_threshold=model.loss.perceptual_loss.model.nms_threshold,
    #     #     nms_topk=model.loss.perceptual_loss.model.nms_topk,
    #     # )

    #     # det_output = DetectionModelOutput(
    #     #     bboxes=bboxes,
    #     #     tracks=None,
    #     #     preds=None,
    #     #     det_outs={1: {"a": 1}},  # to pass lint
    #     #     score_threshold=0.0,
    #     #     nms_threshold=0.3,
    #     #     pre_nms_topk=2000,
    #     #     nms_topk=200,
    #     #     det_feat=None,
    #     #     det_pair_feat=None,
    #     #     adv_loss=None,
    #     # )

    #     pnp_traj_output = model.loss.perceptual_loss.model.convert_to_pnp_output(
    #         det_output, batched_frames.sweep_end_ns
    #     )

    #     return (
    #         {"pnp_traj": pnp_traj_output},
    #         batched_frames.labels if not use_pair_label else batched_frames.pair_labels,
    #         metas,
    #     )

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

    def get_last_layer(self):
        return self.decoder.conv_out.weight


def LidarVQGAN(**kwargs):
    return VQGAN(1024, 1024, **kwargs)
