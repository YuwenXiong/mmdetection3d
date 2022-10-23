import copy
import functools
import math
import time
from os import sync
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import kornia
import kornia.augmentation
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

# from waabi.autonomy.pnp.domain_adaptation.mae import get_2d_sincos_pos_embed
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
from timm.models.vision_transformer import Block, PatchEmbed, Attention, DropPath, Mlp
from timm.models.swin_transformer import BasicLayer, PatchMerging

# from waabi.autonomy.pnp.perception.modules.backbones.resnet_with_attention import PositionEmbeddingSine

train_sparse = False
gumbel_sigmoid_coeff = 10
use_vq = True
novq_in_first2000 = False
curriculum = False
subsample = True
use_pair_label = True
enable_gan = False


def make_gaussian_kernel(sigma):
    # ks = int(sigma * 5)
    ks = 5
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks)
    gauss = torch.exp((-((ts / sigma) ** 2) / 2))
    kernel = gauss / gauss.sum()

    return kernel


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


def KMeans_cosine(x, c, Niter=50, verbose=True):
    """Implements Lloyd's algorithm for the Cosine similarity metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # c = x[:K, :].clone()  # Simplistic initialization for the centroids
    # Normalize the centroids for the cosine similarity:
    K = c.shape[0]
    c = torch.nn.functional.normalize(c, dim=1, p=2)

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
        cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Normalize the centroids, in place:
        c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

    if verbose:  # Fancy display -----------------------------------------------
        end = time.time()
        print(f"K-means for the cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:")
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c


def KMeans(x, c, Niter=50, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # c = x[:K, :].clone()  # Simplistic initialization for the centroids
    K = c.shape[0]

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl.clamp(1e-7)  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        end = time.time()
        print(f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:")
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c


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
        self.register_buffer("reservoir", torch.zeros(self.n_e * 10, e_dim))
        # self.reservoir = torch.zeros(self.n_e * 10, e_dim)

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
        # z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.reshape(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # if self.embedding.weight.requires_grad and use_vq and self.training:
        #     self.update_reservoir(z_flattened.detach())

        # if use_vq:
        #     self.update_codebook(z_flattened, force_update=self.data_initialized.item() == 0)

        # import ipdb; ipdb.set_trace()
        # z_flattened = LazyTensor(z_flattened[:, None, :])
        # all_zq = LazyTensor((self.embedding.weight / self.embedding.weight.norm(dim=-1, keepdim=True).clamp(1e-7))[None, :, :])
        # min_encoding_indices = (z_flattened | all_zq).argmax(dim=1).long().view(-1)
        # min_encoding_indices = torch.matmul(z_flattened, (self.embedding.weight / self.embedding.weight.norm(dim=-1, keepdim=True).clamp(1e-7)).T).max(dim=-1)[1]
        topk_nn = generic_argkmin("SqDist(X,Y)", "a = Vi(1)", f"X = Vi({self.e_dim})", f"Y = Vj({self.e_dim})")
        min_encoding_indices = topk_nn(z_flattened, self.embedding.weight).squeeze()
        # min_encoding_indices = z_q = self.update_code(z_flattened, min_encoding_indices)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # min_encoding_indices = torch.argmin(d, dim=1)

        # z_q = z_q / z_q.norm(dim=-1, keepdim=True).clamp(1e-7)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = (self.beta * torch.mean((z_q.detach() - z) ** 2), torch.mean((z_q - z.detach()) ** 2))
        else:
            loss = (torch.mean((z_q.detach() - z) ** 2), self.beta * torch.mean((z_q - z.detach()) ** 2))

        # if not self.legacy:
        #     loss = (-self.beta * torch.mean((z_q.detach() * z).sum(dim=-1)), -torch.mean((z_q * z.detach()).sum(dim=-1)))
        # else:
        #     loss = (-torch.mean((z_q.detach() * z).sum(dim=-1)),  -self.beta * torch.mean((z_q * z.detach()).sum(dim=-1)))

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # # reshape back to match original input shape
        # z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

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
            # self.reservoir = torch.cat([self.reservoir[num_sample:].cpu(), z_flattened[rp[:num_sample]].data.cpu()])
            self.reservoir = torch.cat([self.reservoir[num_sample:], z_flattened[rp[:num_sample]].data])
        else:
            self.reservoir = torch.cat([self.reservoir[z_flattened.shape[0] :], z_flattened.data])

    # def update_code(self, z, min_encoding_indices):

    #     dead_code_mask = self.code_age > self.dead_limit

    #     if self.training and dead_code_mask.sum() > 10:

    #         all_z = dist.allgather(z).to(z.device)
    #         all_z_q = dist.allgather(self.embedding(min_encoding_indices)).to(z.device)

    #         if dist.rank() == 0:
    #             z_dist = (all_z - all_z_q).norm(dim=-1)
    #             idx = torch.topk(z_dist, dead_code_mask.sum().long())[1]
    #             self.embedding.weight.data[dead_code_mask] = all_z[idx].data
    #             self.code_age[dead_code_mask] = 0

    #         dist.broadcast(self.embedding.weight, src=0)
    #         dist.broadcast(self.code_age, src=0)

    #         topk_nn = generic_argkmin("SqDist(X,Y)", "a = Vi(1)", "X = Vi(512)", "Y = Vj(512)")
    #         min_encoding_indices = topk_nn(z, self.embedding.weight).squeeze()

    #         return min_encoding_indices

    #     else:
    #         return min_encoding_indices

    # def update_codebook(self, z_flattened, force_update=False):

    #     # dead_code_num = (self.code_age >= self.dead_limit).sum()
    #     dead_code_num = torch.tensor(self.embedding.weight.shape[0])
    #     if (self.training and (self.code_age >= self.dead_limit).sum() > self.n_e * 0.5) or force_update:
    #         if dead_code_num > 0:
    #             live_code = self.embedding.weight[self.code_age < self.dead_limit].data.cpu()
    #             # live_code = (live_code / live_code.norm(dim=-1, keepdim=True)).data # .cpu()
    #             # all_z = torch.cat([self.reservoir, self.embedding.weight[self.code_age < self.dead_limit].data.cpu()])
    #             all_z = torch.cat([self.reservoir.cpu(), live_code])
    #             if dist.rank() == 0:
    #                 print("running kmeans!!", dead_code_num.item())  # data driven initialization for the embeddings
    #                 best_dist = 1e10
    #                 best_kd = None
    #                 for i in range(1):
    #                     rp = torch.randperm(all_z.size(0))
    #                     init = torch.cat(
    #                         [live_code, all_z[rp][: (dead_code_num - (self.code_age < self.dead_limit).sum())]]
    #                     )
    #                     c, kd = KMeans_cosine(all_z[rp].cuda(), init.data.cuda(), 50, verbose=False)
    #                     kd = kmeans2(
    #                         all_z[rp].data.cpu().numpy(),
    #                         init.data.cpu().numpy(),
    #                         minit="matrix",
    #                         # dead_code_num.item(),
    #                         # minit="points",
    #                         iter=50,
    #                     )
    #                     z_dist = (all_z[rp] - torch.from_numpy(kd[0][kd[1]]).to(all_z.device)).norm(dim=1).sum().item()
    #                     # z_dist = (all_z[rp] - kd[c]).norm(dim=1).sum().item()
    #                     # if torch.unique(kd[1]).size == dead_code_num.item():
    #                     #     best_kd = kd
    #                     #     best_dist = z_dist
    #                     #     break
    #                     # else:
    #                     #     if z_dist < best_dist:
    #                     #         best_dist = z_dist
    #                     #         best_kd = kd
    #                     #     print("empty cluster", z_dist)
    #                     #     continue
    #                 # kd = best_kd
    #                 # z_dist = best_dist

    #                 # self.embedding.weight.data = kd # torch.from_numpy(kd[0]).to(self.embedding.weight.device)
    #                 self.embedding.weight.data = torch.from_numpy(kd[0]).to(self.embedding.weight.device)

    #                 print("finish kmeans", z_dist)

    #         if force_update:
    #             self.data_initialized.fill_(1)

    #         dist.broadcast(self.embedding.weight, src=0)
    #         # self.code_age[self.code_age >= self.dead_limit] = 0
    #         self.code_age.fill_(0)

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


# class LPIPS(nn.Module):
#     # Learned perceptual metric
#     def __init__(self, use_dropout=True):
#         super().__init__()
#         cfg = setup_config(Path(__file__).parent.parent / "configs", "two_stage_v1.1_512beam")
#         self.model = TwoStage(cfg.perception_model)
#         # cfg = setup_config(Path(__file__).parent.parent / "configs", "two_stage_detr_v1.1_512beam")
#         # self.model = TwoStageDETR(cfg.perception_model)
#         self.load_from_pretrained()
#         for param in self.parameters():
#             param.requires_grad = False

#     def load_from_pretrained(self, name="vgg_lpips"):
#         ckpt = torch.load(
#             "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/two_stage_v1.1_2022-10-08_21-29-02_real_plus_nonoccluded_zh_bottom_box/checkpoint/model_0025e.pth.tar",
#             # "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/two_stage_v1.1_2022-09-24_03-26-17_v5_1sweep_zh_bottom_box/checkpoint/model_00025e.pth.tar",
#             # "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/two_stage_v1.1_2022-09-22_03-40-04_3d_new_voxel/checkpoint/model_00024e.pth.tar",
#             # "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/two_stage_detr_v1.1_2022-09-16_16-05-17/checkpoint/model_00024e.pth.tar",
#             # "/mnt/remote/shared_data/users/yuwen/arch_baselines_aug/two_stage_v1.1_2022-08-26_21-16-35_v5data_1sweep/checkpoint/model_00025e.pth.tar",
#             # "/mnt/remote/shared_data/users/yuwen/arch_baselines_july/two_stage_v1.1_2022-08-26_01-05-55_v2data_1sweep/checkpoint/model_0021e.pth.tar",
#             # "/mnt/remote/shared_data/users/yuwen/arch_baselines_july/two_stage_v1.1_2022-08-23_22-24-04_v3data/checkpoint/model_00025e.pth.tar",
#             # "/mnt/remote/shared_data/users/yuwen/arch_baselines_july/two_stage_v1.1_2022-07-13_23-45-59/checkpoint/model_0025e.pth.tar",
#             map_location=torch.device("cpu"),
#         )["model"]

#         print(self.model.load_state_dict(ckpt, strict=True))

#     def forward(self, input, batched_frames, bev_range, input_ori=None):
#         self.eval()
#         fm = self.model.neck(self.model.backbone(VoxelizerOutput(input)))
#         header_out = self.model.forward_header(fm.float(), bev_range)
#         det_output = self.model.det_post_process(fm, header_out, postprocess=False)
#         # det_out = self.model.head(fm, bev_range=bev_range)

#         gt = gt_from_label(
#             batched_frames.labels if not use_pair_label else batched_frames.pair_labels,
#             self.model.bev_range if bev_range is None else bev_range,
#             classes=self.model.active_classes,
#         )

#         total_loss, metas = self.model.det_loss(det_output.det_outs, gt)
#         # total_loss, metas = self.model.loss(det_out, batched_frames.labels if not use_pair_label else batched_frames.pair_labels, bev_range=bev_range)

#         metas["detection_loss"] = total_loss.item()
#         metas["total_loss"] = total_loss.item()

#         if input_ori is not None:
#             with torch.no_grad():
#                 target_fm = self.model.neck(self.model.backbone(VoxelizerOutput(input_ori)))
#             feat_loss = F.smooth_l1_loss(fm, target_fm) * 5
#             # feat_loss = sum([F.smooth_l1_loss(x, y) * 2 for x, y in zip(fm, target_fm)])
#             metas["det/0/feat_loss"] = feat_loss.item()
#             total_loss = (total_loss, feat_loss)

#         return total_loss, metas


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def subsample_lidar(batched_lidar, global_step):

    step_sizes = [2000, 19500, 37000, 55500]
    # step_sizes = [2000, 19500, 37000, 74000]
    # step_sizes = [0, 1000, 2000, 3000]
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
    # all_beam = np.arange(0, 512)
    # preserved_beam = np.arange(0, 512, 8)
    all_beam = np.arange(-1, 64)
    preserved_beam = np.arange(0, 64)
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
        # self.perceptual_loss = LPIPS().eval()

        # if enable_gan:
        #     self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels, n_layers=disc_num_layers, ndf=disc_ndf)
        #     self.no_grad_discriminator = copy.deepcopy(self.discriminator)
        #     self.gaussian_noise = GaussianNoise(std=0.015)
        # self.discriminator_iter_start = disc_start
        # if disc_loss == "hinge":
        #     self.disc_loss = bce_loss  # hinge_d_loss
        # else:
        #     raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
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
        xrec_feat=None,
        xrec_t=None,
        x_t=None,
        smoothed_x=None,
    ):

        meta = {}

        rec_coeff = 1
        reconstructions = reconstructions * rec_coeff
        # rec_loss = inverse_sigmoid_focal_loss(reconstructions, inputs, alpha=-1, reduction="mean") * 10
        rec_loss = F.binary_cross_entropy_with_logits(reconstructions, inputs, reduction="none") * 10
        if smoothed_x is not None:
            rec_loss = torch.minimum(
                rec_loss, F.binary_cross_entropy_with_logits(reconstructions, smoothed_x, reduction="none") * 10
            )
        rec_loss = rec_loss.mean()
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
                "det/0/rec_iou": ((reconstructions >= 0.5) & (inputs >= 0.5)).sum().item()
                / ((reconstructions >= 0.5) | (inputs >= 0.5)).sum().item(),
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
                p_weight = torch.tensor(0.015, device=reconstructions.device)
            except RuntimeError:
                assert not self.training
                if not self.training:
                    p_weight = torch.tensor(1.0, device=reconstructions.device).item()
                else:
                    p_weight = torch.tensor(0.03, device=reconstructions.device).item()
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


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.attn2 = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=attn_drop, add_bias_kv=True, batch_first=True
        )

        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, key):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.attn2(self.norm2(x), key=key, value=key)[0])
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class VQViT(nn.Module):
    def __init__(
        self,
        n_embed,
        codebook_dim,
        img_size=(512, 512),
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
        **kwargs,
    ):
        super().__init__()
        self.in_chans = 40
        # z_channels = embed_dim
        hidden_dim = 64

        patch_size = 8
        in_chans = 40
        embed_dim = 256
        num_heads = 4
        mlp_ratio = 4
        norm_layer = nn.LayerNorm
        depth = 6

        decoder_embed_dim = 512
        decoder_depth = 12
        decoder_num_heads = 8

        # self.input_size = img_size // (patch_size // 2)

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # self.pos_embed = PositionEmbeddingSine(embed_dim // 2)
        self.patch_embed = PatchEmbed(img_size, patch_size // 2, in_chans, embed_dim, norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        # self.blocks = nn.ModuleList(
        #     [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)]
        # )

        self.blocks = [
            BasicLayer(
                embed_dim,
                None,
                (img_size[0] // (patch_size // 2), img_size[1] // (patch_size // 2)),
                depth,
                num_heads=num_heads,
                window_size=8,
                downsample=PatchMerging,
            ),
            BasicLayer(
                embed_dim * 2,
                None,
                (img_size[0] // patch_size, img_size[1] // patch_size),
                depth,
                num_heads=num_heads * 2,
                window_size=8,
                downsample=None,
            ),
        ]

        self.blocks = nn.Sequential(*self.blocks)

        # self.norm = norm_layer(embed_dim * 2)
        # self.pre_quant = nn.Linear(embed_dim * 2, codebook_dim)
        self.norm = nn.Sequential(norm_layer(embed_dim * 2), nn.GELU())
        self.pre_quant = nn.Sequential(nn.Linear(embed_dim * 2, codebook_dim), norm_layer(codebook_dim))
        self.quantize_t = VectorQuantizer(
            n_embed, codebook_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape, legacy=False
        )

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(codebook_dim, decoder_embed_dim, bias=True)

        # self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches // 4, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList(
        #     [
        #         DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #         for i in range(decoder_depth)
        #     ]
        # )
        self.decoder_blocks = BasicLayer(
            decoder_embed_dim,
            None,
            (img_size[0] // patch_size, img_size[1] // patch_size),
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            window_size=8,
        )

        # self.decoder_pre_norm = norm_layer(decoder_embed_dim)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        # self.quantize_b = VectorQuantizer(
        #     n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape, legacy=True
        # )

        # self.post_quant_conv_b = torch.nn.Conv2d(embed_dim, z_channels, 1)

        # self.quant_conv_t = torch.nn.Conv2d(z_channels, embed_dim, 1)
        # self.post_quant_conv_t = torch.nn.Conv2d(embed_dim, z_channels, 1)

        # self.voxelizer = Voxelizer(kwargs["cfg"].voxel_cfg)
        # voxel_cfg = copy.deepcopy(kwargs["cfg"].voxel_cfg)
        # voxel_cfg.step *= 4
        # voxel_cfg.z_step *= 2
        # self.voxelizer_t = Voxelizer(voxel_cfg)

        self.initialize_weights()

        self.sparse_patch_embed = copy.deepcopy(self.patch_embed)
        self.sparse_blocks = copy.deepcopy(self.blocks)
        self.sparse_norm = copy.deepcopy(self.norm)
        self.sparse_pre_quant = copy.deepcopy(self.pre_quant)

        nn.init.constant_(self.decoder_pred.bias, -5)

        # k = make_gaussian_kernel(0.7)
        # k3d = torch.einsum('i,j,k->ijk', k, k, k)
        # k3d = k3d / k3d.sum()

        # self.register_buffer('k3d', k3d.unsqueeze(0).unsqueeze(0))

        self.aug = nn.Sequential(
            kornia.augmentation.RandomVerticalFlip(),
            # kornia.augmentation.RandomHorizontalFlip(),
            # kornia.augmentation.RandomCrop((256, 256), resample=kornia.Resample.NEAREST),
        )

        self.loss = VQLPIPSWithDiscriminator(
            disc_conditional=False,
            disc_in_channels=self.in_chans,
            disc_start=500000,
            disc_weight=1.0,
            codebook_weight=1.0,
            perceptual_weight=0.0,
            disc_num_layers=3,
            disc_ndf=64,
        )

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(
        #     self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=False
        # )
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(
        #     self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=False
        # )
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=0.02)
        # torch.nn.init.normal_(self.decoder_cls_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def encode(self, x, global_step=100000000):
    #     # embed patches
    #     x = self.patch_embed(x)
    #     # x = x + self.pos_embed(x.tranpose(1, 2).reshape(-1, 256, self.input_size, self.input_size))

    #     # add pos embed w/o cls token
    #     # x = x + self.pos_embed

    #     # # append cls token
    #     # cls_token = self.cls_token + self.pos_embed[:, :1, :]
    #     # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    #     # x = torch.cat((cls_tokens, x), dim=1)

    #     # apply Transformer blocks
    #     # for blk in self.blocks:
    #     #     x = blk(x)
    #     x = self.blocks(x)
    #     x = self.norm(x)
    #     x = self.pre_quant(x)

    #     # x = F.normalize(x, dim=-1, p=2)

    #     if not use_vq:
    #         quant = x
    #         emb_loss = torch.tensor(0.0)
    #         info = None
    #     else:
    #         quant, emb_loss, info = self.quantize_t(x)
    #     if use_vq and (global_step <= 2000):
    #         # coeff = (global_step - 56880) / 53518 + 0.568
    #         # coeff = global_step / 50000
    #         coeff = global_step / 2000
    #         # import ipdb; ipdb.set_trace()
    #         quant = x * (1 - coeff) + quant * coeff
    #         # if global_step > 65000:
    #         #     emb_loss = emb_loss[1] + emb_loss[0] * (global_step - 65000) / 15000
    #         # else:
    #         #     emb_loss = emb_loss[1]
    #         emb_loss = emb_loss[1] + emb_loss[0] * coeff
    #     else:
    #         emb_loss = emb_loss[1] + emb_loss[0]  #  * min(1.0, global_step / 2000)

    #     return quant, emb_loss * 10, info

    def quantize(self, x, global_step=100000000):

        if not use_vq:
            quant = x
            emb_loss = torch.tensor(0.0)
            info = None
        else:
            quant, emb_loss, info = self.quantize_t(x)
        if use_vq and (global_step <= 2000):
            # coeff = (global_step - 56880) / 53518 + 0.568
            # coeff = global_step / 50000
            coeff = global_step / 2000
            # import ipdb; ipdb.set_trace()
            quant = x * (1 - coeff) + quant * coeff
            # if global_step > 65000:
            #     emb_loss = emb_loss[1] + emb_loss[0] * (global_step - 65000) / 15000
            # else:
            #     emb_loss = emb_loss[1]
            emb_loss = emb_loss[1] + emb_loss[0] * coeff
        else:
            emb_loss = emb_loss[1] + emb_loss[0]  #  * min(1.0, global_step / 2000)

        # return quant, emb_loss * 10, info
        return x, emb_loss * 10, info

    def encode(self, x, global_step=100000000):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_quant(x)

        return x

    def sparse_encode(self, x, global_step=100000000):
        # embed patches
        x = self.sparse_patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.sparse_blocks(x)
        x = self.sparse_norm(x)
        x = self.sparse_pre_quant(x)

        return x


    def decode(self, x):

        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # # append cls token
        # cls_token = self.decoder_cls_token + self.decoder_pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # init_x = self.decoder_pre_norm(x)

        # apply Transformer blocks
        # for blk in self.decoder_blocks:
        #     x = blk(x, init_x)
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # # remove cls token
        # x = x[:, 1:, :]

        # import ipdb; ipdb.set_trace()

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def encode_sparse(self, x):
        pre_h = self.sparse_encoder(x)
        h = self.sparse_quant_conv(pre_h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, h, pre_h

    # def decode(self, quant_t, give_pre_end=False):
    #     # quant_t = self.post_quant_conv_t(quant_t)
    #     dec = self.decoder(quant_t)
    #     # quant_b = self.post_quant_conv_b(quant_b)
    #     # quant_t = self.upsample_t(quant_t)
    #     # quant = torch.cat([quant_t, quant_b], dim=1)
    #     # dec = self.decoder(quant, give_pre_end)
    #     return dec # , self.decoder_proj(quant)

    def decode_code(self, code_b, shape=None):
        quant_b = self.quantize.get_codebook_entry(code_b, shape)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant_t, diff, _ = self.quantize(self.sparse_encode(input))
        # quant_t = self.sparse_encode(input)
        # diff = None
        dec = self.decode(quant_t)
        rec = self.unpatchify(dec)
        return rec, diff

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
        p = 16
        # x = F.unfold(imgs, p, stride=p).view(imgs.shape[0], -1, 256, 256, 4).permute(0, 4, 1, 2, 3).flatten(0, 1)
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(shape=(imgs.shape[0] * 4, self.in_chans, p, p))
        return x

    def unpatchify(self, x):

        p = 8
        # h = w = int(x.shape[1] ** 0.5)
        h = 512 // p
        w = 512 // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p))

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

    #     x = self.aug(bev.voxel_features)
    #     # x = bev.voxel_features
    #     # x_t = self.voxelizer_t(batched_lidar).voxel_features.chunk(2, dim=1)[0]

    #     x, sparse_x = x.chunk(2, dim=1)

    #     quant, qloss, qinfo = self.encode(sparse_x, global_step=global_step)

    #     # import ipdb; ipdb.set_trace()
    #     # print(quant.shape)
    #     xrec = self.decode(quant)
    #     xrec = self.unpatchify(xrec)

    #     metas = {}
    #     if qinfo is not None:
    #         metas.update(
    #             {
    #                 "det/0/q_util": qinfo[3].detach().item(),
    #                 "det/0/q_age": qinfo[4].detach().item(),
    #             }
    #         )

    #     # import ipdb; ipdb.set_trace()

    #     # smoothed_x = F.conv3d(x.unflatten(1, (1, -1)), self.k3d, None, 1, self.k3d.shape[-1] // 2).flatten(1, 2)

    #     loss, log_dict, optimizer_idx = self.loss(
    #         qloss,
    #         x,
    #         xrec,
    #         optimizer_idx,
    #         global_step,
    #         last_layer=self.get_last_layer(),
    #         batched_frames=batched_frames,
    #         bev_range=None,
    #         # xrec_t=xrec_t,
    #         # x_t=x_t,
    #         # smoothed_x=smoothed_x
    #     )

    #     metas.update(log_dict)

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

    #     x, sparse_x = x.chunk(2, dim=1)

    #     quant, qloss, qinfo = model.encode(sparse_x, global_step=global_step)

    #     # import ipdb; ipdb.set_trace()
    #     xrec = model.decode(quant)
    #     xrec = model.unpatchify(xrec)

    #     metas = {}
    #     if qinfo is not None:
    #         metas.update(
    #             {
    #                 "det/0/q_util": qinfo[3].detach().item(),
    #                 "det/0/q_age": qinfo[4].detach().item(),
    #             }
    #         )

    #     loss, log_dict, optimizer_idx = model.loss(
    #         qloss,
    #         x,
    #         xrec,
    #         0,
    #         global_step,
    #         last_layer=model.get_last_layer(),
    #         batched_frames=batched_frames,
    #         bev_range=None,
    #         # xrec_t=xrec_t,
    #         # x_t=x_t,
    #     )

    #     metas.update(log_dict)

    #     fm = model.loss.perceptual_loss.model.neck(
    #         model.loss.perceptual_loss.model.backbone(
    #             # VoxelizerOutput(gumbel_sigmoid(xrec + (0 if not train_sparse else sparse_x * 10), hard=True))
    #             # VoxelizerOutput((xrec + (0 if not train_sparse else sparse_x * 20)).sigmoid())
    #             VoxelizerOutput((xrec + sparse_x * 20).sigmoid())
    #         )
    #     )
    #     # fm = xrec_feat
    #     header_out = model.loss.perceptual_loss.model.forward_header(fm.flo[at())
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
        return self.decoder_pred.weight


def LidarVQViT(**kwargs):
    return VQViT(1024, 1024, **kwargs)
    # return VQViT(8192, 2048, **kwargs)
