# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger
# Adapted by Fabio Milentiansen Sim


import torch
import torch.nn.functional as F
from torch import nn


def simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """Fast Non-maximum suppression to remove nearby points"""

    def max_pool(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    scores = scores[:, None]
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()).bool()
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        # 核心修改：替换逻辑运算符为PyTorch显式逻辑函数，适配RKNN算子
        not_supp_mask = torch.logical_not(supp_mask)  # 替换 ~supp_mask
        and_mask = torch.logical_and(new_max_mask, not_supp_mask)  # 替换 new_max_mask & (~supp_mask)
        max_mask = torch.logical_or(max_mask, and_mask)  # 替换 max_mask | (...)
    
    """for _ in range(2):
        supp_mask = max_pool(max_mask.float()).bool()
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))"""
    return torch.where(max_mask, scores, zeros)[:, 0]


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
    """

    weights_url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"

    def __init__(
        self,
        descriptor_dim: int = 256,
        nms_radius: int = 4,
        remove_borders: int | None = 4,
        num_keypoints: int = 1024,
    ):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.nms_radius = nms_radius
        self.remove_borders = remove_borders
        self.num_keypoints = num_keypoints

        if self.remove_borders is not None and self.remove_borders <= 0:
            raise ValueError("remove_borders must be positive or None")
        if self.num_keypoints <= 0:
            raise ValueError("num_keypoints must be positive")

        # Layers
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, self.descriptor_dim, kernel_size=1, stride=1, padding=0)

        # self.load_state_dict(torch.hub.load_state_dict_from_url(self.weights_url))
        state_dict = torch.load("/data/luoshiyong/code/LightGlue-ONNX-main/gim_lightglue_100h.ckpt", map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict.pop(k)
            if k.startswith('superpoint.'):
                state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
        self.load_state_dict(state_dict)
    def forward(
        self,
        image: torch.Tensor,  # (B, 1, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute keypoints, scores, descriptors for image"""
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = F.softmax(scores, 1)[:, :-1]  # 65 -> 64
        b, _, h, w = scores.shape  # C == 64
        s = 8  # scale factor (constant)
        scores = (
            scores.reshape(b, s, s, h, w)
            .permute(0, 3, 1, 4, 2)  # (B, H, S, W, S)
            .reshape(b, h * s, w * s)
        )

        scores = simple_nms(scores, self.nms_radius)  # (B, H, W)

        # Discard keypoints near the image borders
        if pad := self.remove_borders:
            scores[:, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, :pad] = -1
            scores[:, :, -pad:] = -1

        # Select top-K keypoints
        top_scores, top_indices = scores.reshape(b, h * s * w * s).topk(self.num_keypoints)
        if torch.jit.is_tracing():  # type: ignore
            print("not >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            """one = torch.tensor(1)  # Always constant, safe to ignore warning.
            top_indices = top_indices.unsqueeze(2).floor_divide(
                torch.stack([w * s, one]).to(device=top_indices.device)  # type: ignore
            ) % torch.stack([h * s, w * s]).to(device=top_indices.device)  # type: ignore"""
            # 定义这一层需要的除数和模数
            width = w * s
            height = h * s

            # 拆分操作：
            # 1. 对应原代码张量的第0维：(index // width) % height
            #    这是计算行坐标 (y)
            idx_0 = top_indices.div(width, rounding_mode='floor') % height

            # 2. 对应原代码张量的第1维：(index // 1) % width -> 即 index % width
            #    这是计算列坐标 (x)
            idx_1 = top_indices % width

            # 最后堆叠回去，形状恢复为 (..., 2)
            top_indices = torch.stack((idx_0, idx_1), dim=-1)
            print(top_indices)
        else:
            """top_indices = top_indices.unsqueeze(2).floor_divide(
                torch.tensor([w * s, 1], device=top_indices.device)
            ) % torch.tensor([h * s, w * s], device=top_indices.device)"""
            # print(top_indices)
            top_indices_div = top_indices.unsqueeze(2).floor_divide(
                torch.tensor([w * s, 1], device=top_indices.device)
            )
            # 拆分X/Y维度的Mod，分别用单元素除数
            mod_h = torch.tensor([h * s], device=top_indices.device)  # 单元素
            mod_w = torch.tensor([w * s], device=top_indices.device)  # 单元素
            # 对X/Y维度分别做Mod（广播单元素除数）
            top_indices_x = top_indices_div[..., 0] % mod_h  # X维度Mod
            top_indices_y = top_indices_div[..., 1] % mod_w  # Y维度Mod
            # 合并回原形状
            top_indices = torch.stack([top_indices_x, top_indices_y], dim=-1)
        top_keypoints = top_indices.flip(2)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        top_descriptors = self.convDb(cDa)
        top_descriptors = F.normalize(top_descriptors, p=2, dim=1)

        # Extract descriptors at keypoint locations
        if torch.jit.is_tracing():  # type: ignore
            divisor = torch.stack([w, h]) * s - s / 2 - 0.5  # type: ignore
            divisor = divisor.to(device=top_keypoints.device)
        else:
            divisor = torch.tensor([w, h], device=top_keypoints.device) * s - s / 2 - 0.5
        normalized_keypoints = 2 * (top_keypoints - (s / 2 - 0.5)) / divisor - 1
        top_descriptors = F.grid_sample(
            top_descriptors, normalized_keypoints[:, None], mode="bilinear", align_corners=True
        ).reshape(b, self.descriptor_dim, self.num_keypoints)
        top_descriptors = F.normalize(top_descriptors, p=2, dim=1).permute(0, 2, 1)

        return (
            top_keypoints,  # (B, N, 2) with <X, Y>
            top_scores,  # (B, N)
            top_descriptors,  # (B, N, descriptor_dim)
        )
