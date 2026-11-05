#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import __init__ as booger


class KNN(nn.Module):
    def __init__(self, params, nclasses):
        super().__init__()
        print("*" * 80)
        print("Cleaning point-clouds with kNN post-processing | 3D Euclidean distance VLP-16")
        self.knn = params["knn"]

        self.search_h = 3
        self.search_w = 15

        self.sigma = params["sigma"]
        self.cutoff = params["cutoff"]
        self.nclasses = nclasses

        self.ang_res_x = math.radians(0.2)
        self.ang_res_y = math.radians(2.0)

        print("kNN parameters (固定窗口):")
        print("knn:", self.knn)
        print(f"窗口: {self.search_h} × {self.search_w}")
        print("sigma:", self.sigma)
        print("cutoff:", self.cutoff)
        print("nclasses:", self.nclasses)
        print("VLP16 ang_x(rad):", round(self.ang_res_x,6), " ang_y(rad):", round(self.ang_res_y,6))
        print("*" * 80)

    def pixel_to_xyz(self, r, px_coord, py_coord):
        phi = px_coord * self.ang_res_x
        theta = py_coord * self.ang_res_y
        x = r * torch.cos(theta) * torch.cos(phi)
        y = r * torch.cos(theta) * torch.sin(phi)
        z = r * torch.sin(theta)
        return x, y, z

    def forward(self, proj_range, unproj_range, proj_argmax, px, py):
        if proj_range.is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        H, W = proj_range.shape
        P = unproj_range.shape

        pad_h = int((self.search_h - 1) / 2)
        pad_w = int((self.search_w - 1) / 2)

        proj_unfold_k_rang = F.unfold(proj_range[None, None, ...],
                                     kernel_size=(self.search_h, self.search_w),
                                     padding=(pad_h, pad_w))

        idx_list = py * W + px
        unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]
        unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float("inf")
        center = int(((self.search_h * self.search_w) - 1) / 2)
        unproj_unfold_k_rang[:, center, :] = unproj_range

        px_map = torch.arange(W, device=device).repeat(H, 1)[None, None, :, :].float()
        py_map = torch.arange(H, device=device).repeat(W, 1).t()[None, None, :, :].float()
        unfold_px = F.unfold(px_map, kernel_size=(self.search_h, self.search_w), padding=(pad_h, pad_w))[:, :, idx_list]
        unfold_py = F.unfold(py_map, kernel_size=(self.search_h, self.search_w), padding=(pad_h, pad_w))[:, :, idx_list]

        center_r = unproj_range[None, :]
        center_px = px[None, :]
        center_py = py[None, :]

        neigh_r = unproj_unfold_k_rang
        neigh_px = unfold_px
        neigh_py = unfold_py

        xc, yc, zc = self.pixel_to_xyz(center_r, center_px, center_py)
        xn, yn, zn = self.pixel_to_xyz(neigh_r, neigh_px, neigh_py)

        dx = xn - xc
        dy = yn - yc
        dz = zn - zc
        k2_distances = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)

        _, knn_idx = k2_distances.topk(self.knn, dim=1, largest=False, sorted=False)

        proj_unfold_1_argmax = F.unfold(proj_argmax[None, None, ...].float(),
                                        kernel_size=(self.search_h, self.search_w),
                                        padding=(pad_h, pad_w)).long()
        unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]

        knn_argmax = torch.gather(input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

        knn_distances = torch.gather(input=k2_distances, dim=1, index=knn_idx)
        knn_invalid_idx = knn_distances > self.cutoff
        knn_argmax[knn_invalid_idx] = self.nclasses

        gaussian_weights = torch.exp(- (knn_distances ** 2) / (2 * self.sigma ** 2))
        gaussian_weights[knn_invalid_idx] = 0.0

        knn_argmax_onehot = torch.zeros((1, self.nclasses + 1, P[0]), device=device).type(proj_range.type())
        knn_argmax_onehot.scatter_add_(1, knn_argmax, gaussian_weights)

        knn_argmax_out = knn_argmax_onehot[:, 1:-1].argmax(dim=1) + 1
        knn_argmax_out = knn_argmax_out.view(P)
        return knn_argmax_out
