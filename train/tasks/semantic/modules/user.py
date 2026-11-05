#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import imp
import time
import os
import numpy as np

import __init__ as booger
from tasks.semantic.modules.segmentator import *
from tasks.semantic.postproc.KNN import KNN


class User():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir):
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir

        self.global_total_proj = 0.0
        self.global_total_seg = 0.0
        self.global_total_knn = 0.0
        self.global_total_inv_proj = 0.0
        self.global_frame_count = 0

        self.warmup_frames = 20

        parser_path = os.path.join(booger.TRAIN_PATH, 'tasks/semantic/dataset', self.DATA["name"], 'parser.py')
        print("Loading parser from:", parser_path)
        parserModule = imp.load_source("parserModule", parser_path)

        self.parser = parserModule.Parser(
            root=self.datadir,
            train_sequences=self.DATA["split"]["train"],
            valid_sequences=self.DATA["split"]["valid"],
            test_sequences=self.DATA["split"]["test"],
            labels=self.DATA["labels"],
            color_map=self.DATA["color_map"],
            learning_map=self.DATA["learning_map"],
            learning_map_inv=self.DATA["learning_map_inv"],
            sensor=self.ARCH["dataset"]["sensor"],
            max_points=self.ARCH["dataset"]["max_points"],
            batch_size=1,
            workers=self.ARCH["train"]["workers"],
            gt=True,
            shuffle_train=False
        )

        with torch.no_grad():
            self.model = Segmentator(self.ARCH, self.parser.get_n_classes(), self.modeldir)

        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"], self.parser.get_n_classes())

        self.gpu = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Inferring in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        sensor_cfg = self.ARCH["dataset"]["sensor"]
        img_h = sensor_cfg["img_prop"]["height"]
        img_w = sensor_cfg["img_prop"]["width"]
        print("\n" + "=" * 60)
        print(f"📋 YAML配置球面投影分辨率：H = {img_h}, W = {img_w}")
        print(f"📋 Projection image size (H × W): {img_h} × {img_w}")
        print("=" * 60 + "\n")

    def bench_projection(self, dataset, k=100, warmup=5, seed=0):
        from common.laserscan import SemLaserScan
        idxs = np.random.RandomState(seed).choice(
            len(dataset), size=min(k, len(dataset)), replace=False)
        times = []
        for j, i in enumerate(idxs):
            scan_file = dataset.scan_files[i]
            raw = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))
            scan = SemLaserScan(dataset.color_map, project=True,
                                H=dataset.sensor_img_H, W=dataset.sensor_img_W,
                                fov_up=dataset.sensor_fov_up,
                                fov_down=dataset.sensor_fov_down)
            scan.points = raw[:, :3].astype(np.float32)
            scan.remissions = raw[:, 3].astype(np.float32)
            t0 = time.perf_counter()
            scan.do_range_projection()
            dt = (time.perf_counter() - t0) * 1000.0
            if j >= warmup:
                times.append(dt)
        a = np.asarray(times)
        return a.mean(), a.std(ddof=1), a.min(), len(a)

    def infer(self):
        print("\nStarting inference on TEST SET...")
        test_times, test_proj, test_seg, test_knn, test_inv_proj = self.infer_subset(
            loader=self.parser.get_test_set(),
            to_orig_fn=self.parser.to_original
        )
        self.global_total_proj += test_proj
        self.global_total_seg += test_seg
        self.global_total_knn += test_knn
        self.global_total_inv_proj += test_inv_proj
        self.global_frame_count += len(test_times)

        params_m = self.total_params / 1000000

        if self.global_frame_count > 0:
            avg_proj_ms = round((self.global_total_proj / self.global_frame_count) * 1000, 1)
            avg_seg_ms = round((self.global_total_seg / self.global_frame_count) * 1000, 1)
            avg_knn_ms = round((self.global_total_knn / self.global_frame_count) * 1000, 1)
            avg_inv_ms = round((self.global_total_inv_proj / self.global_frame_count) * 1000, 1)

            total_avg_ms = avg_proj_ms + avg_seg_ms + avg_knn_ms + avg_inv_ms
            total_avg_s = total_avg_ms / 1000
            fps = round(1.0 / total_avg_s, 1)

            arr = np.asarray(test_times) * 1000.0
            mean_ms = arr.mean()
            std_ms = arr.std(ddof=1)
            min_ms = arr.min()
            n_frames = len(test_times)
            fps_mean = 1000.0 / mean_ms

            print("\n" + "=" * 80)
            print("【Test Set Metrics】 Params | Pure Compute Time(ms) | FPS")
            print("=" * 80)
            print(f"Total Params (M): {params_m:.1f}")
            print(f"Tensor preprocess (proj)      : {avg_proj_ms} ms")
            print(f"Model segmentation infer      : {avg_seg_ms} ms")
            print(f"KNN boundary optimization     : {avg_knn_ms} ms")
            print(f"2D->3D inverse projection     : {avg_inv_ms} ms")
            print(f"Total pure compute time       : {total_avg_ms} ms")
            print(f"FPS (pure compute)            : {fps}")
            print("-" * 80)
            print(f"Inference protocol: batch=1, FP32, warm-up={self.warmup_frames}, "
                  f"mean±std over {n_frames} frames (TEST set)")
            print(f"Per-frame total latency       : {mean_ms:.2f} ± {std_ms:.2f} ms  "
                  f"(min {min_ms:.2f})")
            print(f"FPS (mean latency)            : {fps_mean:.1f}")
            print(f"FPS (best/min latency)        : {1000.0 / min_ms:.1f}")

            proj_m, proj_s, proj_min, proj_cnt = self.bench_projection(self.parser.test_dataset)
            total_p1 = mean_ms + proj_m
            fps_p1 = 1000.0 / total_p1
            print(f"Spherical projection (P1 bench): {proj_m:.2f} ± {proj_s:.2f} ms "
                  f"({proj_cnt} scans)")
            print(f"Total latency (P1 end-to-end)  : {total_p1:.2f} ms")
            print(f"FPS (P1 end-to-end)            : {fps_p1:.1f}")
            print("=" * 80)
        else:
            print("\nNo frame data recorded.")

        print('\nFinished Inferring')

    def infer_subset(self, loader, to_orig_fn, warmup=None):
        warmup = self.warmup_frames if warmup is None else warmup
        self.model.eval()
        if self.gpu:
            torch.cuda.empty_cache()

        total_proj_time = 0.0
        total_seg_time = 0.0
        total_knn_time = 0.0
        total_inv_time = 0.0
        inference_times = []
        warmup_done = 0
        print_tensor_shape = True

        with torch.no_grad():
            for i, data in enumerate(loader):
                proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, \
                    proj_range, unproj_range, _, _, _, _, npoints = data

                n = int(npoints.flatten()[0])

                p_x = p_x[0, :n]
                p_y = p_y[0, :n]
                proj_range = proj_range[0, :n]
                unproj_range = unproj_range[0, :n]
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()
                    p_x = p_x.cuda()
                    p_y = p_y.cuda()
                    if self.post:
                        proj_range = proj_range.cuda()
                        unproj_range = unproj_range.cuda()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_proj = time.time()
                proj_in = proj_in.float()
                proj_mask = proj_mask.bool()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_proj = time.time() - start_proj

                if print_tensor_shape:
                    print("\n" + "-"*70)
                    print(f"📐 网络输入 proj_in shape: {proj_in.shape}")
                    print(f"📐 输入mask proj_mask shape: {proj_mask.shape}")

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_seg = time.time()
                proj_output = self.model(proj_in, proj_mask)
                proj_argmax = proj_output[0].argmax(dim=0)

                if print_tensor_shape:
                    print(f"📐 model输出proj_output[0] shape: {proj_output[0].shape}")
                    print(f"📐 argmax后 proj_argmax shape: {proj_argmax.shape}")
                    print("-"*70 + "\n")
                    print_tensor_shape = False

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_seg = time.time() - start_seg

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_knn = time.time()
                if self.post:
                    unproj_argmax = self.post(proj_range, unproj_range, proj_argmax, p_x, p_y)
                else:
                    unproj_argmax = proj_argmax
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_knn = time.time() - start_knn

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_inv = time.time()
                if not self.post:
                    unproj_argmax = unproj_argmax[p_y, p_x]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_inv = time.time() - start_inv

                pred_np = unproj_argmax.cpu().numpy().reshape((-1)).astype(np.int32)
                pred_np = to_orig_fn(pred_np)
                pred_np = pred_np[:n]
                save_dir = os.path.join(self.logdir, "sequences", path_seq, "predictions")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, path_name)
                pred_np.tofile(save_path)

                if warmup_done < warmup:
                    warmup_done += 1
                    continue

                total_proj_time += t_proj
                total_seg_time += t_seg
                total_knn_time += t_knn
                total_inv_time += t_inv

                total_frame_t = t_proj + t_seg + t_knn + t_inv
                inference_times.append(total_frame_t)

                print(
                    f"{path_seq} {path_name} | 预处理:{t_proj:.4f}s 推理:{t_seg:.4f}s "
                    f"KNN:{t_knn:.4f}s 逆投影:{t_inv:.4f}s 总纯运算:{total_frame_t:.4f}s")

        return inference_times, total_proj_time, total_seg_time, total_knn_time, total_inv_time
