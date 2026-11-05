#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np

class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, project=False, H=16, W=900, fov_up=15.0, fov_down=-15.0):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.reset()

  def reset(self):
    self.points = np.zeros((0, 3), dtype=np.float32)
    self.remissions = np.zeros((0, 1), dtype=np.float32)

    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)

    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    self.proj_x = np.zeros((0, 1), dtype=np.int32)
    self.proj_y = np.zeros((0, 1), dtype=np.int32)
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)

  def size(self):

    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    self.reset()

    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    points = scan[:, 0:3]
    remissions = scan[:, 3]
    self.set_points(points, remissions)

  def set_points(self, points, remissions=None):
    self.reset()
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    self.points = points
    if remissions is not None:
      self.remissions = remissions
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    if self.project:
      self.do_range_projection()

  def do_range_projection(self):
    phi_max = np.radians(89.0)
    phi_min = np.radians(-90.0)
    phi_span = phi_max - phi_min

    theta_max = np.radians(15.0)
    theta_min = np.radians(-15.0)
    theta_span = theta_max - theta_min

    H = self.proj_H
    W = self.proj_W

    depth = np.linalg.norm(self.points, 2, axis=1)
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    phi = np.arctan2(scan_y, scan_x)
    theta = np.arcsin(scan_z / depth)

    fov_mask = (phi >= phi_min) & (phi <= phi_max) & (theta >= theta_min) & (theta <= theta_max)

    self.proj_range.fill(-1.0)
    self.proj_xyz.fill(-1.0)
    self.proj_remission.fill(-1.0)
    self.proj_idx.fill(-1)
    self.proj_mask.fill(0)

    self.unproj_range = np.copy(depth)
    self.proj_x = np.zeros((self.points.shape[0],), dtype=np.int32)
    self.proj_y = np.zeros((self.points.shape[0],), dtype=np.int32)

    self.proj_x[~fov_mask] = -1
    self.proj_y[~fov_mask] = -1

    depth_fov = depth[fov_mask]
    phi_fov = phi[fov_mask]
    theta_fov = theta[fov_mask]
    pts_fov = self.points[fov_mask, :]
    rem_fov = self.remissions[fov_mask]
    idx_fov = np.arange(self.points.shape[0])[fov_mask]

    delta_phi = phi_span / W
    delta_theta = theta_span / H

    u_float = (phi_max - phi_fov) / delta_phi
    v_float = (theta_max - theta_fov) / delta_theta

    u_1based = np.floor(u_float).astype(np.int32) + 1
    v_1based = np.floor(v_float).astype(np.int32) + 1

    u_1based = np.clip(u_1based, 1, W)
    v_1based = np.clip(v_1based, 1, H)

    proj_x = u_1based - 1
    proj_y = v_1based - 1

    self.proj_x[fov_mask] = proj_x
    self.proj_y[fov_mask] = proj_y

    order = np.argsort(depth_fov)[::-1]
    depth_fov = depth_fov[order]
    proj_x = proj_x[order]
    proj_y = proj_y[order]
    pts_fov = pts_fov[order]
    rem_fov = rem_fov[order]
    idx_fov = idx_fov[order]

    self.proj_range[proj_y, proj_x] = depth_fov

    points_abs_y = pts_fov.copy()
    points_abs_y[:, 1] = np.abs(points_abs_y[:, 1])
    self.proj_xyz[proj_y, proj_x] = points_abs_y

    self.proj_remission[proj_y, proj_x] = rem_fov
    self.proj_idx[proj_y, proj_x] = idx_fov
    self.proj_mask = (self.proj_idx >= 0).astype(np.int32)


class SemLaserScan(LaserScan):

  EXTENSIONS_LABEL = ['.label']

  def __init__(self,  sem_color_dict=None, project=False, H=16, W=900, fov_up=15.0, fov_down=-15.0, max_classes=300):
    super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
    self.reset()

    if sem_color_dict:
      max_sem_key = 0
      for key, data in sem_color_dict.items():
        if key + 1 > max_sem_key:
          max_sem_key = key + 1
      self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
      for key, value in sem_color_dict.items():
        self.sem_color_lut[key] = np.array(value, np.float32) / 255.0
    else:
      max_sem_key = max_classes
      self.sem_color_lut = np.random.uniform(low=0.0,
                                             high=1.0,
                                             size=(max_sem_key, 3))
      self.sem_color_lut[0] = np.full((3), 0.1)

    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    self.inst_color_lut[0] = np.full((3), 0.1)

  def reset(self):

    super(SemLaserScan, self).reset()

    self.sem_label = np.zeros((0, 1), dtype=np.int32)
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)

    self.inst_label = np.zeros((0, 1), dtype=np.int32)
    self.inst_label_color = np.zeros((0, 3), dtype=np.float32)

    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                   dtype=np.int32)
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                   dtype=np.float)

    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                    dtype=np.int32)
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                    dtype=np.float)

  def open_label(self, filename):

    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    label = np.fromfile(filename, dtype=np.int32)
    label = label.reshape((-1))

    self.set_label(label)

  def set_label(self, label):

    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    if label.shape[0] == self.points.shape[0]:
      self.sem_label = label & 0xFFFF
      self.inst_label = label >> 16
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")

    assert((self.sem_label + (self.inst_label << 16) == label).all())

    if self.project:
      self.do_label_projection()

  def colorize(self):

    self.sem_label_color = self.sem_color_lut[self.sem_label]
    self.sem_label_color = self.sem_label_color.reshape((-1, 3))

    self.inst_label_color = self.inst_color_lut[self.inst_label]
    self.inst_label_color = self.inst_label_color.reshape((-1, 3))

  def do_label_projection(self):
    mask = self.proj_idx >= 0

    self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
    self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

    self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
    self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]