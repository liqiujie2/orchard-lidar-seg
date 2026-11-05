import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import LaserScan, SemLaserScan

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class SemanticKitti(Dataset):
    def __init__(self, root,
                 sequences,
                 labels,
                 color_map,
                 learning_map,
                 learning_map_inv,
                 sensor,
                 max_points=150000,
                 gt=True,
                 augment=False,
                 rot_degree=5.0):
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
        self.gt = gt
        self.augment = augment
        self.rot_degree = rot_degree

        self.nclasses = len(self.learning_map_inv)

        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        assert(isinstance(self.labels, dict))
        assert(isinstance(self.color_map, dict))
        assert(isinstance(self.learning_map, dict))
        assert(isinstance(self.sequences, list))

        self.scan_files = []
        self.label_files = []

        for seq in self.sequences:
            seq = '{0:02d}'.format(int(seq))
            print("parsing seq {}".format(seq))

            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")

            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_path)) for f in fn if is_label(f)]

            if self.gt:
                assert(len(scan_files) == len(label_files))

            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)

        self.scan_files.sort()
        self.label_files.sort()

        print("Using {} scans from sequences {}".format(len(self.scan_files), self.sequences))
        print("=============================================")
        print("DATASET CONFIG: Intensity using per-frame LOCAL Z-SCORE normalization")
        print("几何通道(range/x/y/z): global Z-score | 强度通道(remission): 逐帧局部Z-score")
        print(f"Online Augment: {'ENABLE' if self.augment else 'DISABLE'} | Rot range ±{self.rot_degree} deg")
        print("=============================================\n")

    @staticmethod
    def get_rot_matrix(deg, axis):
        rad = np.radians(deg)
        if axis == "x":
            mat = np.array([
                [1, 0, 0],
                [0, np.cos(rad), -np.sin(rad)],
                [0, np.sin(rad), np.cos(rad)]
            ], dtype=np.float32)
        elif axis == "y":
            mat = np.array([
                [np.cos(rad), 0, np.sin(rad)],
                [0, 1, 0],
                [-np.sin(rad), 0, np.cos(rad)]
            ], dtype=np.float32)
        elif axis == "z":
            mat = np.array([
                [np.cos(rad), -np.sin(rad), 0],
                [np.sin(rad), np.cos(rad), 0],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            mat = np.eye(3, dtype=np.float32)
        return mat

    def __getitem__(self, index):
        scan_file = self.scan_files[index]
        if self.gt:
            label_file = self.label_files[index]

        if self.gt:
            scan = SemLaserScan(self.color_map,
                                project=True,
                                H=self.sensor_img_H,
                                W=self.sensor_img_W,
                                fov_up=self.sensor_fov_up,
                                fov_down=self.sensor_fov_down)
        else:
            scan = LaserScan(project=True,
                             H=self.sensor_img_H,
                             W=self.sensor_img_W,
                             fov_up=self.sensor_fov_up,
                             fov_down=self.sensor_fov_down)

        scan.open_scan(scan_file)

        if self.augment:
            pts = scan.points.copy()
            rem = scan.remissions.copy()
            axis = np.random.choice(["x", "y", "z"])
            angle = np.random.uniform(-self.rot_degree, self.rot_degree)
            rot_mat = SemanticKitti.get_rot_matrix(angle, axis)
            pts = pts @ rot_mat
            scan.points = pts
            scan.remissions = rem
            scan.unproj_range = np.linalg.norm(pts, axis=1)

        if self.gt:
            scan.open_label(label_file)
            scan.sem_label = self.map(scan.sem_label, self.learning_map)
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        if self.gt:
            unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        mask_float = proj_mask.float()

        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []

        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

        ch_range = proj_range.unsqueeze(0)
        ch_xyz = proj_xyz.permute(2, 0, 1)
        ch_rem = proj_remission.unsqueeze(0)

        four_chan = torch.cat([ch_range, ch_xyz], dim=0)
        mean_4 = self.sensor_img_means[:4]
        std_4 = self.sensor_img_stds[:4]
        four_chan = (four_chan - mean_4[:, None, None]) / std_4[:, None, None]

        mask_3d = mask_float.unsqueeze(0)
        valid_rem = ch_rem[mask_3d > 0]
        if len(valid_rem) > 0:
            r_mean = valid_rem.mean()
            r_std = valid_rem.std()
            if r_std > 1e-6:
                ch_rem = (ch_rem - r_mean) / r_std
            else:
                ch_rem = ch_rem - r_mean

        proj = torch.cat([four_chan, ch_rem], dim=0)
        proj = proj * mask_3d

        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        return (
            proj,
            proj_mask,
            proj_labels,
            unproj_labels,
            path_seq,
            path_name,
            proj_x,
            proj_y,
            proj_range,
            unproj_range,
            proj_xyz,
            unproj_xyz,
            proj_remission,
            unproj_remissions,
            unproj_n_points
        )

    def __len__(self):
        return len(self.scan_files)

    @staticmethod
    def map(label, mapdict):
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        return lut[label]


class Parser():
    def __init__(self,
                 root,
                 train_sequences,
                 valid_sequences,
                 test_sequences,
                 labels,
                 color_map,
                 learning_map,
                 learning_map_inv,
                 sensor,
                 max_points,
                 batch_size,
                 workers,
                 gt=True,
                 shuffle_train=True):
        super(Parser, self).__init__()

        self.root = root
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.max_points = max_points
        self.batch_size = batch_size
        self.workers = workers
        self.gt = gt
        self.shuffle_train = shuffle_train
        self.nclasses = len(self.learning_map_inv)

        self.train_dataset = SemanticKitti(root=self.root,
                                           sequences=self.train_sequences,
                                           labels=self.labels,
                                           color_map=self.color_map,
                                           learning_map=self.learning_map,
                                           learning_map_inv=self.learning_map_inv,
                                           sensor=self.sensor,
                                           max_points=max_points,
                                           gt=self.gt,
                                           augment=True)

        self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=self.shuffle_train,
                                                      num_workers=self.workers,
                                                      pin_memory=True,
                                                      drop_last=True)
        assert len(self.trainloader) > 0
        self.trainiter = iter(self.trainloader)

        self.valid_dataset = SemanticKitti(root=self.root,
                                           sequences=self.valid_sequences,
                                           labels=self.labels,
                                           color_map=self.color_map,
                                           learning_map=self.learning_map,
                                           learning_map_inv=self.learning_map_inv,
                                           sensor=self.sensor,
                                           max_points=max_points,
                                           gt=self.gt,
                                           augment=False)

        self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=self.workers,
                                                      pin_memory=True,
                                                      drop_last=True)
        assert len(self.validloader) > 0
        self.validiter = iter(self.validloader)

        if self.test_sequences:
            self.test_dataset = SemanticKitti(root=self.root,
                                              sequences=self.test_sequences,
                                              labels=self.labels,
                                              color_map=self.color_map,
                                              learning_map=self.learning_map,
                                              learning_map_inv=self.learning_map_inv,
                                              sensor=self.sensor,
                                              max_points=max_points,
                                              gt=False,
                                              augment=False)

            self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                         batch_size=self.batch_size,
                                                         shuffle=False,
                                                         num_workers=self.workers,
                                                         pin_memory=True,
                                                         drop_last=True)
            assert len(self.testloader) > 0
            self.testiter = iter(self.testloader)

    def get_train_batch(self):
        scans = self.trainiter.next()
        return scans

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        scans = self.testiter.next()

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)

    def get_n_classes(self):
        return self.nclasses

    def get_original_class_string(self, idx):
        return self.labels[idx]

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def to_original(self, label):
        return SemanticKitti.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        return SemanticKitti.map(label, self.learning_map)

    def to_color(self, label):
        label = SemanticKitti.map(label, self.learning_map_inv)
        return SemanticKitti.map(label, self.color_map)
