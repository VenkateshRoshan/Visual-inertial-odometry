#!/usr/bin/env python3
"""
DataLoader.py

Core module for the drone VIO project. It holds everything that Train.py and
Test.py both need, so that the preprocessing used at training time is exactly
the preprocessing used at inference time:

  * CONFIG        : one dictionary of knobs you can play with.
  * pose math     : quaternion and euler helpers, relative pose, integration.
  * preprocessing : image transform, IMU resampling and normalisation.
  * VIODataset    : reads the folders produced by the data collector.
  * utilities     : discover sequences, compute IMU normalisation stats.

Dataset layout expected (produced by the collector script):
    <dataset_root>/<run_name>/<path_name>/images/frame_XXXXXX.png
    <dataset_root>/<run_name>/<path_name>/samples.csv
    <dataset_root>/<run_name>/<path_name>/imu.csv

samples.csv columns:
    image_id, image_timestamp, image_path, pose_timestamp,
    x, y, z, qx, qy, qz, qw,
    imu_start_timestamp, imu_end_timestamp, imu_start_index, imu_end_index
"""

import os
import csv
import glob
import math

import numpy as np

import torch
from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms as T


# ===========================================================================
# CONFIG  (change these to experiment)
# ===========================================================================

CONFIG = {
    # ---- data ----
    "dataset_root": "vio_dataset",   # root that contains the run folders
    "stack_size": 4,                 # last 3 + current frame
    "window_stride": 1,              # 1 = densest set of training windows
    "image_size": [224, 224],        # H, W fed to the CNN
    "imu_seq_len": 50,               # IMU samples per window after resampling
    "imu_channels": 6,               # ax, ay, az, gx, gy, gz

    # ---- model ----
    "pretrained_resnet": True,
    "visual_embed_dim": 256,         # per frame feature size after projection
    "imu_hidden": 128,               # GRU hidden size
    "imu_layers": 2,                 # GRU layers
    "rnn_type": "gru",               # "gru" or "lstm"
    "fusion_hidden": 512,            # MLP hidden size in the pose head
    "output_dim": 6,                 # dx, dy, dz, droll, dpitch, dyaw

    # ---- training ----
    "batch_size": 16,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "rot_weight": 10.0,              # rotation loss is in radians, scale it up
    "val_fraction": 0.2,
    "num_workers": 4,
    "seed": 42,
    "checkpoint": "vio_model.pt",

    # ---- runtime ----
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ---- test / gazebo ----
    "camera": "bottom",              # bottom or front image topic
    "test_path": "square",           # which trajectory to fly during the test
    "test_control_rate": 20.0,       # Hz for velocity commands
}

# ImageNet statistics, since the ResNet backbone is pretrained on them.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ===========================================================================
# Pose math
# ===========================================================================

def quat_to_matrix(qx, qy, qz, qw):
    """Unit quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def matrix_to_euler(R):
    """Rotation matrix to (roll, pitch, yaw), ZYX intrinsic convention."""
    sy = math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], sy)
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def euler_to_matrix(roll, pitch, yaw):
    """(roll, pitch, yaw) to a rotation matrix, R = Rz @ Ry @ Rx."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def relative_pose(p_a, R_a, p_b, R_b):
    """
    Relative motion from frame A to frame B, expressed in A's body frame.
    Returns a length 6 vector: dx, dy, dz, droll, dpitch, dyaw.
    """
    dp = R_a.T @ (p_b - p_a)
    R_rel = R_a.T @ R_b
    roll, pitch, yaw = matrix_to_euler(R_rel)
    return np.array([dp[0], dp[1], dp[2], roll, pitch, yaw], dtype=np.float32)


def integrate_pose(R_est, p_est, pred6):
    """
    Chain a predicted relative pose onto the running estimate.
    pred6 = dx, dy, dz, droll, dpitch, dyaw in the previous body frame.
    Returns updated R_est, p_est.
    """
    dp = np.array(pred6[:3], dtype=np.float64)
    R_rel = euler_to_matrix(pred6[3], pred6[4], pred6[5])
    p_new = p_est + R_est @ dp
    R_new = R_est @ R_rel
    return R_new, p_new


# ===========================================================================
# Preprocessing (identical for training and live inference)
# ===========================================================================

def build_image_transform(image_size):
    return T.Compose([
        T.Resize((int(image_size[0]), int(image_size[1]))),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def bgr_to_input_tensor(img_bgr, transform):
    """Convert an OpenCV BGR image (numpy) into a normalised CHW tensor."""
    img_rgb = img_bgr[:, :, ::-1]                 # BGR to RGB
    pil = Image.fromarray(np.ascontiguousarray(img_rgb))
    return transform(pil)


def load_image_tensor(path, transform):
    pil = Image.open(path).convert("RGB")
    return transform(pil)


def resample_imu(imu_data, seq_len):
    """
    Resample a variable length IMU block (M, C) to a fixed length (seq_len, C)
    by linear interpolation along time. Handles empty and single row inputs.
    """
    imu_data = np.asarray(imu_data, dtype=np.float32)
    channels = imu_data.shape[1] if imu_data.ndim == 2 else CONFIG["imu_channels"]

    if imu_data.ndim != 2 or imu_data.shape[0] == 0:
        return np.zeros((seq_len, channels), dtype=np.float32)
    if imu_data.shape[0] == 1:
        return np.repeat(imu_data, seq_len, axis=0).astype(np.float32)

    src = np.linspace(0.0, 1.0, imu_data.shape[0])
    dst = np.linspace(0.0, 1.0, seq_len)
    out = np.zeros((seq_len, channels), dtype=np.float32)
    for c in range(channels):
        out[:, c] = np.interp(dst, src, imu_data[:, c])
    return out


def normalize_imu(imu_seq, mean, std):
    if mean is None or std is None:
        return imu_seq.astype(np.float32)
    return ((imu_seq - mean) / std).astype(np.float32)


# ===========================================================================
# Sequence discovery and IMU statistics
# ===========================================================================

def discover_sequences(dataset_root):
    """Every samples.csv under the root is one continuous trajectory."""
    pattern = os.path.join(dataset_root, "**", "samples.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def _read_samples(samples_csv):
    """Read a samples.csv into a dict of numpy arrays plus the run root path."""
    rows = []
    with open(samples_csv, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for r in reader:
            if len(r) < 11:
                continue
            rows.append(r)

    run_root = os.path.dirname(os.path.dirname(samples_csv))  # .../<run_name>
    data = {
        "run_root": run_root,
        "image_path": [r[2] for r in rows],
        "image_ts": np.array([float(r[1]) for r in rows], dtype=np.float64),
        "pos": np.array([[float(r[4]), float(r[5]), float(r[6])] for r in rows], dtype=np.float64),
        "quat": np.array([[float(r[7]), float(r[8]), float(r[9]), float(r[10])] for r in rows], dtype=np.float64),
    }
    return data


def _read_imu(imu_csv):
    """Read imu.csv into timestamps (M,) and the 6 motion channels (M, 6)."""
    ts, data = [], []
    if not os.path.isfile(imu_csv):
        return np.zeros((0,)), np.zeros((0, 6), dtype=np.float32)
    with open(imu_csv, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            if len(r) < 7:
                continue
            ts.append(float(r[0]))
            data.append([float(r[1]), float(r[2]), float(r[3]),
                         float(r[4]), float(r[5]), float(r[6])])
    return np.array(ts, dtype=np.float64), np.array(data, dtype=np.float32)


def compute_imu_stats(sequence_csv_list):
    """Mean and std over all IMU channels across the given sequences."""
    total = None
    total_sq = None
    count = 0
    for samples_csv in sequence_csv_list:
        imu_csv = os.path.join(os.path.dirname(samples_csv), "imu.csv")
        _, imu = _read_imu(imu_csv)
        if imu.shape[0] == 0:
            continue
        s = imu.sum(axis=0)
        sq = (imu ** 2).sum(axis=0)
        total = s if total is None else total + s
        total_sq = sq if total_sq is None else total_sq + sq
        count += imu.shape[0]

    if count == 0:
        return (np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32))

    mean = total / count
    var = np.maximum(total_sq / count - mean ** 2, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


# ===========================================================================
# Dataset
# ===========================================================================

class _Sequence:
    """One trajectory: cached samples, IMU arrays and its list of windows."""

    def __init__(self, samples_csv, stack_size, stride):
        s = _read_samples(samples_csv)
        imu_ts, imu_data = _read_imu(os.path.join(os.path.dirname(samples_csv), "imu.csv"))

        self.run_root = s["run_root"]
        self.image_path = s["image_path"]
        self.image_ts = s["image_ts"]
        self.pos = s["pos"]
        self.quat = s["quat"]
        self.imu_ts = imu_ts
        self.imu_data = imu_data

        n = len(self.image_path)
        self.windows = list(range(0, max(0, n - stack_size + 1), stride))


class VIODataset(Dataset):
    """
    Yields, for each window of `stack_size` frames:
        images : (N, 3, H, W) tensor
        imu    : (L, 6) tensor
        target : (6,) relative pose from the first to the last frame
    """

    def __init__(self, sequence_csv_list, config, imu_mean=None, imu_std=None):
        self.cfg = config
        self.N = config["stack_size"]
        self.L = config["imu_seq_len"]
        self.imu_mean = imu_mean
        self.imu_std = imu_std
        self.transform = build_image_transform(config["image_size"])

        self.sequences = []
        self.index = []  # (sequence_idx, window_start)
        for csv_path in sequence_csv_list:
            seq = _Sequence(csv_path, self.N, config["window_stride"])
            if not seq.windows:
                continue
            si = len(self.sequences)
            self.sequences.append(seq)
            for start in seq.windows:
                self.index.append((si, start))

    def __len__(self):
        return len(self.index)

    def _imu_between(self, seq, t0, t1):
        if seq.imu_ts.shape[0] == 0:
            return np.zeros((self.L, self.cfg["imu_channels"]), dtype=np.float32)
        lo = np.searchsorted(seq.imu_ts, t0, side="left")
        hi = np.searchsorted(seq.imu_ts, t1, side="right")
        block = seq.imu_data[lo:hi]
        seq_arr = resample_imu(block, self.L)
        return normalize_imu(seq_arr, self.imu_mean, self.imu_std)

    def __getitem__(self, idx):
        si, start = self.index[idx]
        seq = self.sequences[si]
        end = start + self.N - 1

        # image stack
        imgs = []
        for k in range(start, start + self.N):
            path = os.path.join(seq.run_root, seq.image_path[k])
            imgs.append(load_image_tensor(path, self.transform))
        images = torch.stack(imgs, dim=0)  # (N, 3, H, W)

        # IMU between first and last frame timestamps
        imu_seq = self._imu_between(seq, seq.image_ts[start], seq.image_ts[end])
        imu = torch.from_numpy(imu_seq)

        # relative pose label
        R_a = quat_to_matrix(*seq.quat[start])
        R_b = quat_to_matrix(*seq.quat[end])
        target = relative_pose(seq.pos[start], R_a, seq.pos[end], R_b)
        target = torch.from_numpy(target)

        return images, imu, target


# quick sanity check when run directly
if __name__ == "__main__":
    seqs = discover_sequences(CONFIG["dataset_root"])
    print(f"Found {len(seqs)} sequences under {CONFIG['dataset_root']}")
    if seqs:
        ds = VIODataset(seqs, CONFIG)
        print(f"Total training windows: {len(ds)}")
        if len(ds):
            img, imu, tgt = ds[0]
            print("images:", tuple(img.shape), "imu:", tuple(imu.shape), "target:", tgt.numpy())